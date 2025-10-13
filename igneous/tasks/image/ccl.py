"""
Perform connected components labeling on the image.

Result will be a 6-connected labeling of the input
image. All steps must use the same task shape. 

As each task uses a label offset to differentiate its
CCL labels from adjacent tasks, the largest possible
image that can be handled has 2^64 voxels which is
order of magnitude about size of a whole mouse brain.

Each of the steps are labeled with their sequence number.
Their order is:
  (1) Generate 3 back faces for each task with 
    1 voxel overlap (so they can be referenced by 
    adjacent tasks). [ CCLFacesTask ]
  (2) Compute linkages between CCL tasks and 
    save the results. [ CCLEquivalancesTask ]
  (3) Compute a global union find from the linkage 
    data and from that a global relabeling scheme which 
    is saved in the database. [ create_relabeling ]
  (4) Apply the relabeling scheme to the image. [ RelabelCCLTask ]
"""
from typing import Optional, Union
from collections import defaultdict

import cc3d
import fastremap
import numpy as np
import crackle

from tqdm import tqdm

from cloudfiles import CloudFiles
from taskqueue import queueable

from cloudvolume import CloudVolume, Bbox, Vec
from cloudvolume.lib import sip

from ...types import ShapeType

import kimimaro
from sklearn.neighbors import NearestNeighbors

__all__ = [
  "CCLFacesTask",
  "CCLEquivalancesTask",
  "RelabelCCLTask",
]

class DisjointSet:
  def __init__(self):
    self.data = {} 
  def makeset(self, x):
    self.data[x] = x
    return x
  def find(self, x):
    if not x in self.data:
      return None
    i = self.data[x]
    while i != self.data[i]:
      self.data[i] = self.data[self.data[i]]
      i = self.data[i]
    return i
  def union(self, x, y):
    i = self.find(x)
    j = self.find(y)
    if i is None:
      i = self.makeset(x)
    if j is None:
      j = self.makeset(y)

    if i < j:
      self.data[j] = i
    else:
      self.data[i] = j

def compute_task_number(grid_size, gridpoint) -> int:
  return int(
    gridpoint.x + grid_size.x * (
      gridpoint.y + grid_size.y * gridpoint.z
    )
  )

def compute_label_offset(shape, grid_size, gridpoint) -> int:
  # a task sequence number counting from 0 according to
  # where the task is located in space (so we can recompute 
  # it in the second pass easily)
  task_num = compute_task_number(grid_size, gridpoint)
  return task_num * shape.x * shape.y * shape.z

def threshold_image(
  image:np.ndarray, 
  threshold_lte:Optional[Union[int,float]], 
  threshold_gte:Optional[Union[int,float]]
) -> np.ndarray:
  if threshold_gte is None and threshold_lte is None:
    return image
  elif threshold_gte is None and threshold_lte is not None:
    return image <= threshold_lte
  elif threshold_gte is not None and threshold_lte is None:
    return image >= threshold_gte
  else:
    return (image >= threshold_gte) & (image <= threshold_lte)

def blackout_non_face_rails(
  labels:np.ndarray, shape:ShapeType
) -> np.ndarray:
  """
  For 6-connectivity, we need to black out the
  "rails" that would represent a higher connectivity
  to ensure that the labels are actually present
  when computing the relabeling later.
  """
  slcs = [
    np.s_[shape[0],shape[1],:],
    np.s_[shape[0],:,shape[2]],
    np.s_[:,shape[1],shape[2]]
  ]

  for slc in slcs:
    try:
      labels[slc] = 0
    except IndexError:
      pass

  return labels

def skeleton_based_connected_components(labels, out_dtype=np.uint64, return_N=False):
    """Deterministic version of skeleton-based CCL."""
    np.random.seed(42)
    random.seed(42)
    
    # Convert to binary
    if labels.dtype != np.bool_:
        binary_labels = labels > 0
    else:
        binary_labels = labels.copy()
    
    if not np.any(binary_labels):
        result = np.zeros_like(labels, dtype=out_dtype)
        return (result, 0) if return_N else result
    
    # Step 1: Skeletonize with deterministic settings, need to expose more params
    teasar_params = {
        'scale': 4,
        'const': 10,
        'pdrf_scale': 100000,
        'pdrf_exponent': 4,
        'soma_acceptance_threshold': 3500,
        'soma_detection_threshold': 1100,
        'soma_invalidation_scale': 1.0,
        'soma_invalidation_const': 300
    }
    
    skeletons_dict = kimimaro.skeletonize(
        binary_labels, 
        teasar_params,
        dust_threshold=0,
        anisotropy=(1,1,1),
        fix_branching=True,
        fix_borders=True,
        progress=False,
        parallel=1  # Force single-threaded for determinism
    )
    
    # Step 2: Process skeletons in DETERMINISTIC ORDER
    component_skeletons = []
    next_id = 1
    
    # Sort by label for deterministic processing
    for label in sorted(skeletons_dict.keys()):
        skeleton = skeletons_dict[label]

        if len(skeleton.vertices) == 0 or len(skeleton.edges) == 0:
            continue
            
        # Step 2a: Split at branch points (your method)
        branch_nodes = skeleton.branches()
        
        if len(branch_nodes) > 0:
            # Create mask excluding branch nodes
            mask = np.ones(len(skeleton.vertices), dtype=bool)
            mask[branch_nodes] = False
            
            # Create mapping from old to new indices
            old_to_new_indices = np.full(len(skeleton.vertices), -1, dtype=int)
            old_to_new_indices[mask] = np.arange(np.sum(mask))
            
            # Clone and modify skeleton
            split_skeleton = skeleton.clone()
            split_skeleton.vertices = skeleton.vertices[mask]
            if hasattr(skeleton, 'radius') and skeleton.radius is not None:
                split_skeleton.radius = skeleton.radius[mask]
            
            # Remove edges containing branch nodes and remap
            valid_edges_mask = ~np.isin(skeleton.edges, branch_nodes).any(axis=1)
            if np.any(valid_edges_mask):
                valid_edges = skeleton.edges[valid_edges_mask]
                remapped_edges = []
                for edge in valid_edges:
                    new_v1 = old_to_new_indices[edge[0]]
                    new_v2 = old_to_new_indices[edge[1]]
                    if new_v1 >= 0 and new_v2 >= 0:
                        remapped_edges.append([new_v1, new_v2])
                
                if remapped_edges:
                    split_skeleton.edges = np.array(remapped_edges)
                else:
                    split_skeleton.edges = np.array([]).reshape(0, 2)
            else:
                split_skeleton.edges = np.array([]).reshape(0, 2)
            
            # Step 2b: Get components
            components = split_skeleton.components()
        else:
            components = skeleton.components()
        
        # Sort components by some deterministic criteria
        components = sorted(components, key=lambda c: tuple(c.vertices[0]) if len(c.vertices) > 0 else (0,0,0))
        
        for comp in components:
          if len(comp.vertices) > 0:
            comp.id = next_id
            component_skeletons.append(comp)
            next_id += 1
    
    # Step 3: Deterministic voxel relabeling
    cc_labels = relabel_voxels_deterministic(binary_labels, component_skeletons, out_dtype)
    unique_labels = np.unique(cc_labels)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    # Step 4: Ensure each component is contiguous, using cc3d for final pass
    final_labels = cc3d.connected_components(cc_labels, connectivity=6, out_dtype=out_dtype)
    unique_labels = np.unique(final_labels)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background
    print(f"Final number of connected components: {len(unique_labels)}")

    if return_N:
        N = len(unique_labels)
        return final_labels, N
    else:
        return final_labels

def relabel_voxels_deterministic(binary_img, skeletons, out_dtype):
    """Fully deterministic voxel relabeling."""
    labeled_img = np.zeros_like(binary_img, dtype=out_dtype)
    
    fg_coords = np.where(binary_img > 0)
    fg_points = np.column_stack(fg_coords)
    
    if len(fg_points) == 0 or len(skeletons) == 0:
        return labeled_img
    
    # Sort skeletons by ID for deterministic processing
    skeletons = sorted(skeletons, key=lambda s: s.id)
    
    # Sort vertices within each skeleton
    all_vertices = []
    vertex_labels = []
    
    for skel in skeletons:
      if len(skel.vertices) > 0:
        # Sort vertices by coordinates for tie-breaking consistency
        sorted_indices = np.lexsort(skel.vertices.T)
        sorted_vertices = skel.vertices[sorted_indices]
          
        all_vertices.append(sorted_vertices)
        vertex_labels.extend([skel.id] * len(sorted_vertices))
    
    all_vertices = np.vstack(all_vertices)
    vertex_labels = np.array(vertex_labels)
    
    # Use deterministic KNN algorithm
    knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')  # Fixed algorithm
    knn.fit(all_vertices)
    
    distances, indices = knn.kneighbors(fg_points)
    
    # Process voxels in deterministic order
    sorted_fg_indices = np.lexsort(fg_points.T)
    
    for i in sorted_fg_indices:
        coord = fg_points[i]
        idx = indices[i, 0]
        labeled_img[tuple(coord)] = vertex_labels[idx]
    
    return labeled_img

def skeleton_based_connected_components_with_oversegment(labels, out_dtype=np.uint64, return_N=False):
    """
    Multi-step skeleton-based CCL using oversegment segments attribute:
    1. Skeletonize
    2. Remove branch nodes from skeletons
    3. Use branch-removed skeletons for oversegment to create supervoxels
    4. Split skeleton components and use segments attribute for accurate regrouping
    """
    
    # Set seeds for determinism
    np.random.seed(42)
    random.seed(42)
    
    # Convert to binary
    if labels.dtype != np.bool_:
        binary_labels = labels > 0
    else:
        binary_labels = labels.copy()
    
    if not np.any(binary_labels):
        result = np.zeros_like(labels, dtype=out_dtype)
        return (result, 0) if return_N else result
    
    # Step 1: Skeletonize with deterministic settings
    teasar_params = {
        'scale': 4,
        'const': 10,
        'pdrf_scale': 100000,
        'pdrf_exponent': 4,
        'soma_acceptance_threshold': 3500,
        'soma_detection_threshold': 1100,
        'soma_invalidation_scale': 1.0,
        'soma_invalidation_const': 300
    }
    
    skeletons_dict = kimimaro.skeletonize(
        binary_labels, 
        teasar_params,
        dust_threshold=0,
        anisotropy=(1,1,1),
        fix_branching=True,
        fix_borders=True,
        progress=False,
        parallel=1  # Force single-threaded for determinism
    )
    
    if not skeletons_dict:
        result = np.zeros_like(labels, dtype=out_dtype)
        return (result, 0) if return_N else result
    
    # Step 2: Remove branch nodes from skeletons
    branch_removed_skeletons = []
    
    for label in sorted(skeletons_dict.keys()):
        skeleton = skeletons_dict[label]
        
        if len(skeleton.vertices) == 0 or len(skeleton.edges) == 0:
            continue
            
        # Remove branch nodes
        branch_nodes = skeleton.branches()
        
        if len(branch_nodes) > 0:
            # Create mask excluding branch nodes
            mask = np.ones(len(skeleton.vertices), dtype=bool)
            mask[branch_nodes] = False
            
            # Create mapping from old to new indices
            old_to_new_indices = np.full(len(skeleton.vertices), -1, dtype=int)
            old_to_new_indices[mask] = np.arange(np.sum(mask))
            
            # Clone and modify skeleton
            split_skeleton = skeleton.clone()
            split_skeleton.vertices = skeleton.vertices[mask]
            if hasattr(skeleton, 'radius') and skeleton.radius is not None:
                split_skeleton.radius = skeleton.radius[mask]
            
            # Remove edges containing branch nodes and remap
            valid_edges_mask = ~np.isin(skeleton.edges, branch_nodes).any(axis=1)
            if np.any(valid_edges_mask):
                valid_edges = skeleton.edges[valid_edges_mask]
                remapped_edges = []
                for edge in valid_edges:
                    new_v1 = old_to_new_indices[edge[0]]
                    new_v2 = old_to_new_indices[edge[1]]
                    if new_v1 >= 0 and new_v2 >= 0:
                        remapped_edges.append([new_v1, new_v2])
                
                if remapped_edges:
                    split_skeleton.edges = np.array(remapped_edges)
                else:
                    split_skeleton.edges = np.array([]).reshape(0, 2)
            else:
                split_skeleton.edges = np.array([]).reshape(0, 2)
            
            # Assign temporary ID for oversegment
            split_skeleton.id = label
            branch_removed_skeletons.append(split_skeleton)
        else:
            # No branch points, keep original skeleton
            skeleton.id = label
            branch_removed_skeletons.append(skeleton)
    
    # Step 3: Use branch-removed skeletons for oversegmentation
    try:
        overseg_labels, updated_skeletons = kimimaro.oversegment(
            binary_labels,
            branch_removed_skeletons,
            anisotropy=(1,1,1),
            progress=False,
            fill_holes=False,
            in_place=False,
            downsample=0  # No downsampling for accuracy
        )
        
    except Exception as e:
        print(f"Oversegment failed: {e}, falling back to traditional CCL")
        import cc3d
        result = cc3d.connected_components(binary_labels, connectivity=6, out_dtype=out_dtype)
        if return_N:
            N = len(np.unique(result)) - 1
            return result, N
        else:
            return result
    
    # Step 4: Split skeleton components and use segments attribute for regrouping
    final_labels = regroup_using_segments_attribute(
        overseg_labels, updated_skeletons, out_dtype
    )
    
    # Count final components
    unique_labels = np.unique(final_labels)
    N = len(unique_labels) - (1 if 0 in unique_labels else 0)
    
    if return_N:
        return final_labels, N
    else:
        return final_labels
    
def regroup_using_segments_attribute(overseg_labels, updated_skeletons, out_dtype):
    """
    Simple and correct regrouping method:
    1. Break each updated skeleton into individual components
    2. For each component, comp.segments directly gives supervoxel IDs
    3. Remap all those supervoxels to a new component ID
    """
    result = np.zeros_like(overseg_labels, dtype=out_dtype)
    next_component_id = 1
    
    for skeleton in updated_skeletons:
        if len(skeleton.vertices) == 0:
            continue
            
        # Step 1: Break skeleton into individual components
        components = skeleton.components()
        
        # Sort components deterministically
        components = sorted(components, 
                          key=lambda c: tuple(c.vertices[0]) if len(c.vertices) > 0 else (0,0,0))
        
        # Step 2: For each component, get supervoxel IDs and remap
        for comp in components:
            if len(comp.vertices) == 0:
                continue
                
            # Step 3: comp.segments directly gives us the supervoxel IDs!
            if hasattr(comp, 'segments') and comp.segments is not None:
                comp_supervoxel_ids = set(comp.segments)
                comp_supervoxel_ids.discard(0)  # Remove background
                
                # Remap all supervoxels in this component to the same new component ID
                for supervoxel_id in comp_supervoxel_ids:
                    mask = (overseg_labels == supervoxel_id)
                    result[mask] = next_component_id
                
                if len(comp_supervoxel_ids) > 0:
                    next_component_id += 1
    
    return result

@queueable
def CCLFacesTask(
  cloudpath:str, mip:int, 
  shape:ShapeType, offset:ShapeType,
  threshold_gte:Optional[Union[float,int]] = None,
  threshold_lte:Optional[Union[float,int]] = None,
  fill_missing:bool = False,
  dust_threshold:int = 0,
):
  """
  (1) Generate x,y,z back faces of each 1vx overlap task
  as crackle encoded 2D images.

  For continuous data, greater than or equal to (gte) 
  or less than or equal to (lte) thresholds 
  can be applied either individually or together. If neither
  is specified, no modification of the input will occur.

  These images are stored in e.g. 32_32_40/ccl/faces/ and have
  the following scheme where the numbers are the gridpoint
  location and the letters indicate which face plane.
    1-2-0-xy.ckl
    1-2-0-xz.ckl
    1-2-0-yz.ckl
  """
  shape = Vec(*shape)
  offset = Vec(*offset)
  bounds = Bbox(offset, offset + shape + 1) # 1 vx overlap

  if bounds.subvoxel():
    return

  cv = CloudVolume(cloudpath, mip=mip, fill_missing=fill_missing)
  bounds = Bbox.clamp(bounds, cv.meta.bounds(mip))

  grid_size = np.ceil(cv.bounds.size() / shape)
  gridpoint = np.floor(bounds.center() / shape).astype(int)
  label_offset = compute_label_offset(shape + 1, grid_size, gridpoint)
  
  labels = cv[bounds][...,0]
  labels = threshold_image(labels, threshold_lte, threshold_gte)
  labels = blackout_non_face_rails(labels, shape)
  if dust_threshold > 0:
    labels = cc3d.dust(
      labels, threshold=dust_threshold, 
      connectivity=6, in_place=True
    )
  # cc_labels = cc3d.connected_components(labels, connectivity=6, out_dtype=np.uint64)
  cc_labels = skeleton_based_connected_components(labels, out_dtype=np.uint64)
  cc_labels += np.uint64(label_offset)
  cc_labels[labels == 0] = 0

  # Uploads leading faces for adjacent tasks to examine
  slices = [
    cc_labels[:,:,-1],
    cc_labels[:,-1,:],
    cc_labels[-1,:,:],
  ]
  slices = [ crackle.compress(slc) for slc in slices ]
  filenames = [
    f'{gridpoint.x}-{gridpoint.y}-{gridpoint.z}-xy.ckl',
    f'{gridpoint.x}-{gridpoint.y}-{gridpoint.z}-xz.ckl',
    f'{gridpoint.x}-{gridpoint.y}-{gridpoint.z}-yz.ckl'
  ]

  cf = CloudFiles(cloudpath)
  filenames = [
    cf.join(cv.key, 'ccl', 'faces', fname) for fname in filenames
  ]
  cf.puts(zip(filenames, slices), compress='br')

@queueable
def CCLEquivalancesTask(
  cloudpath:str, mip:int,
  shape:ShapeType, offset:ShapeType,
  threshold_gte:Optional[Union[float,int]] = None,
  threshold_lte:Optional[Union[float,int]] = None,
  fill_missing:bool = False,
  dust_threshold:int = 0,
):
  """
  (2) Generate linkages between tasks by comparing the 
  front face of the task with the back faces of the 
  three adjacent tasks saved from the first step.

  Writes output to e.g. 32_32_40/ccl/equivalences/
  """
  shape = Vec(*shape)
  offset = Vec(*offset)
  bounds = Bbox(offset, offset + shape + 1) # 1 vx overlap

  if bounds.subvoxel():
    return

  cv = CloudVolume(cloudpath, mip=mip, fill_missing=fill_missing)
  bounds = Bbox.clamp(bounds, cv.meta.bounds(mip))

  grid_size = np.ceil(cv.bounds.size() / shape)
  gridpoint = np.floor(bounds.center() / shape).astype(int)
  label_offset = compute_label_offset(shape + 1, grid_size, gridpoint)
  
  equivalences = DisjointSet()

  labels = threshold_image(cv[bounds][...,0], threshold_lte, threshold_gte)
  labels = blackout_non_face_rails(labels, shape)
  if dust_threshold > 0:
    labels = cc3d.dust(
      labels, threshold=dust_threshold, 
      connectivity=6, in_place=True
    )
  # cc_labels, N = cc3d.connected_components(
  #   labels, connectivity=6, 
  #   out_dtype=np.uint64, return_N=True
  # )
  cc_labels, N = skeleton_based_connected_components(
    labels,  
    out_dtype=np.uint64, return_N=True
  )
  cc_labels += np.uint64(label_offset)
  cc_labels[labels == 0] = 0

  for i in range(1, N+1):
    equivalences.makeset(i + label_offset)

  cf = CloudFiles(cloudpath)
  filenames = [
    f'{gridpoint.x}-{gridpoint.y}-{gridpoint.z-1}-xy.ckl',
    f'{gridpoint.x}-{gridpoint.y-1}-{gridpoint.z}-xz.ckl',
    f'{gridpoint.x-1}-{gridpoint.y}-{gridpoint.z}-yz.ckl'
  ]
  filenames = [
    cf.join(cv.key, 'ccl', 'faces', fname) for fname in filenames
  ]

  slices = cf.get(filenames, return_dict=True)
  for key in slices:
    if slices[key] is None:
      continue
    face = crackle.decompress(slices[key])
    if '-xy' in key:
      face = face[:shape.x,:shape.y,0]
    elif '-xz' in key:
      face = face[:shape.x,:shape.z,0]
    else:
      face = face[:shape.y,:shape.z,0]
    slices[key] = face

  prev = [ slices[filenames[i]] for i in (0,1,2) ]

  cur = [
    cc_labels[:shape.x,:shape.y,0],
    cc_labels[:shape.x,0,:shape.z],
    cc_labels[0,:shape.y,:shape.z],
  ]

  for prev_i, cur_i in zip(prev, cur):
    if prev_i is None:
      continue

    mapping = fastremap.inverse_component_map(cur_i, prev_i)
    for task_label, adj_labels in mapping.items():
      if task_label == 0:
        continue
      for adj_label in fastremap.unique(adj_labels):
        if adj_label != 0:
          equivalences.union(int(task_label), int(adj_label))

  cf = CloudFiles(cloudpath)
  out_name = cf.join(cv.key, 'ccl', 'equivalences', f'{gridpoint.x}-{gridpoint.y}-{gridpoint.z}.json')
  cf.put_json(
    out_name, 
    { str(k): int(v) for k,v in equivalences.data.items() }, 
    compress='br'
  )

@queueable
def RelabelCCLTask(
  src_path:str, dest_path:str, mip:int,
  shape:ShapeType, offset:ShapeType,
  threshold_gte:Optional[Union[float,int]] = None,
  threshold_lte:Optional[Union[float,int]] = None,
  fill_missing:bool = False,
  dust_threshold:int = 0,
):
  """
  (4) Retrieves the relabeling for this task from the
  database, applies it, and saves the resulting image
  to the destination path. Upon saving, the 1 voxel
  overlap is omitted.
  """
  shape = Vec(*shape)
  offset = Vec(*offset)
  bounds = Bbox(offset, offset + shape + 1) # 1 vx overlap

  if bounds.subvoxel():
    return

  cv = CloudVolume(src_path, mip=mip, fill_missing=fill_missing)
  bounds = Bbox.clamp(bounds, cv.meta.bounds(mip))

  grid_size = np.ceil(cv.bounds.size() / shape)
  gridpoint = np.floor(bounds.center() / shape).astype(int)
  label_offset = compute_label_offset(shape + 1, grid_size, gridpoint)
  task_num = compute_task_number(grid_size, gridpoint)
  
  cf = CloudFiles(src_path)
  mapping_path = cf.join(cv.key, "ccl", "relabel", f"{task_num}.json")
  mapping = cf.get_json(mapping_path) or {}
  mapping =  { int(k):int(v) for k,v in mapping.items() }
  mapping[0] = 0

  labels = threshold_image(cv[bounds][...,0], threshold_lte, threshold_gte)
  labels = blackout_non_face_rails(labels, shape)
  if dust_threshold > 0:
    labels = cc3d.dust(
      labels, threshold=dust_threshold, 
      connectivity=6, in_place=True
    )
  # cc_labels, N = cc3d.connected_components(
  #   labels, connectivity=6, 
  #   out_dtype=np.uint64, return_N=True
  # )
  cc_labels, N = skeleton_based_connected_components(
    labels, 
    out_dtype=np.uint64, return_N=True
  )
  cc_labels += np.uint64(label_offset)
  cc_labels[labels == 0] = 0

  cc_labels = fastremap.remap(cc_labels, mapping, in_place=True)

  # Final upload without overlap
  dest_cv = CloudVolume(dest_path, mip=mip)
  bounds = Bbox(offset, offset + shape)
  bounds = Bbox.clamp(bounds, dest_cv.meta.bounds(mip))
  shape = bounds.size3()

  cc_labels = cc_labels[:shape.x,:shape.y,:shape.z].astype(dest_cv.dtype)
  cc_labels = cc_labels[:,:,:,np.newaxis]
  dest_cv[bounds] = cc_labels

def create_relabeling(cloudpath, mip, shape):
  """
  (3) Computes a relabeling from the linkages saved 
  from (2) and then saves them.

  Writes output to e.g. 32_32_40/ccl/relabel/
  and also .../ccl/max_label.json which contains
  the largest label.
  """
  cv = CloudVolume(cloudpath, mip=mip)
  cf = CloudFiles(cloudpath)
  all_eqpaths = cf.list(cf.join(cv.key, "ccl", "equivalences"))

  equivalences = DisjointSet()

  with tqdm(desc="Creating Union-Find", total=0) as pbar:
    for eqpaths in sip(all_eqpaths, 5000):
      eqdicts = cf.get_json(eqpaths)
      pbar.total += sum(( len(datum) for datum in eqdicts ))
      pbar.refresh()
      
      for data in eqdicts:
        for val1, val2 in data.items():
          equivalences.union(int(val1), int(val2))
          pbar.update()

      del eqdicts

  relabel = {}
  next_label = 1
  for key in tqdm(equivalences.data.keys(), desc="Renumbering"):
    lbl = equivalences.find(key)
    if lbl not in relabel:
      relabel[key] = next_label
      relabel[lbl] = next_label
      next_label += 1
    else:
      relabel[key] = relabel[lbl]

  del equivalences

  max_label_fname = cf.join(cv.key, "ccl", "max_label.json")
  cf.put_json(max_label_fname, [ next_label - 1 ])

  task_size = Vec(*shape) + 1
  task_voxels = task_size.x * task_size.y * task_size.z

  buckets = defaultdict(dict)
  for before_val, after_val in relabel.items():
    task_num = int(before_val // task_voxels)
    buckets[task_num][before_val] = after_val

  del relabel

  cf.put_jsons(
    (
      (cf.join(cv.key, "ccl", "relabel", f"{task_num}.json"), relabeling) 
      for task_num, relabeling in buckets.items()
    ),
    total=len(buckets),
    compress="br",
    progress=True
  )

def clean_intermediate_files(src, mip):
  cv = CloudVolume(src, mip)
  cf = CloudFiles(src)
  cf.delete(cf.list(cf.join(cv.key, "ccl")))





