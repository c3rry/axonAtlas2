import numpy as np
from skimage.io import imread, imsave
from skimage.draw import line_nd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

import os
import datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from skimage.io import imread, imsave
import shutil
from sklearn.cluster import DBSCAN
import matplotlib

# Ensure line_nd is imported
# from your_module import line_nd 

def connections(
    stack_input, 
    output_dir=None, 
    filename_prefix="cluster", 
    metadata=None, 
    scale_factor=25,
    annotation_volume=None, # <--- NEW: 3D Numpy array (Z, Y, X) of region IDs
    structure_map=None      # <--- NEW: Dict {id: {'acronym': 'STR', 'name': 'Striatum'}}
):
    """
    Modified to:
      1. Accept batch lists or single files.
      2. Generate SWC files with specific Janelia/MouseLight formatting.
      3. Scale coordinates by scale_factor.
      4. Append Region ID, Acronym, and Name columns (Extended SWC).
    """

    # --- 0. Handle List Input (Batch Processing) ---
    if isinstance(stack_input, list):
        print(f"Batch processing {len(stack_input)} files with scale factor {scale_factor}...")
        tif_list = []
        csv_list = []
        
        for path in stack_input:
            _, saved_tif, saved_csv = connections(
                stack_input=path, 
                output_dir=output_dir,
                filename_prefix=None,
                metadata=metadata,
                scale_factor=scale_factor,
                annotation_volume=annotation_volume, # Pass atlas down
                structure_map=structure_map          # Pass map down
            )
            tif_list.append(saved_tif)
            csv_list.append(saved_csv)
            
        return tif_list, csv_list

    # --- 1. Handle Single Input ---
    if isinstance(stack_input, str):
        print(f"Loading TIFF stack from: {stack_input}")
        stack = imread(stack_input)
        base_name = os.path.basename(stack_input)
        file_name, file_ext = os.path.splitext(base_name)
    else:
        stack = stack_input
        file_name = filename_prefix
        file_ext = ".tif"

    # --- 2. Setup Output Paths ---
    if output_dir is None:
        if isinstance(stack_input, str):
             output_dir = os.path.dirname(stack_input)
        else:
             output_dir = os.getcwd()
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename_tif = f"{file_name}_connected.tif"
    output_filename_csv = f"{file_name}.swc"
    
    output_tiff_path = os.path.join(output_dir, output_filename_tif)
    output_csv_path = os.path.join(output_dir, output_filename_csv)

    # --- 3. Extract Coordinates ---
    coordinates_tuple = np.nonzero(stack)
    points = np.stack(coordinates_tuple, axis=1) # (z, y, x)
    
    if points.shape[0] < 2:
        print(f"Skipping {file_name}: Fewer than 2 points found.")
        return np.zeros_like(stack), None, None

    # --- 4. Compute MST ---
    distance_matrix_square = squareform(pdist(points, 'euclidean'))
    mst_sparse_matrix = minimum_spanning_tree(distance_matrix_square)
    G = nx.from_scipy_sparse_array(mst_sparse_matrix)

    # --- 5. Generate Formatted SWC Data ---
    try:
        parents = dict(nx.bfs_predecessors(G, source=0))
    except Exception as e:
        print(f"Graph error on {file_name}: {e}. Defaulting to linear.")
        parents = {}

    with open(output_csv_path, "w") as f:
        
        # A. Write Metadata Header
        current_date = datetime.datetime.now().strftime("%Y/%m/%d")
        
        if metadata:
            f.write(metadata)
        else:
            f.write(f"# Generated {current_date}.\n")
            f.write(f"# Generator: AxonAtlas2 Pipeline\n")
            f.write(f"# Neuron Id: {file_name}\n")
            f.write(f"# Scaling Factor: {scale_factor} (Applied to x,y,z)\n")
            f.write("# Annotation Space: CCFv2.5 (ML legacy) Axes> Z: Anterior-Posterior; Y: Inferior-Superior; X:Left-Right\n")
        
        # B. Write Column Headers (Extended)
        # Standard: n, type, x, y, z, radius, parent
        # Extended: region_id, region_acronym, region_name
        f.write("# n\ttype\tx\ty\tz\tradius\tparent\tregion_id\tregion_acronym\tregion_name\n")
        
        # C. Write Data Rows
        for i in range(len(points)):
            sample_number = i + 1 
            
            # Raw Indices (for Atlas Lookup)
            z_idx, y_idx, x_idx = points[i, 0], points[i, 1], points[i, 2]

            # 1. Look up Region Info
            region_id = 0
            region_acronym = "undefined"
            region_name = "undefined"

            if annotation_volume is not None:
                try:
                    # Check bounds just in case the mask is slightly larger than the atlas
                    if (z_idx < annotation_volume.shape[0] and 
                        y_idx < annotation_volume.shape[1] and 
                        x_idx < annotation_volume.shape[2]):
                        
                        region_id = annotation_volume[z_idx, y_idx, x_idx]
                        
                        # Fetch details from map if available
                        if structure_map and region_id in structure_map:
                            info = structure_map[region_id]
                            # Handle different map structures safely
                            if isinstance(info, dict):
                                region_acronym = info.get('acronym', "undefined")
                                region_name = info.get('name', "undefined")
                                # Sanitize strings to remove tabs or newlines
                                region_name = str(region_name).replace('\t', ' ').replace('\n', '')
                except IndexError:
                    pass # Keep default 0/undefined

            # 2. Apply Scaling Factor (Physical Coordinates)
            z_coord = z_idx * scale_factor
            y_coord = y_idx * scale_factor
            x_coord = x_idx * scale_factor
            
            radius = 1.0 
            
            if i == 0:
                # ROOT Node
                structure_id = 1
                parent_number = -1
            else:
                # BRANCH Node
                structure_id = 2
                parent_idx = parents.get(i, -1) 
                parent_number = parent_idx + 1 if parent_idx != -1 else -1

            # 3. Format Line with Extended Columns
            # Note: region_id is usually integer, but written as string for safety
            line = (f"{sample_number}\t{structure_id}\t"
                    f"{x_coord:.6f}\t{y_coord:.6f}\t{z_coord:.6f}\t"
                    f"{radius:.6f}\t{parent_number}\t"
                    f"{region_id}\t{region_acronym}\t{region_name}\n")
            
            f.write(line)

    # --- 6. "Burn" Lines into New Stack (Visuals) ---
    output_stack = stack.copy()
    
    if np.issubdtype(stack.dtype, np.integer):
        draw_value = np.iinfo(stack.dtype).max 
    else:
        draw_value = 1.0

    for node1, node2 in G.edges():
        p1 = points[node1]
        p2 = points[node2]
        try:
            # Assuming line_nd is available in your scope
            zz, yy, xx = line_nd(p1, p2)
            zz = np.clip(zz, 0, output_stack.shape[0]-1)
            yy = np.clip(yy, 0, output_stack.shape[1]-1)
            xx = np.clip(xx, 0, output_stack.shape[2]-1)
            output_stack[zz, yy, xx] = draw_value
        except NameError:
             # print("line_nd function missing, skipping visualization.")
             pass 

    # --- 7. Save TIFF and Return ---
    imsave(output_tiff_path, output_stack, check_contrast=False)
    
    return output_stack, output_tiff_path, output_csv_path

def dbscan(
    input_tiff_path: str, 
    output_dir: str, 
    eps: float = 3.0, 
    min_samples: int = 10,
    min_cluster_size: int = 50 
) -> list:
    """
    Loads a 3D TIFF stack, applies DBSCAN clustering, and saves results.
    
    Now filters output based on cluster size.

    Args:
        input_tiff_path (str): Path to the binary input TIFF.
        output_dir (str): Directory where results will be saved.
        eps (float): Max distance between samples for neighborhood.
        min_samples (int): Min samples for a core point.
        min_cluster_size (int): The minimum number of points (pixels) a cluster 
                                must have to be saved as an individual TIFF file.

    Returns:
        list: A list of file paths to the generated isolated cluster TIFFs.
    """
    
    # --- 1. Setup Directories ---
    base_name = os.path.splitext(os.path.basename(input_tiff_path))[0]
    clusters_subdir = os.path.join(output_dir, f"{base_name}_clusters")
    
    if os.path.exists(clusters_subdir):
        print(f"Cleaning previous run: Removing {clusters_subdir}")
        shutil.rmtree(clusters_subdir)
        
    os.makedirs(clusters_subdir, exist_ok=True)
    
    # --- 2. Load Data ---
    print(f"Loading TIFF: {input_tiff_path}")
    stack = imread(input_tiff_path)
    
    coords_tuple = np.nonzero(stack)
    points = np.stack(coords_tuple, axis=1)
    
    if len(points) == 0:
        print("Error: Input TIFF is empty.")
        return []

    print(f"Processing {len(points)} points with DBSCAN (eps={eps}, min_samples={min_samples})...")

    # --- 3. Run DBSCAN ---
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_ 
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} unique clusters.")

    # --- 4. Create Combined "Colored" (RGB) Stack ---
    # Note: We still visualize ALL clusters here so you can see what is being filtered out.
    print(f"Mapping {n_clusters} clusters to vibrant RGB colors...")
    
    cluster_labels = [l for l in unique_labels if l != -1]
    output_shape = stack.shape + (3,)
    labeled_stack_rgb = np.zeros(output_shape, dtype=np.uint8)
    
    if n_clusters > 0:
        hues = np.linspace(0, 1.0, n_clusters, endpoint=False)
        vibrant_colors_float = matplotlib.cm.hsv(hues)
        vibrant_colors = (vibrant_colors_float[:, :3] * 255).astype(np.uint8)
        np.random.shuffle(vibrant_colors)
        
        max_original_label = max(unique_labels)
        label_map_rgb = np.zeros((max_original_label + 3, 3), dtype=np.uint8)

        label_map_rgb[1] = [50, 50, 50] # Noise color
        
        for i, label in enumerate(cluster_labels):
            label_map_rgb[label + 2] = vibrant_colors[i]
            
        values_to_assign = labels + 2
        mapped_rgb_values = label_map_rgb[values_to_assign]
        labeled_stack_rgb[coords_tuple] = mapped_rgb_values
        
    else:
        labeled_stack_rgb[coords_tuple] = [50, 50, 50]
        
    combined_output_path = os.path.join(output_dir, f"{base_name}_combined_labels_rgb.tif")
    imsave(combined_output_path, labeled_stack_rgb, check_contrast=False)
    print(f"Saved combined RGB labeled stack: {combined_output_path}")

    # --- 5. Save Individual Cluster TIFFs (With Filtering) ---
    saved_paths = []
    print(f"Saving individual cluster stacks (Min points required: {min_cluster_size})...")
    
    skipped_count = 0

    for label in unique_labels:
        if label == -1:
            continue
            
        mask = (labels == label)
        cluster_points = points[mask]
        
        # --- NEW CHECK: Filter by Size ---
        # If the cluster is smaller than the minimum, skip it
        if len(cluster_points) < min_cluster_size:
            skipped_count += 1
            continue

        single_cluster_stack = np.zeros_like(stack, dtype=np.uint8)
        single_cluster_stack[cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2]] = 255
        
        filename = f"cluster_{label:03d}.tif"
        file_path = os.path.join(clusters_subdir, filename)
        
        imsave(file_path, single_cluster_stack, check_contrast=False)
        saved_paths.append(file_path)
        
    print(f"Successfully saved {len(saved_paths)} cluster files.")
    print(f"Skipped {skipped_count} clusters because they had fewer than {min_cluster_size} points.")
    
    return saved_paths