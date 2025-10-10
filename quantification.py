import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import seaborn as sns
import os
from bg_atlasapi import BrainGlobeAtlas
import brainglobe_heatmap as bgh
import tempfile
import nibabel as nib
import shutil
import gc
import re
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict



"""def calculate_axon_percentage(axon_mask_path, atlas_name='allen_mouse_25um'):
    
    Calculate percentage of axon voxels per brain region in 25um atlas space.
    
    Args:
        axon_mask_path (str): Path to 3D binary axon segmentation TIFF file
        atlas_name (str): Name of BrainGlobe atlas (default: 'allen_mouse_25um')
    
    Returns:
        pd.DataFrame: Results with all atlas metadata plus axon statistics
    
    # Load atlas and annotation
    atlas = BrainGlobeAtlas(atlas_name)
    annotation = atlas.annotation
    
    # Load axon mask
    axon_mask = tifffile.imread(axon_mask_path).astype(bool)
    
    # Verify shape compatibility
    if axon_mask.shape != annotation.shape:
        raise ValueError(f"Mask shape {axon_mask.shape} doesn't match atlas shape {annotation.shape}")
    
    # Prepare results storage
    results = []
    lookup_df = atlas.lookup_df
    
    print(f"Processing {len(lookup_df)} brain regions...")

    # Assume axis 0 is the left/right hemisphere axis (check atlas documentation if unsure)
    midline = annotation.shape[0] // 2
    
    # Calculate axon percentages per region
    for region_id, region_info in lookup_df.iterrows():
        # Skip background/root regions
        if region_id == 0 or region_info['acronym'] == 'root': 
            continue
        
        for hemi, hemi_suffix, hemi_slice in [
            ('Left', '_left', slice(0, midline)),
            ('Right', '_right', slice(midline, annotation.shape[0]))
        ]:
            region_mask = (annotation[hemi_slice, :, :] == region_id)

            # --- Insert this block here ---
            atlas_mask = atlas.get_structure_mask(region_info['acronym'])
            if hemi == 'Left':
                left_manual = region_mask
                left_atlas = atlas_mask[0:midline, :, :]
                print(f"{region_info['acronym']} Left: {np.array_equal(left_manual, left_atlas)}")
            else:
                right_manual = region_mask
                right_atlas = atlas_mask[midline:, :, :]
                print(f"{region_info['acronym']} Right: {np.array_equal(right_manual, right_atlas)}")
        # --- End insert ---


            total_voxels = np.sum(region_mask)
            if total_voxels == 0:
                continue

            axon_in_region = np.logical_and(axon_mask[hemi_slice, :, :], region_mask)
            axon_voxels = np.sum(axon_in_region)
            percentage = (axon_voxels / total_voxels) * 100

            acronym_hemi = region_info['acronym'] + hemi_suffix
            print(f"{acronym_hemi}: {percentage:.4f}%")

            result = region_info.to_dict()
            result.update({
                'acronym': acronym_hemi,
                'hemisphere': hemi,
                'axon_voxels': axon_voxels,
                'total_voxels': total_voxels,
                'percentage': percentage
            })
            results.append(result)

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('percentage', ascending=False)

    print(f"\nCompleted! Processed {len(result_df)} region-hemisphere pairs.")
    return result_df
    """

def plot_top_regions(df, top_n=20):
    plt.figure(figsize=(12, 8))
    top_df = df.nlargest(top_n, 'percentage')
    ax = sns.barplot(x='percentage', y='acronym', data=top_df, palette='viridis')
    plt.title(f'Top {top_n} Regions by Axon Percentage')
    plt.xlabel('Axon Percentage (%)')
    plt.ylabel('Brain Region')
    plt.tight_layout()
    plt.show()
    

def get_hemisphere_from_acronym(acronym):
    """Infers hemisphere from region acronym for CSV output."""
    acronym_lower = str(acronym).lower()
    if acronym_lower.endswith(('_l', '_left')):
        return "Left"
    if acronym_lower.endswith(('_r', '_right')):
        return "Right"
    if 'left' in acronym_lower:
        return "Left"
    if 'right' in acronym_lower:
        return "Right"
    return "Unknown"

def calculate_axon_percentage(input_volume, temp_dir, voxel_size, output_dir):
    """
    Calculates axon percentages for ALL brain regions and returns a DataFrame.
    """
    print(f"[INFO] Starting calculation for input: {input_volume}")
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    final_df = pd.DataFrame()

    try:
        # 1. Load Input Volume
        ext = os.path.splitext(input_volume)[1].lower()
        print(f"[INFO] Loading {ext} file...")
        if ext in [".tif", ".tiff"]:
            data = tifffile.imread(input_volume)
        elif ext in [".nii", ".nii.gz"]:
            data = nib.load(input_volume).get_fdata()
        elif ext == ".npy":
            data = np.load(input_volume)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        print(f"[INFO] Loaded data with shape: {data.shape}")

        # 2. Load BrainGlobe Atlas
        print(f"[INFO] Loading BrainGlobe Atlas at {voxel_size}um resolution...")
        bg_atlas = BrainGlobeAtlas(f"allen_mouse_{voxel_size}um", check_latest=False)
        
        # 3. Get all region acronyms from the atlas
        all_structure_data = bg_atlas.structures.values()
        all_regions = [s['acronym'] for s in all_structure_data if 'acronym' in s]
        print(f"[INFO] Found {len(all_regions)} named regions in the atlas to process.")
        
        if not all_regions:
            print("[ERROR] Could not find any region acronyms in the loaded atlas.")
            return final_df

        # 4. Process each region
        for i, region_acronym in enumerate(all_regions):
            print(f"--- Processing region {i+1}/{len(all_regions)}: {region_acronym} ---")
            
            if region_acronym == 'root':
                continue

            region_id = bg_atlas.structures[region_acronym]['id']
            region_name = bg_atlas.structures[region_acronym]['name']
            hemisphere = get_hemisphere_from_acronym(region_acronym)
            
            mask = bg_atlas.get_structure_mask(region_id)

            if data.shape != mask.shape:
                min_shape = np.minimum(data.shape, mask.shape)
                data_sliced = data[:min_shape[0], :min_shape[1], :min_shape[2]]
                mask_sliced = mask[:min_shape[0], :min_shape[1], :min_shape[2]]
            else:
                data_sliced, mask_sliced = data, mask

            total_voxels = np.sum(mask_sliced)
            
            if total_voxels > 0:
                # --- MEMORY-EFFICIENT CALCULATION (THIS IS THE FIX) ---
                # This performs a boolean AND operation instead of creating a large data slice,
                # which prevents the MemoryError.
                axon_voxels = np.count_nonzero(np.logical_and(data_sliced > 0, mask_sliced))
            else:
                axon_voxels, percentage = 0, 0.0
            
            percentage = (axon_voxels / total_voxels) * 100.0 if total_voxels > 0 else 0.0

            results.append({
                'acronym': region_acronym, 'id': region_id, 'name': region_name,
                'hemisphere': hemisphere, 'axon_voxels': axon_voxels,
                'total_voxels': total_voxels, 'percentage': percentage
            })
        
        del data
        
        # 5. Create DataFrame, save it, and prepare for return
        if results:
            df = pd.DataFrame(results)[['acronym', 'id', 'name', 'hemisphere', 'axon_voxels', 'total_voxels', 'percentage']]
            output_csv_path = os.path.join(output_dir, "axon_percentages_all_regions.csv")
            final_df = df.sort_values(by='percentage', ascending=False)
            final_df.to_csv(output_csv_path, index=False)
            print(f"\n[SUCCESS] Successfully saved axon percentages to: {output_csv_path}")
        else:
            print("\n[INFO] No results were generated.")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()
        return final_df


def axonDict(df):
    """
    Convert a DataFrame with 'acronym' and 'percentage' columns
    to a dictionary: {acronym: percentage}
    """
    return pd.Series(df['percentage'].values, index=df['acronym']).to_dict()

def volumeExtract(directory, mode="threshold"):
    """
    Search for a TIFF file starting with 'downsampled_standard' and ending with
    either '_threshold' or '_skeletonize', based on the 'mode' parameter.

    Args:
        directory (str): The root directory to search.
        mode (str): Either 'threshold' or 'skeletonize' (default: 'threshold').

    Returns:
        str or None: Full file path of the matching file, or None if not found.
    """
    if mode not in ("threshold", "skeletonize"):
        raise ValueError("mode must be 'threshold' or 'skeletonize'")

    suffix = f"_{mode}"
    for root, dirs, files in os.walk(directory):
        for file in files:
            fname = file.lower()
            if fname.startswith("downsampled_standard") and (
                fname.endswith(f"{suffix}.tif") or fname.endswith(f"{suffix}.tiff")
            ):
                return os.path.join(root, file)
    return None

'''def bg_heatmap(
    values,
    vmin,
    vmax,
    cmap='Reds',
    atlas_name='allen_mouse_25um',
    thickness=1000,
    annotate_regions=True,
    format='2D',
):
    """
    Create and display anatomical heatmaps in frontal, sagittal, and horizontal orientations.
    
    Parameters:
        values (dict): {acronym: value} region-value mapping.
        vmin, vmax (float): Color scale limits.
        cmap (str): Matplotlib colormap name.
        atlas_name (str): Name of the atlas to use.
        thickness (int): Thickness of the slice in microns.
        annotate_regions (bool): Annotate region labels on the plot.
        format (str): '2D' (matplotlib) or '3D' (brainrender).
    """
    import brainglobe_heatmap as bgh

    orientations = [
        ("frontal", "Frontal (coronal) view"),
        ("sagittal", "Sagittal (side) view"),
        ("horizontal", "Horizontal (top-down) view"),
    ]
    
    for orientation, title in orientations:
        print(f"Rendering {title}...")
        bgh.Heatmap(
            values=values,
            position=None,  # Centered slice
            orientation=orientation,
            thickness=thickness,
            atlas_name=atlas_name,
            title=title,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            annotate_regions=annotate_regions,
            format=format
        ).show()'''

def bg_heatmap(
    values,
    vmin,
    vmax,
    cmap='Reds',
    atlas_name='allen_mouse_25um',
    thickness=1000,
    annotate_regions=True,
    format='2D',
    figsize=(10, 8), 
    frame=None
):
    """
    Processes a dictionary of brain region data and plots the left and right
    hemispheres together on a single heatmap for each standard orientation.

    Args:
        values (dict): A single dictionary mapping region acronyms to values.
                       Keys must end in '_left' or '_right'.
        vmin (float): The minimum value for the color scale.
        vmax (float): The maximum value for the color scale.
        cmap (str): Matplotlib colormap name.
        atlas_name (str): Name of the brain atlas to use.
        thickness (int): Thickness of the slice in microns.
        annotate_regions (bool): Whether to annotate region labels on the plot.
        format (str): The format of the plot, must be '2D'.
        figsize (tuple): Figure size for each plot.

    Returns:
        None

    Requires:
        pip install pandas matplotlib bg-heatmap numpy
    """
    # --- 1. Data Processing: Split the single dictionary into two ---
    left_dict = {}
    right_dict = {}
    for key, value in values.items():
        if isinstance(key, str) and pd.notna(value):
            if key.endswith('_left'):
                base_key = key.removesuffix('_left')
                if base_key != 'nan':
                    left_dict[base_key] = value
            elif key.endswith('_right'):
                base_key = key.removesuffix('_right')
                if base_key != 'nan':
                    right_dict[base_key] = value

    # --- 2. Visualization Loop ---
    orientations = ["frontal", "sagittal", "horizontal"]
    
    for orientation in orientations:
        if format == '2D':
            print(f"Rendering {orientation.capitalize()} view...")
            # Create a figure with a single subplot for each orientation
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            
            # --- Plot Left Hemisphere ---
            hm_left = bgh.Heatmap(
                left_dict, position=frame, orientation=orientation, hemisphere='left',
                thickness=thickness, atlas_name=atlas_name, format=format, cmap=cmap,
                annotate_regions=annotate_regions, vmin=vmin, vmax=vmax
            )
            # Plot on the single axis, but don't show the colorbar yet
            hm_left.plot_subplot(fig=fig, ax=ax, show_cbar=False)

            # --- Plot Right Hemisphere ---
            hm_right = bgh.Heatmap(
                right_dict, position=frame, orientation=orientation, hemisphere='right',
                thickness=thickness, atlas_name=atlas_name, format=format, cmap=cmap,
                annotate_regions=annotate_regions, vmin=vmin, vmax=vmax
            )
            # Plot on the same axis, and now show the colorbar
            hm_right.plot_subplot(fig=fig, ax=ax, show_cbar=True)
            
            ax.set_title(f"Combined Projection Heatmap - {orientation.capitalize()} View", fontsize=16)
            plt.tight_layout()
            plt.show()
        
        elif format == '3D':
            raise NotImplementedError("3D plotting is not yet implemented in this function.")
        else:
            raise ValueError("format must be '2D'")
        

def bg_heatmap_slices(
    values,
    output_dir,
    view,
    vmin,
    vmax,
    cmap='Reds',
    atlas_name='allen_mouse_25um',
    annotate_regions=True
):
    """
    Generates and saves a TIFF image for every single plane of a heatmap for a given view.

    Args:
        values (dict): A dictionary mapping region acronyms to values. 
                       Keys must end in '_left' or '_right'.
        output_dir (str): The directory where the output files will be saved.
        view (str): The anatomical orientation to slice through. 
                    Must be one of "frontal", "sagittal", or "horizontal".
        vmin (float): The minimum value for the color scale.
        vmax (float): The maximum value for the color scale.
        cmap (str, optional): Matplotlib colormap name. Defaults to 'Reds'.
        atlas_name (str, optional): Name of the brain atlas to use. Defaults to 'allen_mouse_25um'.
        annotate_regions (bool, optional): Whether to annotate region labels on the plot. Defaults to True.
    """
    # --- 1. Setup and Data Processing ---
    print(f"--- Starting Heatmap Slice Generation for '{view}' view ---")
    
    heatmap_dir = os.path.join(output_dir, "heatmap")
    os.makedirs(heatmap_dir, exist_ok=True)
    print(f"Output will be saved in: {heatmap_dir}")

    left_dict, right_dict = {}, {}
    for key, value in values.items():
        if isinstance(key, str) and pd.notna(value):
            if key.endswith('_left'):
                base_key = key.removesuffix('_left')
                if base_key != 'nan': left_dict[base_key] = value
            elif key.endswith('_right'):
                base_key = key.removesuffix('_right')
                if base_key != 'nan': right_dict[base_key] = value

    # --- 2. Determine Slicing Range from Atlas ---
    print("Loading atlas to determine slicing range...")
    atlas = BrainGlobeAtlas(atlas_name)
    
    view_map = {
        "frontal": atlas.reference.shape[0],
        "horizontal": atlas.reference.shape[1],
        "sagittal": atlas.reference.shape[2]
    }
    if view not in view_map:
        raise ValueError(f"View must be one of {list(view_map.keys())}")
    
    n_slices = view_map[view]
    slice_thickness_um = atlas.resolution[0] 
    print(f"Atlas loaded. Found {n_slices} slices for the '{view}' view.")

    # --- 3. Iterate Through Each Plane and Save ---
    for i in range(n_slices):
        position_um = i * slice_thickness_um
        fig, ax = plt.subplots(figsize=(10, 8))

        # --- Plot Left Hemisphere ---
        hm_left = bgh.Heatmap(
            left_dict, position=position_um, orientation=view, hemisphere='left',
            atlas_name=atlas_name, cmap=cmap, vmin=vmin, vmax=vmax,
            annotate_regions=annotate_regions
        )
        # --- THIS IS THE FIX ---
        hm_left.plot_subplot(fig=fig, ax=ax, show_cbar=False)

        # --- Plot Right Hemisphere on the same axes ---
        hm_right = bgh.Heatmap(
            right_dict, position=position_um, orientation=view, hemisphere='right',
            atlas_name=atlas_name, cmap=cmap, vmin=vmin, vmax=vmax,
            annotate_regions=annotate_regions
        )
        # --- THIS IS THE FIX ---
        hm_right.plot_subplot(fig=fig, ax=ax, show_cbar=False)

        ax.axis('off')
        fig.tight_layout(pad=0)
        
        filename = f"{view}_slice_{i:04d}.tif"
        filepath = os.path.join(heatmap_dir, filename)
        
        fig.savefig(filepath, format='tiff', bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)
        
        print(f"Saved: {filename}")

    print("\n--- Heatmap slice generation complete! ---")

def plot_region_clustering(df, output_dir, top_n=50):
    """
    Creates a clustered heatmap (dendrogram) to visualize region similarity.

    Args:
        df (pd.DataFrame): DataFrame with axon data ('acronym', 'total_voxels', 'percentage', 'axon_voxels').
        output_dir (str): Directory to save the plot image.
        top_n (int, optional): Number of top innervated regions to include in the plot. Defaults to 50.
    """
    print("\n--- Generating Region Clustering Visualization ---")
    
    # --- 1. Prepare data for clustering ---
    # Filter for relevant data and select the top N regions by percentage
    cluster_df = df[df['axon_voxels'] > 0].copy()
    cluster_df = cluster_df.sort_values(by='percentage', ascending=False).head(top_n)
    cluster_df['log_total_voxels'] = np.log10(cluster_df['total_voxels'])
    cluster_df = cluster_df.set_index('name')

    # --- 2. Define and scale features for clustering ---
    # We cluster based on the region's size and its innervation percentage
    features_for_clustering = cluster_df[['log_total_voxels', 'percentage']]
    
    # Scale features to have a mean of 0 and variance of 1. This is crucial for accurate clustering.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_for_clustering)

    # --- 3. Perform Hierarchical Clustering ---
    # We use the 'ward' method, which is effective at finding distinct clusters
    linkage_matrix = linkage(scaled_features, method='ward')

    # --- 4. Create the Clustered Heatmap ---
    # The heatmap color will represent the axon percentage
    heatmap_data = cluster_df[['percentage']]

    # Use seaborn's clustermap to combine the heatmap and dendrogram
    g = sns.clustermap(
        heatmap_data,
        row_linkage=linkage_matrix,  # Use our custom clustering
        col_cluster=False,           # Don't cluster the single column
        cmap='viridis',              # Colormap for the heatmap
        figsize=(12, 16),
        cbar_pos=(0.05, 0.85, 0.03, 0.1), # Position the colorbar
        cbar_kws={'label': 'Axon Percentage (%)'}
    )

    # --- 5. Style and Save the Plot ---
    g.fig.suptitle('Hierarchical Clustering of Axon Innervation', fontsize=16, weight='bold', y=0.95)
    ax = g.ax_heatmap
    ax.set_xlabel('') # Remove default x-label
    ax.set_ylabel('Brain Region', fontsize=12)
    ax.tick_params(axis='y', labelsize=10) # Adjust y-tick label size
    
    # Adjust the dendrogram line width
    g.ax_row_dendrogram.set_visible(True)
    for collection in g.ax_row_dendrogram.collections:
        collection.set_linewidth(1.5)

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "axon_clustering_heatmap.png")
    g.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Clustering plot saved to: {plot_path}")
    plt.show()
# --- Main Analysis Function ---
def calculate_hemisphere_axon_percentage(input_volume, output_dir, voxel_size=25, temp_dir=None):
    """
    Splits every brain region by the midline and calculates axon percentages for each side.
    """
    print("\n--- Starting Analysis for HEMISPHERE-SPLIT Brain Regions ---")
    os.makedirs(output_dir, exist_ok=True)
    results = []
    final_df = pd.DataFrame()

    try:
        data = tifffile.imread(input_volume)
        bg_atlas = BrainGlobeAtlas(f"allen_mouse_{voxel_size}um", check_latest=False)
        atlas_shape = bg_atlas.reference.shape
        sagittal_midpoint = atlas_shape[2] // 2
        
        left_hemisphere_mask = np.zeros(atlas_shape, dtype=bool)
        left_hemisphere_mask[:, :, :sagittal_midpoint] = True
        right_hemisphere_mask = np.zeros(atlas_shape, dtype=bool)
        right_hemisphere_mask[:, :, sagittal_midpoint:] = True

        all_acronyms = [s['acronym'] for s in bg_atlas.structures.values() if 'acronym' in s]
        base_acronyms = sorted(list(set(re.sub(r'(_left|_right|_l|_r)$', '', ac, flags=re.IGNORECASE) 
                                        for ac in all_acronyms)))
        
        for i, base_acronym in enumerate(base_acronyms):
            if base_acronym in ['root', 'CH']: continue
            print(f"--- Processing {base_acronym} ({i+1}/{len(base_acronyms)}) ---")
            
            try:
                structure_mask = bg_atlas.get_structure_mask(base_acronym)
            except KeyError: continue

            for hemi_name, hemi_mask, hemi_suffix in [('Left', left_hemisphere_mask, '_left'), ('Right', right_hemisphere_mask, '_right')]:
                final_mask = np.logical_and(structure_mask, hemi_mask)
                total_voxels = np.sum(final_mask)
                if total_voxels >= 0:
                    axon_voxels = np.count_nonzero(np.logical_and(data > 0, final_mask))
                    results.append({
                        'acronym': f"{base_acronym}{hemi_suffix}",
                        'id': bg_atlas.structures[base_acronym]['id'],
                        'name': bg_atlas.structures[base_acronym]['name'],
                        'hemisphere': hemi_name, 'axon_voxels': axon_voxels,
                        'total_voxels': total_voxels,
                        'percentage': (axon_voxels / (total_voxels if total_voxels > 0 else 1)) * 100.0
                    })
        
        if results:
            cols = ['acronym', 'id', 'name', 'hemisphere', 'axon_voxels', 'total_voxels', 'percentage']
            final_df = pd.DataFrame(results)[cols]
            output_csv_path = os.path.join(output_dir, "axon_percentages_hemispheres.csv")
            final_df.sort_values(by='percentage', ascending=False).to_csv(output_csv_path, index=False)
            print(f"\n[SUCCESS] Saved hemisphere analysis to: {output_csv_path}")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()
        return final_df

# --- 2D DENDROGRAM FUNCTION (BASED ON AXON PERCENTAGE SIMILARITY) ---
def plot_anatomical_clustering(df_hemispheres, output_dir, parent_region='grey', max_depth=4, c_map='viridis'):
    """
    Takes hemisphere data, combines it into bilateral totals, then creates a 2D dendrogram
    based on axon percentage similarity.
    """
    print("\n--- Generating 2D Anatomical Clustering Visualization ---")

    # --- 1. Aggregate Hemisphere Data into Bilateral Data ---
    print("[INFO] Aggregating hemisphere data into bilateral results...")
    df_hemispheres['base_acronym'] = df_hemispheres['acronym'].str.replace(r'_left|_right$', '', regex=True)
    
    df_bilateral = df_hemispheres.groupby('base_acronym').agg(
        id=('id', 'first'),
        name=('name', 'first'),
        axon_voxels=('axon_voxels', 'sum'),
        total_voxels=('total_voxels', 'sum')
    ).reset_index()
    df_bilateral.rename(columns={'base_acronym': 'acronym'}, inplace=True)
    df_bilateral['percentage'] = (df_bilateral['axon_voxels'] / df_bilateral['total_voxels'].replace(0, np.nan)) * 100
    df_bilateral['percentage'].fillna(0, inplace=True)
    
    # --- 2. Build Anatomical Tree ---
    atlas = BrainGlobeAtlas('allen_mouse_25um')
    children_map = defaultdict(list)
    for struct_id, struct_info in atlas.structures.items():
        if len(struct_info['structure_id_path']) > 1:
            parent_id = struct_info['structure_id_path'][-2]
            children_map[parent_id].append(struct_info)

    def get_descendants(region_id, current_depth, max_depth):
        if current_depth >= max_depth:
            return [atlas.structures[region_id]['acronym']]
        children = children_map.get(region_id, [])
        if not children:
            return [atlas.structures[region_id]['acronym']]
        desc_list = []
        for child in children:
            desc_list.extend(get_descendants(child['id'], current_depth + 1, max_depth))
        return desc_list

    try:
        parent_id = atlas.structures[parent_region]['id']
    except KeyError:
        print(f"[ERROR] Parent region '{parent_region}' not found in atlas.")
        return

    anatomical_leaves = sorted(list(set(get_descendants(parent_id, 0, max_depth))))
    print(f"Found {len(anatomical_leaves)} anatomical regions under '{parent_region}' up to depth {max_depth}.")

    # --- 3. Prepare Data Matrix ---
    plot_df = df_bilateral[df_bilateral['acronym'].isin(anatomical_leaves)].set_index('acronym').copy()

    if plot_df.empty:
        print("[ERROR] No overlapping data found for the specified anatomical regions. Cannot generate plot.")
        return
    print(f"Found {len(plot_df)} matching regions in the data to plot.")
    
    # --- 4. Y-axis Clustering (Data-driven, BASED ON PERCENTAGE ONLY) ---
    # The features for clustering are now only the axon percentages
    features = plot_df[['percentage']] 
    scaled_features = StandardScaler().fit_transform(features)
    data_dist_condensed = pdist(scaled_features, metric='euclidean')
    data_linkage = linkage(data_dist_condensed, method='ward')
    similarity_matrix = 1 / (1 + squareform(data_dist_condensed))
    
    # --- 5. X-axis Ordering (Anatomy-driven) ---
    anat_dist_matrix = np.zeros((len(anatomical_leaves), len(anatomical_leaves)))
    for i in range(len(anatomical_leaves)):
        for j in range(len(anatomical_leaves)):
            anat_dist_matrix[i, j] = abs(i - j)
    anat_dist_condensed = squareform(anat_dist_matrix)
    anat_linkage = linkage(anat_dist_condensed, method='average')
    
    # --- 6. Plot the 2D Dendrogram ---
    plot_similarity_df = pd.DataFrame(similarity_matrix, index=plot_df.index, columns=plot_df.index)
    valid_anatomical_cols = [col for col in anatomical_leaves if col in plot_similarity_df.columns]
    plot_similarity_df = plot_similarity_df.reindex(columns=valid_anatomical_cols)
    
    g = sns.clustermap(
        plot_similarity_df, row_linkage=data_linkage, col_linkage=anat_linkage,
        cmap=c_map, figsize=(20, 20), cbar_kws={'label': 'Axon Percentage Similarity'}
    )
    
    g.fig.suptitle(f"Axon Percentage vs. Anatomical Clustering (Root: {parent_region})", fontsize=20, weight='bold', y=0.98)
    g.ax_heatmap.set_xlabel("Regions (Clustered Anatomically)", fontsize=12)
    g.ax_heatmap.set_ylabel("Regions (Clustered by Axon Percentage)", fontsize=12)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "anatomical_clustering_percentage_2D.png")
    g.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] 2D Clustering plot saved to: {plot_path}")
    plt.show()

# --- NEW 1D DENDROGRAM HEATMAP FUNCTION ---
def plot_1d_cluster_heatmap(df_hemispheres, output_dir, parent_region='grey', max_depth=4, c_map='viridis'):
    """
    Creates a 1D clustered heatmap for a specific anatomical subdivision.
    """
    print("\n--- Generating 1D Functional Clustering Visualization ---")

    # --- 1. Aggregate Hemisphere Data into Bilateral Data ---
    print("[INFO] Aggregating hemisphere data into bilateral results...")
    df_hemispheres['base_acronym'] = df_hemispheres['acronym'].str.replace(r'_left|_right$', '', regex=True)
    
    df_bilateral = df_hemispheres.groupby('base_acronym').agg(
        id=('id', 'first'), name=('name', 'first'),
        axon_voxels=('axon_voxels', 'sum'), total_voxels=('total_voxels', 'sum')
    ).reset_index()
    df_bilateral.rename(columns={'base_acronym': 'acronym'}, inplace=True)
    df_bilateral['percentage'] = (df_bilateral['axon_voxels'] / df_bilateral['total_voxels'].replace(0, np.nan)) * 100
    df_bilateral['percentage'].fillna(0, inplace=True)
    
    # --- 2. Select Anatomical Regions ---
    atlas = BrainGlobeAtlas('allen_mouse_25um')
    children_map = defaultdict(list)
    for struct_id, struct_info in atlas.structures.items():
        if len(struct_info['structure_id_path']) > 1:
            parent_id = struct_info['structure_id_path'][-2]
            children_map[parent_id].append(struct_info)

    def get_descendants(region_id, current_depth, max_depth):
        if current_depth >= max_depth: return [atlas.structures[region_id]['acronym']]
        children = children_map.get(region_id, [])
        if not children: return [atlas.structures[region_id]['acronym']]
        desc_list = []
        for child in children:
            desc_list.extend(get_descendants(child['id'], current_depth + 1, max_depth))
        return desc_list

    try:
        parent_id = atlas.structures[parent_region]['id']
    except KeyError:
        print(f"[ERROR] Parent region '{parent_region}' not found in atlas.")
        return

    anatomical_leaves = sorted(list(set(get_descendants(parent_id, 0, max_depth))))
    print(f"Found {len(anatomical_leaves)} anatomical regions under '{parent_region}' up to depth {max_depth}.")

    # --- 3. Prepare Data for Plotting ---
    plot_df = df_bilateral[df_bilateral['acronym'].isin(anatomical_leaves)].set_index('name').copy()
    plot_df = plot_df[plot_df['axon_voxels'] > 0] # Filter out empty regions for a cleaner plot

    if plot_df.empty:
        print("[ERROR] No overlapping data with axon signal found for the specified regions.")
        return
    print(f"Found {len(plot_df)} matching regions with axon signal to plot.")
    
    # --- 4. Y-axis Clustering (based on Axon Percentage) ---
    features = plot_df[['percentage']]
    scaled_features = StandardScaler().fit_transform(features)
    data_dist_condensed = pdist(scaled_features, metric='euclidean')
    data_linkage = linkage(data_dist_condensed, method='ward')
    
    # --- 5. Plot the 1D Dendrogram and Heatmap ---
    heatmap_data = plot_df[['percentage']]
    
    g = sns.clustermap(
        heatmap_data,
        row_linkage=data_linkage,
        col_cluster=False,
        cmap=c_map,
        figsize=(8, 14),
        cbar_pos=(0.85, 0.8, 0.05, 0.15),
        cbar_kws={'label': 'Axon Percentage (%)'}
    )
    
    g.fig.suptitle(f"Functional Clustering of Regions in {parent_region}", fontsize=16, weight='bold', y=0.98)
    ax = g.ax_heatmap
    ax.set_xlabel('')
    ax.set_ylabel('Brain Region', fontsize=12)
    ax.tick_params(axis='y', labelsize=10, rotation=0)
    ax.tick_params(axis='x', bottom=False, labelbottom=False) # Hide x-axis ticks and labels

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"clustering_1D_{parent_region}.png")
    g.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] 1D Clustering plot saved to: {plot_path}")
    plt.show()