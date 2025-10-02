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


def calculate_hemisphere_axon_percentage(input_volume, output_dir, voxel_size=25, temp_dir=None):
    """
    Splits every brain region by the midline and calculates axon percentages for each side.
    """
    print("\n\n--- Starting Analysis for HEMISPHERE-SPLIT Brain Regions ---")
    print(f"[INFO] Input volume: {input_volume}")
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    final_df = pd.DataFrame()

    try:
        data = tifffile.imread(input_volume)
        bg_atlas = BrainGlobeAtlas(f"allen_mouse_{voxel_size}um", check_latest=False)
        print(f"[INFO] Loaded data with shape: {data.shape}")

        # Universal hemisphere split by dividing the atlas volume in half
        atlas_shape = bg_atlas.reference.shape
        sagittal_midpoint = atlas_shape[2] // 2
        
        left_hemisphere_mask = np.zeros(atlas_shape, dtype=bool)
        left_hemisphere_mask[:, :, :sagittal_midpoint] = True
        
        right_hemisphere_mask = np.zeros(atlas_shape, dtype=bool)
        right_hemisphere_mask[:, :, sagittal_midpoint:] = True
        print(f"[INFO] Successfully created hemisphere masks by splitting atlas at midline: {sagittal_midpoint}.")

        # --- THIS IS THE FIX ---
        # Robustly get all acronyms by iterating through structure metadata
        all_acronyms = [s['acronym'] for s in bg_atlas.structures.values() if 'acronym' in s]
        # Then create the set of base acronyms from that complete list
        base_acronyms = sorted(list(set(re.sub(r'(_left|_right|_l|_r)$', '', ac, flags=re.IGNORECASE) 
                                        for ac in all_acronyms)))
        print(f"[INFO] Found {len(base_acronyms)} base regions to process.")

        for i, base_acronym in enumerate(base_acronyms):
            if base_acronym in ['root', 'CH']: continue
            print(f"--- Processing {base_acronym} ({i+1}/{len(base_acronyms)}) ---")
            
            try:
                structure_mask = bg_atlas.get_structure_mask(base_acronym)
            except KeyError:
                continue

            for hemi_name, hemi_mask, hemi_suffix in [('Left', left_hemisphere_mask, '_left'), ('Right', right_hemisphere_mask, '_right')]:
                final_mask = np.logical_and(structure_mask, hemi_mask)
                total_voxels = np.sum(final_mask)

                if total_voxels > 0:
                    axon_voxels = np.count_nonzero(np.logical_and(data > 0, final_mask))
                    results.append({
                        'acronym': f"{base_acronym}{hemi_suffix}",
                        'id': bg_atlas.structures[base_acronym]['id'],
                        'name': bg_atlas.structures[base_acronym]['name'],
                        'hemisphere': hemi_name,
                        'axon_voxels': axon_voxels,
                        'total_voxels': total_voxels,
                        'percentage': (axon_voxels / total_voxels) * 100.0
                    })
        
        if results:
            cols = ['acronym', 'id', 'name', 'hemisphere', 'axon_voxels', 'total_voxels', 'percentage']
            final_df = pd.DataFrame(results)[cols].sort_values(by='percentage', ascending=False)
            output_csv_path = os.path.join(output_dir, "axon_percentages_hemispheres.csv")
            final_df.to_csv(output_csv_path, index=False)
            print(f"\n[SUCCESS] Saved hemisphere analysis to: {output_csv_path}")

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