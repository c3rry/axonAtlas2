import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import seaborn as sns
import os
from bg_atlasapi import BrainGlobeAtlas
import brainglobe_heatmap as bgh


def calculate_axon_percentage(axon_mask_path, atlas_name='allen_mouse_25um'):
    """
    Calculate percentage of axon voxels per brain region in 25um atlas space.
    
    Args:
        axon_mask_path (str): Path to 3D binary axon segmentation TIFF file
        atlas_name (str): Name of BrainGlobe atlas (default: 'allen_mouse_25um')
    
    Returns:
        pd.DataFrame: Results with all atlas metadata plus axon statistics
    """
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

def plot_top_regions(df, top_n=20):
    plt.figure(figsize=(12, 8))
    top_df = df.nlargest(top_n, 'percentage')
    ax = sns.barplot(x='percentage', y='acronym', data=top_df, palette='viridis')
    plt.title(f'Top {top_n} Regions by Axon Percentage')
    plt.xlabel('Axon Percentage (%)')
    plt.ylabel('Brain Region')
    plt.tight_layout()
    plt.show()

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
    figsize=(10, 8)
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
                left_dict, position=None, orientation=orientation, hemisphere='left',
                thickness=thickness, atlas_name=atlas_name, format=format, cmap=cmap,
                annotate_regions=annotate_regions, vmin=vmin, vmax=vmax
            )
            # Plot on the single axis, but don't show the colorbar yet
            hm_left.plot_subplot(fig=fig, ax=ax, show_cbar=False)

            # --- Plot Right Hemisphere ---
            hm_right = bgh.Heatmap(
                right_dict, position=None, orientation=orientation, hemisphere='right',
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
        
