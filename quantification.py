import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import seaborn as sns
import os
from bg_atlasapi import BrainGlobeAtlas

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
    
    # Calculate axon percentages per region
    for region_id, region_info in lookup_df.iterrows():
        # Skip background/root regions
        if region_id == 0 or region_info['acronym'] == 'root': 
            continue
        
        # Create region mask
        region_mask = (annotation == region_id)
        total_voxels = np.sum(region_mask)
        
        # Skip empty regions
        if total_voxels == 0:
            continue
            
        # Calculate axon voxels in region
        axon_in_region = np.logical_and(axon_mask, region_mask)
        axon_voxels = np.sum(axon_in_region)
        
        # Calculate percentage
        percentage = (axon_voxels / total_voxels) * 100
        
        # Print progress for current region
        print(f"{region_info['acronym']}: {percentage:.4f}%")
        
        # Combine all info from lookup_df with new metrics
        result = region_info.to_dict()
        result.update({
            'axon_voxels': axon_voxels,
            'total_voxels': total_voxels,
            'percentage': percentage
        })
        results.append(result)
    
    # Create final dataframe and sort
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('percentage', ascending=False)
    
    print(f"\nCompleted! Processed {len(result_df)} regions.")
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

def bg_heatmap(
    values,
    vmin=None,
    vmax=None,
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
            format=format,
        ).show()

