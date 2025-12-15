import os
import numpy as np
import tifffile
from skimage import morphology, filters, exposure, util
from scipy import ndimage

def _normalize_and_save(img_data, original_path, output_dir, suffix):
    """
    Helper function to normalize data to standard uint16 range and save it.
    """
    filename = os.path.basename(original_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_{suffix}{ext}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Normalizing and saving to: {os.path.basename(output_path)} ---")
    
    # Handle float data (common output from filters like Sato)
    if img_data.dtype.kind == 'f':
        # Normalize to 0-1 range first
        img_min, img_max = img_data.min(), img_data.max()
        if img_max > img_min:
            img_data = (img_data - img_min) / (img_max - img_min)
        # Convert to uint16 (standard for fluorescence TIFFs)
        img_data = (img_data * 65535).astype(np.uint16)
    
    # Save using tifffile to handle 3D stacks efficiently
    tifffile.imwrite(output_path, img_data, photometric='minisblack')
    print("Save complete.")
    return output_path

def log_transform(tiff_path, output_dir):
    """
    Applies a logarithmic transform to compress dynamic range.
    Good for when the outline is massively brighter than the internal signal.
    """
    print(f"\n[Log Transform] Starting for: {os.path.basename(tiff_path)}")
    
    # 1. Read Image
    print("Reading TIFF stack...")
    img = tifffile.imread(tiff_path)
    print(f"Image loaded. Shape: {img.shape}, Dtype: {img.dtype}")

    # 2. Apply Log Transform
    # We use log1p (log(1+x)) to safely handle zero-value pixels without getting -inf
    print("Applying log1p transform...")
    img_float = img.astype(np.float32)
    log_img = np.log1p(img_float)

    # 3. Save
    return _normalize_and_save(log_img, tiff_path, output_dir, suffix="log")

def white_tophat(tiff_path, output_dir, radius=5, slice_by_slice=True):
    """
    Applies a White Top-Hat transform to remove large bright background structures.
    
    Parameters:
    - radius: Size of the features you want to KEEP. Anything larger than this 
              (like the thick background shell) will be removed.
    - slice_by_slice: If True, processes 2D frames individually (faster, less RAM).
                      If False, uses a 3D ball (better for 3D isotropic data, but slower).
    """
    print(f"\n[White Top-Hat] Starting for: {os.path.basename(tiff_path)}")
    
    img = tifffile.imread(tiff_path)
    print(f"Image loaded. Shape: {img.shape}")

    if slice_by_slice:
        print(f"Processing slice-by-slice with 2D disk radius={radius}...")
        footprint = morphology.disk(radius)
        processed_img = np.zeros_like(img)
        
        for i in range(img.shape[0]):
            if i % 50 == 0: # Print progress every 50 frames
                print(f"Processing slice {i}/{img.shape[0]}...")
            processed_img[i] = morphology.white_tophat(img[i], footprint=footprint)
    else:
        print(f"Processing full 3D volume with ball radius={radius}...")
        # warning: 3D top-hat is very RAM intensive on large stacks
        footprint = morphology.ball(radius)
        processed_img = morphology.white_tophat(img, footprint=footprint)

    return _normalize_and_save(processed_img, tiff_path, output_dir, suffix="tophat")

def sato_filter(tiff_path, output_dir, sigmas=range(1, 5, 1)):
    """
    Applies Sato tubeness filtering to highlight ridge-like (axon) structures.
    
    Parameters:
    - sigmas: Iterable of floats (e.g., range(1,5)). These match the expected 
              widths (in pixels) of the axons you want to find.
    """
    print(f"\n[Sato Filter] Starting for: {os.path.basename(tiff_path)}")
    print("Note: Sato filtering 3D volumes is computationally intensive.")

    img = tifffile.imread(tiff_path)
    print(f"Image loaded. Shape: {img.shape}")

    # Sato works best on normalized float data
    img_float = util.img_as_float32(img)

    print(f"Running Sato filter with sigmas={list(sigmas)}...")
    # black_ridges=False because axons are bright on dark background
    filtered_img = filters.sato(img_float, sigmas=sigmas, black_ridges=False, mode='reflect')
    print("Sato filtering complete.")

    return _normalize_and_save(filtered_img, tiff_path, output_dir, suffix="sato")

def surgical_mask(tiff_path, output_dir, erosion_iterations=10):
    """
    Creates a binary mask of the whole tissue and erodes it significantly 
    to physically cut off the bright outer shell.
    
    Parameters:
    - erosion_iterations: How many pixels deep to "shave" off the exterior.
                          Increase this if the bright outline is very thick.
    """
    print(f"\n[Surgical Mask] Starting for: {os.path.basename(tiff_path)}")
    
    img = tifffile.imread(tiff_path)
    print(f"Image loaded. Shape: {img.shape}")

    # 1. Create coarse mask of the tissue
    # We use a simple Otsu threshold to find "tissue" vs "empty background"
    print("Calculating threshold for initial tissue mask...")
    # Subsample [::10] for speed in calculating threshold
    thresh = filters.threshold_otsu(img[::10]) 
    print(f"Otsu threshold value: {thresh}")
    
    binary_mask = img > thresh

    # 2. Erode the mask to remove the 'shell'
    print(f"Eroding mask by {erosion_iterations} iterations (shaving the edges)...")
    # Using scipy.ndimage.binary_erosion is often faster for simple iterative erosion
    eroded_mask = ndimage.binary_erosion(binary_mask, iterations=erosion_iterations)

    # 3. Apply mask to original image
    print("Applying eroded mask to original data...")
    # where mask is False, set image to 0
    masked_img = np.where(eroded_mask, img, 0)

    return _normalize_and_save(masked_img, tiff_path, output_dir, suffix=f"masked_eroded{erosion_iterations}")