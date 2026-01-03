import os
import re
import shutil
from typing import Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import tifffile
from skimage.color import rgb2gray
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_otsu

# --- Core Helper for Memory Safety & Naming ---

def _save_chunked(img_data, output_dir, output_filename, normalize=False):
    """
    Internal helper to save large 3D stacks slice-by-slice.
    
    Args:
        img_data: The 3D numpy array to save.
        output_dir: Target directory.
        output_filename: The exact name of the file to save (e.g., "step1.tif").
        normalize: If True, scales data to 0-65535 (uint16) based on min/max.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Saving to: {output_path}")

    # Calculate stats only if normalization is requested
    if normalize:
        print("Calculating global statistics for normalization...")
        img_min = img_data.min()
        img_max = img_data.max()
        if img_max == img_min:
            denom = 1.0
        else:
            denom = img_max - img_min
        print(f"  Range: {img_min} to {img_max}")
    
    dtype_out = np.uint16 if normalize else img_data.dtype
    
    try:
        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            num_slices = img_data.shape[0]
            for z in range(num_slices):
                # 1. Get slice
                slice_data = img_data[z]
                
                # 2. Normalize if requested (convert to float, scale, convert back)
                if normalize:
                    slice_data = slice_data.astype(np.float32)
                    slice_data = (slice_data - img_min) / denom
                    slice_data = (slice_data * 65535).astype(np.uint16)
                
                # 3. Write
                tif.write(slice_data, contiguous=True)
                
                if (z + 1) % 100 == 0:
                    print(f"  Saved slice {z + 1}/{num_slices}")
                    
        print("Save complete.")
        return output_path

    except Exception as e:
        print(f"Error during save: {e}")
        return None

# --- Main Processing Functions ---

def binarize(file_path: str, output_dir: str, output_filename: str = None):
    """
    Binarizes a stack using Otsu's method.
    """
    if not os.path.exists(file_path): return
    
    # Naming logic: Use provided name OR short default
    if output_filename is None:
        base = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{base}_bin.tif"

    try:
        with tifffile.TiffFile(file_path) as tif:
            original_stack = tif.asarray()

        if original_stack.ndim == 2:
            frames = np.expand_dims(original_stack, axis=0)
        else:
            frames = original_stack

        binarized_frames = []
        print(f"Binarizing {frames.shape[0]} frames...")
        
        for i, frame in enumerate(frames):
            if frame.ndim > 2:
                frame = rgb2gray(frame) if frame.shape[-1] in [3, 4] else frame
            
            thresh = threshold_otsu(frame)
            binarized_frames.append(frame > thresh)
            
            if (i + 1) % 100 == 0: print(f"  Processed {i+1}...")

        binarized_stack = np.stack(binarized_frames).astype(np.uint8) * 255
        
        # Use simple saver (no normalization needed for binary)
        return _save_chunked(binarized_stack, output_dir, output_filename, normalize=False)

    except Exception as e:
        print(f"Error in binarize: {e}")

def dimCompose(mask_path, image_path, output_dir, output_filename=None, dim_factor=0.5):
    """
    Composites a mask onto an image.
    """
    if output_filename is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f"{base}_comp.tif"

    mask_stack = tifffile.imread(mask_path)
    image_stack = tifffile.imread(image_path)

    if mask_stack.shape != image_stack.shape:
        raise ValueError("Dimensions mismatch")

    image_stack = image_stack.astype(np.uint8)
    mask_binary = (mask_stack > 0)
    composite = (image_stack * dim_factor).astype(np.uint8)
    composite[mask_binary] = 255

    return _save_chunked(composite, output_dir, output_filename, normalize=False)

def threshold(input_path, output_dir, output_filename=None, thresh_val=100):
    """
    Simple hard thresholding.
    """
    if output_filename is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{base}_thresh.tif"
        
    stack = tifffile.imread(input_path)
    processed = np.where(stack < thresh_val, 0, 255).astype(np.uint8)
    return _save_chunked(processed, output_dir, output_filename, normalize=False)

def arrayStats(input_path):
    """Prints stats, no file output."""
    arr = tifffile.imread(input_path)
    print(f"Shape: {arr.shape}, Dtype: {arr.dtype}")
    print(f"Min: {np.min(arr)}, Max: {np.max(arr)}")
    print(f"Mean: {np.mean(arr):.2f}, Non-zero: {np.count_nonzero(arr)}")

def skeletonize2(input_path, output_dir, output_filename=None, num_top_bins_to_combine=5):
    """
    Intensity binning + composite.
    """
    base = os.path.splitext(os.path.basename(input_path))[0]
    
    # Folder for bins (always created)
    bin_folder = os.path.join(output_dir, "bins")
    os.makedirs(bin_folder, exist_ok=True)
    
    if output_filename is None:
        output_filename = f"{base}_skel.tif"

    print(f"Loading: {input_path}")
    stack = tifffile.imread(input_path).astype(np.float32)

    bins = np.array([25.5, 51, 76.5, 102, 127.5, 153, 178.5, 204, 229.5])
    top_multipliers = np.linspace(0.2, 1.0, num=num_top_bins_to_combine)
    indices = np.digitize(stack, bins)
    
    combined_stack = np.zeros_like(stack, dtype=np.float32)

    for i in range(10):
        mask = (indices == i)
        # Save individual bin (keep naming short)
        bin_name = f"bin_{i}.tif"
        sub_stack = np.zeros_like(stack, dtype=np.uint8)
        sub_stack[mask] = stack[mask]
        tifffile.imwrite(os.path.join(bin_folder, bin_name), sub_stack, photometric='minisblack')

        if i >= (10 - num_top_bins_to_combine):
            mult = top_multipliers[i - (10 - num_top_bins_to_combine)]
            combined_stack[mask] = stack[mask] * mult

    combined_stack = np.clip(combined_stack, 0, 255).astype(np.uint8)
    return _save_chunked(combined_stack, output_dir, output_filename, normalize=False)

def max_median_scaler(input_path, output_dir, output_filename=None, min_value=0, max_value=255):
    """
    Replaces max-value pixels with median.
    """
    if output_filename is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{base}_med.tif"

    print(f"Processing: {input_path}")
    data = tifffile.imread(input_path)
    
    valid_mask = (data > min_value) & (data < max_value)
    signal = data[valid_mask]
    
    median_val = np.median(signal) if signal.size > 0 else min_value
    print(f"Median: {median_val}")

    data[data == max_value] = median_val
    return _save_chunked(data, output_dir, output_filename, normalize=False)

def grid_slice_volume(input_path, output_dir, divisions):
    """
    Slices volume into a grid. 
    NOTE: Output creates a FOLDER, not a single file.
    """
    if not os.path.exists(input_path): return None
    
    # Clean folder name
    base = os.path.splitext(os.path.basename(input_path))[0]
    slices_dir = os.path.join(output_dir, f"{base}_slices")
    
    if os.path.exists(slices_dir): shutil.rmtree(slices_dir)
    os.makedirs(slices_dir, exist_ok=True)

    print("Loading and transposing...")
    data = tifffile.imread(input_path)
    data = np.transpose(data, (2, 1, 0)) # Transpose logic preserved
    
    h, w, d = data.shape
    r_h, c_w = h // divisions, w // divisions
    
    for r in range(divisions):
        for c in range(divisions):
            slice_data = data[r*r_h:(r+1)*r_h, c*c_w:(c+1)*c_w, :]
            # Keep slice names standard so 'boost_and_restitch' can find them
            fname = f"slice_row-{r}_col-{c}.tif" 
            tifffile.imwrite(os.path.join(slices_dir, fname), slice_data)
            
    print(f"Slices saved to: {slices_dir}")
    return slices_dir

def visualize_grid(slices_dir, target_row, target_col, cmap='viridis'):
    """Visualization tool. No file output changes needed."""
    if not os.path.isdir(slices_dir): return None
    files = [f for f in os.listdir(slices_dir) if 'slice_row-' in f]
    if not files: return None
    
    # Regex to find grid size
    rows = max(int(re.search(r'row-(\d+)', f).group(1)) for f in files) + 1
    cols = max(int(re.search(r'col-(\d+)', f).group(1)) for f in files) + 1
    
    sample = tifffile.imread(os.path.join(slices_dir, files[0]))
    h, w, _ = sample.shape
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    target_path = None
    
    for r in range(rows):
        for c in range(cols):
            ax = axes[r,c] if rows > 1 and cols > 1 else axes
            fname = f"slice_row-{r}_col-{c}.tif"
            path = os.path.join(slices_dir, fname)
            
            if os.path.exists(path):
                img = tifffile.imread(path)[:, :, sample.shape[2]//2]
                my_cmap = cmap if (r==target_row and c==target_col) else 'gray'
                if r==target_row and c==target_col: target_path = path
                ax.imshow(img, cmap=my_cmap)
            ax.axis('off')
            
    plt.tight_layout()
    plt.show()
    return target_path

def boost_and_restitch(target_slice_path, output_dir, output_filename=None, gamma=0.5, views=['xy']):
    """
    Boosts a specific slice and restitches the grid.
    """
    if not os.path.exists(target_slice_path): return None
    
    print(f"Boosting {os.path.basename(target_slice_path)}...")
    
    # 1. Boost target slice
    data = tifffile.imread(target_slice_path)
    norm = data / np.iinfo(data.dtype).max
    boosted = ((norm ** gamma) * np.iinfo(data.dtype).max).astype(data.dtype)
    tifffile.imwrite(target_slice_path, boosted) # Overwrites the slice!

    # 2. Restitch
    slices_dir = os.path.dirname(target_slice_path)
    files = [f for f in os.listdir(slices_dir) if 'slice_row-' in f]
    rows = max(int(re.search(r'row-(\d+)', f).group(1)) for f in files) + 1
    cols = max(int(re.search(r'col-(\d+)', f).group(1)) for f in files) + 1
    
    sample = tifffile.imread(os.path.join(slices_dir, files[0]))
    sh, sw, d = sample.shape
    
    full_vol = np.zeros((sh*rows, sw*cols, d), dtype=sample.dtype)
    
    for r in range(rows):
        for c in range(cols):
            path = os.path.join(slices_dir, f"slice_row-{r}_col-{c}.tif")
            full_vol[r*sh:(r+1)*sh, c*sw:(c+1)*sw, :] = tifffile.imread(path)

    if output_filename is None:
        output_filename = "restitched.tif"
        
    return _save_chunked(full_vol, output_dir, output_filename, normalize=False)

def downscale(input_path, output_dir, output_filename=None, scale_factor=0.5):
    """
    Downscales volume.
    """
    if output_filename is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{base}_down.tif"

    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Downscaling {input_path}...")
    stack = tifffile.imread(input_path)
    
    # Zoom
    small_stack = scipy.ndimage.zoom(stack, scale_factor, order=1)
    
    # Use chunked saver
    return _save_chunked(small_stack, output_dir, output_filename, normalize=False)

def gaussian_blur(input_path, output_dir, output_filename=None, sigma=1.0, subtract_background=False):
    """
    Streaming Gaussian Blur.
    """
    if output_filename is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{base}_blur.tif" # Simple name

    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle sigma logic
    sigma_2d = (sigma, sigma) if isinstance(sigma, (float, int)) else sigma
    if isinstance(sigma, (tuple, list)) and len(sigma) == 3:
        sigma_2d = (sigma[1], sigma[2])

    print(f"Blurring to: {output_path}")

    try:
        with tifffile.TiffFile(input_path) as tif_reader:
            dtype = tif_reader.series[0].dtype
            min_val = np.iinfo(dtype).min if np.issubdtype(dtype, np.integer) else 0
            max_val = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1
            
            with tifffile.TiffWriter(output_path, bigtiff=True) as tif_writer:
                for i, page in enumerate(tif_reader.pages):
                    img = page.asarray()
                    blur = scipy.ndimage.gaussian_filter(img, sigma=sigma_2d)
                    
                    if subtract_background:
                        res = img.astype(np.float32) - blur.astype(np.float32)
                        out = np.clip(res, min_val, max_val).astype(dtype)
                    else:
                        out = blur.astype(dtype)
                        
                    tif_writer.write(out, contiguous=True)
                    if (i+1)%100==0: print(f"  Processed {i+1}...")
                    
        return output_path
    except Exception as e:
        print(f"Blur Error: {e}")
        return None

def CLAHE(input_path, output_dir, output_filename=None, kernel_size=None, clip_limit=0.01):
    """
    Streaming CLAHE.
    """
    if output_filename is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{base}_clahe.tif"

    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Kernel logic
    k_2d = None
    if isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 3:
        k_2d = (kernel_size[1], kernel_size[2])
    elif isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2:
        k_2d = kernel_size
        
    print(f"CLAHE to: {output_path}")

    try:
        with tifffile.TiffFile(input_path) as tif_reader:
            dtype = tif_reader.series[0].dtype
            info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
            
            with tifffile.TiffWriter(output_path, bigtiff=True) as tif_writer:
                for i, page in enumerate(tif_reader.pages):
                    # Equalize returns float 0-1
                    res = equalize_adapthist(page.asarray(), kernel_size=k_2d, clip_limit=clip_limit)
                    
                    # Convert back
                    if info:
                        out = (res * info.max).astype(dtype)
                    else:
                        out = res.astype(dtype)
                        
                    tif_writer.write(out, contiguous=True)
                    if (i+1)%100==0: print(f"  Processed {i+1}...")

        return output_path
    except Exception as e:
        print(f"CLAHE Error: {e}")
        return None

def surgical_mask(tiff_path_or_array, output_dir, output_filename=None, erosion_iterations=30):
    """
    Surgical mask generation.
    """
    print(f"Generating mask (erosion={erosion_iterations})...")
    
    if isinstance(tiff_path_or_array, str):
        img = tifffile.imread(tiff_path_or_array)
        if output_filename is None:
            base = os.path.splitext(os.path.basename(tiff_path_or_array))[0]
            output_filename = f"{base}_mask.tif"
    else:
        img = tiff_path_or_array
        if output_filename is None:
            output_filename = "processed_mask.tif"

    # Process
    mask = (img > 0)
    eroded = scipy.ndimage.binary_erosion(mask, iterations=erosion_iterations)
    masked_img = np.where(eroded, img, 0)
    
    # Save using the safe helper with Normalization enabled
    return _save_chunked(masked_img, output_dir, output_filename, normalize=True)