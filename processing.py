import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.io import imread, imsave
from tifffile import imread, imwrite
from skimage.morphology import skeletonize_3d
def binarize(file_path: str, output_dir: str):
    """
    Binarizes an entire TIFF stack using Otsu's threshold method for each frame
    and saves the result as a new TIFF stack. The output name will be
    "[original_tiff_name]_binary.tif".

    Args:
        file_path (str): Path to the TIFF stack file.
        output_dir (str): Directory where the binarized TIFF stack will be saved.
    """
    if not os.path.exists(file_path):
        print(f"Error: Input file not found at '{file_path}'")
        return
    if not os.path.isdir(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist.")
        return

    try:
        # Read the TIFF stack
        with tifffile.TiffFile(file_path) as tif:
            original_stack = tif.asarray()

        if original_stack.ndim < 2:
            print(f"Error: TIFF data from '{file_path}' is not at least 2-dimensional. Cannot binarize.")
            return

        # Ensure the stack is iterable, even if it's a 2D image (single frame)
        if original_stack.ndim == 2:
            frames_to_process = np.expand_dims(original_stack, axis=0) # Make it 3D (1, Y, X)
        else: # Assumes (Z, Y, X) or similar
            frames_to_process = original_stack

        binarized_frames = []
        print(f"Binarizing {frames_to_process.shape[0]} frames from '{file_path}'...")
        for i, frame in enumerate(frames_to_process):
            # Ensure frame is 2D for Otsu thresholding
            if frame.ndim > 2:
                # If frame has more than 2 dimensions (e.g., color channels),
                # convert to grayscale or pick a channel. For simplicity, convert to grayscale.
                print(f"Warning: Frame {i} has {frame.ndim} dimensions. Converting to grayscale for binarization.")
                from skimage.color import rgb2gray
                if frame.shape[-1] in [3, 4]: # Assuming last dimension is color channel
                    frame = rgb2gray(frame)
                else:
                    print(f"Could not convert frame {i} to grayscale. Skipping binarization for this frame.")
                    binarized_frames.append(np.zeros_like(frame, dtype=bool)) # Append a blank frame
                    continue

            # Apply Otsu's threshold
            threshold = threshold_otsu(frame)
            binarized_frame = frame > threshold
            binarized_frames.append(binarized_frame)
            if (i + 1) % 100 == 0 or i == frames_to_process.shape[0] - 1:
                print(f"  Processed {i+1}/{frames_to_process.shape[0]} frames...")

        # Stack the binarized frames back into a single stack
        binarized_stack = np.stack(binarized_frames).astype(np.uint8) * 255 # Convert boolean to 0/255 for TIFF

        # Determine output file name
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file_name = f"{base_name}_binary.tif"
        output_file_path = os.path.join(output_dir, output_file_name)

        # Save the binarized stack
        tifffile.imwrite(output_file_path, binarized_stack)
        print(f"Successfully created binarized TIFF stack: '{output_file_path}'.")
        return output_file_path

    except Exception as e:
        print(f"An error occurred during binarization: {e}")


def dimCompose(mask_path, image_path, output_dir, dim_factor=0.5):
    """
    Highlights pixels in image_path that correspond to white pixels in mask_path,
    and dims all other pixels. Saves composite to output_dir and returns its path.
    
    Parameters:
    - mask_path (str): path to the binary mask TIFF stack
    - image_path (str): path to the source TIFF stack
    - output_dir (str): directory to save the composite output
    - dim_factor (float): multiplier for dimming non-mask pixels (0.0 to 1.0)
    
    Returns:
    - str: full path to the saved composite TIFF
    """
    # Load the TIFF stacks
    mask_stack = tifffile.imread(mask_path)
    image_stack = tifffile.imread(image_path)

    # Validate dimensions
    if mask_stack.shape != image_stack.shape:
        raise ValueError("mask_stack and image_stack must have the same dimensions")

    # Convert image to uint8 if needed
    image_stack = image_stack.astype(np.uint8)

    # Create binary mask (True where mask is > 0)
    mask_binary = (mask_stack > 0)

    # Initialize composite with dimmed version of image
    composite_stack = (image_stack * dim_factor).astype(np.uint8)

    # Set masked pixels to white
    composite_stack[mask_binary] = 255

    # Construct output path
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{base_name}_composite.tif"
    output_path = os.path.join(output_dir, output_filename)

    # Save composite stack
    tifffile.imwrite(output_path, composite_stack)

    return output_path

import os
import numpy as np
from tifffile import imread, imwrite

def threshold(input_path, output_dir, threshold=100):
    """
    Process a TIFF stack by thresholding pixel values and save the result.
    
    Args:
        input_path (str): Path to input TIFF stack
        output_dir (str): Directory to save processed stack
        threshold (int): Values < threshold become 0, >= become 255 (default: 100)
    
    Returns:
        str: Path to the processed TIFF stack
    
    Requires:
        pip install tifffile numpy
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    orig_name = os.path.basename(input_path)
    name, ext = os.path.splitext(orig_name)
    output_path = os.path.join(output_dir, f"{name}_threshold{ext}")
    
    # Read TIFF stack
    stack = imread(input_path)
    
    # Process stack using vectorized operations
    processed = np.where(stack < threshold, 0, 255).astype(np.uint8)
    
    # Save processed stack with same metadata
    imwrite(output_path, processed, photometric='minisblack')
    
    return output_path

def arrayStats(input_path):
    """
    Print basic statistics of a numpy array.
    
    Args:
        arr (np.ndarray): Input numpy array
        name (str): Name to identify the array in output (default: "Array")
    """

    arr = imread(input_path)

    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Min: {np.min(arr)}")
    print(f"  Max: {np.max(arr)}")
    print(f"  Mean: {np.mean(arr)}")
    print(f"  Std Dev: {np.std(arr)}")
    print(f"  Non-zero count: {np.count_nonzero(arr)}")
    print(f"arr: {arr}")



def skeletonize2(input_path, output_dir, num_top_bins_to_combine =5):
    """
    Processes a multipage TIFF by separating pixels into 10 bins, saving each
    bin as a separate file, and also creating a composite TIFF from the sum
    of the top N multiplied bins.

    Args:
        input_path (str): Path to the input multi-page TIFF file.
        output_dir (str): Directory where the output subfolder will be created.
        num_top_bins_to_combine (int): The number of top (brightest) bins to
                                       multiply and combine.

    Returns:
        str: Path to the composite TIFF file created from the top bins.

    Requires:
        pip install tifffile numpy
    """
    # --- 1. Setup output path ---
    os.makedirs(output_dir, exist_ok=True)
    orig_name = os.path.basename(input_path)
    name, ext = os.path.splitext(orig_name)
    output_folder = os.path.join(output_dir, f"{name}_binned_tiffs")
    os.makedirs(output_folder, exist_ok=True)

    # --- 2. Load the image stack ---
    print(f"Loading TIFF stack from: {input_path}")
    stack = imread(input_path)
    stack = stack.astype(np.float32)

    # --- 3. Define bins and multipliers for the top bins ---
    bins = np.array([25.5, 51, 76.5, 102, 127.5, 153, 178.5, 204, 229.5])
    
    # These multipliers will be applied to the top N bins.
    # For N=5, this generates [0.2, 0.4, 0.6, 0.8, 1.0]
    top_multipliers = np.linspace(0.2, 1.0, num=num_top_bins_to_combine)

    # --- 4. Isolate bins, save them, and combine top bins ---
    print(f"Saving 10 binned sub-tiffs to: {output_folder}")
    indices = np.digitize(stack, bins)
    
    # Initialize an empty array to hold the sum of the multiplied top bins
    combined_top_bins_stack = np.zeros_like(stack, dtype=np.float32)

    for i in range(10): # Loop through each of the 10 bins
        mask = (indices == i)
        
        # Create and save the individual sub-tiff for the current bin
        sub_stack = np.zeros_like(stack, dtype=np.uint8)
        sub_stack[mask] = stack[mask]

        if i == 0: range_str = f"0.0-{bins[0]}"
        elif i < len(bins): range_str = f"{bins[i-1]}-{bins[i]}"
        else: range_str = f"{bins[-1]}-255.0"

        sub_tiff_path = os.path.join(output_folder, f"bin_{i}_{range_str}.tif")
        imwrite(sub_tiff_path, sub_stack, photometric='minisblack')
        print(f"  - Saved bin {i} to {os.path.basename(sub_tiff_path)}")

        # Check if the current bin is one of the top bins to be combined
        if i >= (10 - num_top_bins_to_combine):
            # Calculate which multiplier to use (e.g., the 5th bin gets the 1st multiplier)
            multiplier_index = i - (10 - num_top_bins_to_combine)
            multiplier = top_multipliers[multiplier_index]
            
            print(f"  - Adding bin {i} to composite with multiplier {multiplier:.2f}")
            
            # Apply multiplier to the pixels in this bin and add to the composite stack
            combined_top_bins_stack[mask] = stack[mask] * multiplier

    # --- 5. Save the combined top-bins tiff ---
    print("\nSaving combined top bins TIFF...")
    combined_path = os.path.join(output_folder, f"combined_top_{num_top_bins_to_combine}_skeletonize.tif")
    
    # Clip values to ensure they are within the 0-255 range and convert to 8-bit
    combined_top_bins_stack = np.clip(combined_top_bins_stack, 0, 255)
    output_stack = combined_top_bins_stack.astype(np.uint8)
    
    imwrite(combined_path, output_stack, photometric='minisblack')
    print(f"Saved composite TIFF to: {combined_path}")

    return combined_path