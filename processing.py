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



def skeletonize(input_path, output_dir):
    """
    Skeletonize white pixels (255) in a binary TIFF stack (axon images).
    
    Args:
        input_path (str): Path to input binary TIFF stack (white=axon=255, black=0)
        output_dir (str): Directory to save skeletonized stack
    
    Returns:
        str: Path to the skeletonized TIFF stack
    
    Requires:
        pip install tifffile scikit-image numpy
    """
    os.makedirs(output_dir, exist_ok=True)
    orig_name = os.path.basename(input_path)
    name, ext = os.path.splitext(orig_name)
    output_path = os.path.join(output_dir, f"{name}_skeletonize{ext}")
    stack = imread(input_path)
    # Ensure binary (0, 255) -> (0, 1)
    binary_stack = (stack > 0).astype(np.uint8)
    # Skeletonize (works for 2D or 3D)
    skeleton = skeletonize_3d(binary_stack)
    # Convert back to 0, 255 for saving
    skeleton = (skeleton > 0).astype(np.uint8) * 255
    imwrite(output_path, skeleton, photometric='minisblack')
    return output_path