
import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.io import imread, imsave
import subprocess
import shutil
import datetime
from auxilary import run_command_line

def check_trailmap():
    """
    Checks if a folder named 'TRAILMAP' exists in the current working directory.
    If not found, it attempts to clone the TRAILMAP repository.
    """
    folder_name = "TRAILMAP"
    if os.path.isdir(folder_name):
        print(f"The folder '{folder_name}' was found in the current directory. Proceed!")
        return True
    else:
        print(f"The folder '{folder_name}' was NOT found in the current directory.")
        # Attempt to clone the repository if the folder does not exist
        run_command_line("git clone https://github.com/albert597/TRAILMAP.git")
        print(f"TRAILMAP successfully installed!")
        # Re-check if the folder exists after attempting to clone
        if os.path.isdir(folder_name):
            return True
        else:
            print(f"Error: TRAILMAP folder still not found after cloning attempt.")
            return False


def stackToFolder(file_path: str, output_dir: str) -> str:
    """
    Takes a TIFF stack and saves each frame as a separate TIFF file
    within a new folder named after the original TIFF stack.

    Args:
        file_path (str): Path to the TIFF stack file.
        output_dir (str): Directory where the new folder will be created.

    Returns:
        str: The full path to the created folder, or an empty string if failed.
    """
    if not os.path.exists(file_path):
        print(f"Error: Input file not found at '{file_path}'")
        return ""
    if not os.path.isdir(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist.")
        return ""

    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        folder_name = f"{base_name}_folder"
        full_output_folder_path = os.path.join(output_dir, folder_name)

        os.makedirs(full_output_folder_path, exist_ok=True)
        print(f"Created output folder: '{full_output_folder_path}'")

        with tifffile.TiffFile(file_path) as tif:
            data = tif.asarray()

        if data.ndim == 2: # Handle 2D image as a single frame stack
            data = np.expand_dims(data, axis=0)

        print(f"Processing {data.shape[0]} frames from '{file_path}'...")
        for i, frame in enumerate(data):
            frame_filename = f"{base_name}_frame_{i:04d}.tif" # Padded to 4 digits for consistent sorting
            output_frame_path = os.path.join(full_output_folder_path, frame_filename)
            tifffile.imwrite(output_frame_path, frame)
        print(f"Successfully saved all frames from '{file_path}' to '{full_output_folder_path}'.")
        return full_output_folder_path

    except Exception as e:
        print(f"An error occurred while splitting stack to folder: {e}")
        return ""

def folderToStack(folder_path: str, output_dir: str) -> str:
    """
    Takes a folder of TIFF frames and saves them as a single TIFF stack
    in the specified output directory. The output stack's name will be
    derived from the folder name by removing '_folder'.

    Args:
        folder_path (str): Path to the folder containing TIFF frames.
        output_dir (str): Directory where the new TIFF stack will be saved.

    Returns:
        str: The full path to the created TIFF stack, or an empty string if failed.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Input folder not found at '{folder_path}'")
        return ""
    if not os.path.isdir(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist.")
        return ""

    try:
        frame_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.lower().endswith(('.tif', '.tiff'))]
        frame_files.sort()

        if not frame_files:
            print(f"Error: No TIFF files found in '{folder_path}'.")
            return ""

        frames = [tifffile.imread(f_path) for f_path in frame_files]

        if not frames:
            print("Error: No frames were successfully read.")
            return ""

        # Basic check for consistent frame shapes before stacking
        first_frame_shape = frames[0].shape
        valid_frames = []
        for i, frame in enumerate(frames):
            if frame.shape == first_frame_shape:
                valid_frames.append(frame)
            else:
                print(f"Warning: Frame {i} has inconsistent shape {frame.shape}. Skipping.")

        if not valid_frames:
            print("Error: No valid frames to stack after shape check.")
            return ""

        tiff_stack = np.stack(valid_frames)

        folder_base_name = os.path.basename(folder_path)
        if folder_base_name.endswith("_folder"):
            stack_name = folder_base_name[:-7] # Remove "_folder"
        else:
            stack_name = folder_base_name + "_stack"

        output_file_path = os.path.join(output_dir, f"{stack_name}.tif")

        tifffile.imwrite(output_file_path, tiff_stack)
        print(f"Successfully saved TIFF stack to '{output_file_path}'.")
        return output_file_path

    except Exception as e:
        print(f"An error occurred while combining folder to stack: {e}")
        return ""

def axonSegment(tiff_file_path: str, output_base_dir: str):
    """
    Performs the TRAILMAP inference workflow:
    1. Creates necessary output directories.
    2. Converts a TIFF stack to a folder of individual frames.
    3. Runs the TRAILMAP segmentation command on the frame folder.
    4. Visualizes selected original and segmented frames side-by-side.
    5. Converts the segmented frames folder back into a TIFF stack.
    6. Cleans up intermediate frame folders.


    Args:
        tiff_file_path (str): Path to the input TIFF stack file.
        output_base_dir (str): Base directory where all output folders/files will be created.
                               This directory must exist.
    """
    if not os.path.exists(tiff_file_path):
        print(f"Error: Input TIFF file not found at '{tiff_file_path}'.")
        return
    if not os.path.isdir(output_base_dir):
        print(f"Error: Output base directory '{output_base_dir}' does not exist. Please create it.")
        return


    print(f"\n--- Starting TRAILMAP Inference for '{os.path.basename(tiff_file_path)}' ---")


    # 1. Create all folders within dir (if they don't exist)
    inference_temp_dir = os.path.join(output_base_dir, "trailmap_inference_temp")
    os.makedirs(inference_temp_dir, exist_ok=True)
    print(f"Created temporary directory: '{inference_temp_dir}'")


    # 2. Utilize stackToFolder to convert the tiff_file_path into a tiff folder
    print(f"\nConverting '{os.path.basename(tiff_file_path)}' to frames folder...")
    original_tiff_frames_folder = stackToFolder(tiff_file_path, inference_temp_dir)
    if not original_tiff_frames_folder:
        print("Failed to convert TIFF stack to folder. Aborting inference.")
        return


    # Extract the base name of the original frames folder for the segmentation output
    original_folder_name_base = os.path.basename(original_tiff_frames_folder)


    # 3. Utilize run_command_line to run the command "python TRAILMAP/segment_brain_batch.py {TIFF FOLER PATH HERE}"
    # This assumes TRAILMAP is installed in the current working directory.
    # The output folder name created by segment_brain_batch.py is typically 'seg-{input_folder_name}'.
    # We construct the expected segmented folder path.
    segmented_tiff_frames_folder = os.path.join(inference_temp_dir, f"seg-{original_folder_name_base}")


    print(f"\nRunning TRAILMAP segmentation command...")
    # Using posixpath for path formatting in the command string to ensure cross-platform compatibility
    # for the command itself, even if os.path is used for local filesystem operations.
    command = f"python TRAILMAP/segment_brain_batch.py {os.path.normpath(original_tiff_frames_folder).replace(os.sep, '/')}"
    stdout, stderr, returncode = run_command_line(command)


    if returncode != 0:
        print(f"TRAILMAP segmentation command failed with error code {returncode}. Check STDERR above.")
        print("Please ensure TRAILMAP is correctly installed and its 'segment_brain_batch.py' script is accessible.")
        print("Also verify the input folder format expected by TRAILMAP.")
        # Attempt to clean up even if command failed
        if os.path.exists(original_tiff_frames_folder):
            shutil.rmtree(original_tiff_frames_folder)
            print(f"Cleaned up '{original_tiff_frames_folder}'.")
        return



    # 5. Use folderToStack to convert seg-folder into a tiffstack
    print(f"\nConverting segmented frames folder '{os.path.basename(segmented_tiff_frames_folder)}' back to TIFF stack...")
    final_segmented_stack_path = folderToStack(segmented_tiff_frames_folder, output_base_dir)
    if not final_segmented_stack_path:
        print("Failed to convert segmented folder to TIFF stack. Partial inference completed.")
        return


    # 6. Delete the normal tiff folder and the seg tiff folder
    print("\nCleaning up temporary folders...")
    if os.path.exists(original_tiff_frames_folder):
        shutil.rmtree(original_tiff_frames_folder)
        print(f"Deleted original frames folder: '{original_tiff_frames_folder}'")
    if os.path.exists(segmented_tiff_frames_folder):
        shutil.rmtree(segmented_tiff_frames_folder)
        print(f"Deleted segmented frames folder: '{segmented_tiff_frames_folder}'")
    if os.path.exists(inference_temp_dir):
        shutil.rmtree(inference_temp_dir)
        print(f"Deleted temporary base directory: '{inference_temp_dir}'")




    # 7. Print that inference is completed
    print(f"\n--- TRAILMAP Inference Completed Successfully! ---")
    print(f"Final segmented TIFF stack saved to: '{final_segmented_stack_path}'")
    return final_segmented_stack_path
