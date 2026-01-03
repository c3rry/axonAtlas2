
import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.io import imread, imsave
import subprocess
import shutil
import datetime

def tiffVisualize(file_path: str, views: int = 1, colormap: str = 'gray', contrast_factor: float = 1.0, brightness_factor: float = 0.0):
    """
    Visualizes the center frame of a TIFF stack in 1, 2, or 3 planes.

    Args:
        file_path (str): Path to the TIFF stack file.
        views (int): Number of views to create (1=XY, 2=XY+XZ, 3=XY+XZ+YZ).
                     Defaults to 1.
        colormap (str): Matplotlib colormap to apply. Defaults to 'gray'.
                        If an invalid colormap is provided, 'gray' will be used.
        contrast_factor (float): Factor to adjust contrast. A value > 1.0 increases contrast
                                 (narrows the intensity window), < 1.0 decreases contrast
                                 (widens the intensity window). Defaults to 1.0 (no change).
        brightness_factor (float): Value to add/subtract from pixel intensities to adjust brightness.
                                   Positive values increase brightness, negative values decrease.
                                   Defaults to 0.0 (no change).
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    try:
        # Read the TIFF stack
        # Use a context manager for tifffile.TiffFile to ensure proper closing
        with tifffile.TiffFile(file_path) as tif:
            # Check if it's a multi-page TIFF
            if not tif.is_multipage:
                print(f"Warning: '{file_path}' is not a multi-page TIFF. Visualizing as a single image.")
                data = tif.asarray()
                # If it's a 2D image, expand dimensions to make it 3D for consistent slicing
                if data.ndim == 2:
                    data = np.expand_dims(data, axis=0) # Add a Z-dimension
                elif data.ndim > 3: # Handle cases like (frames, Z, Y, X)
                    print("Warning: TIFF has more than 3 dimensions, selecting first 3 for visualization.")
                    data = data[0, :3] # Try to get the first volume
            else:
                data = tif.asarray()

        if data.ndim < 3:
            print(f"Error: TIFF data from '{file_path}' is not 3-dimensional (Z, Y, X). Dimensions found: {data.ndim}. Cannot visualize multiple planes.")
            return

        # Get dimensions
        nz, ny, nx = data.shape
        center_z = nz // 2
        center_y = ny // 2
        center_x = nx // 2

        # Calculate vmin and vmax based on contrast_factor and brightness_factor
        data_min_original = data.min()
        data_max_original = data.max()
        data_range_original = data_max_original - data_min_original

        # First, apply contrast adjustment
        adjusted_vmin_contrast = data_min_original
        adjusted_vmax_contrast = data_max_original

        if data_range_original > 0: # Avoid division by zero for flat images
            new_range = data_range_original / contrast_factor
            center_val = (data_min_original + data_max_original) / 2
            adjusted_vmin_contrast = center_val - (new_range / 2)
            adjusted_vmax_contrast = center_val + (new_range / 2)

        # Then, apply brightness adjustment to the already contrast-adjusted range
        adjusted_vmin = adjusted_vmin_contrast + brightness_factor
        adjusted_vmax = adjusted_vmax_contrast + brightness_factor

        # Clamp adjusted vmin/vmax to original data type limits
        # This prevents display issues if the adjustment goes beyond the valid range for the data type.
        if data.dtype.kind == 'u': # unsigned integer (e.g., uint8, uint16)
            dtype_min = np.iinfo(data.dtype).min
            dtype_max = np.iinfo(data.dtype).max
            adjusted_vmin = max(adjusted_vmin, dtype_min)
            adjusted_vmax = min(adjusted_vmax, dtype_max)
        elif data.dtype.kind == 'f': # float
            # For floats, it's generally okay to let matplotlib handle out-of-range values.
            # If explicit clamping to [0,1] or original data min/max is desired, it can be added here.
            pass


        # Prepare plots
        fig, axes = plt.subplots(1, views, figsize=(5 * views, 5))
        if views == 1:
            axes = [axes] # Make it iterable for consistent loop

        fig.suptitle(f"Visualization of: {os.path.basename(file_path)} (Contrast: {contrast_factor:.1f}, Brightness: {brightness_factor:.1f})", fontsize=16)

        try:
            # Check if colormap is valid
            plt.get_cmap(colormap)
        except ValueError:
            print(f"Warning: Colormap '{colormap}' not found. Using 'gray' instead.")
            colormap = 'gray'

        # XY Plane (always included)
        if views >= 1:
            axes[0].imshow(data[center_z, :, :], cmap=colormap, vmin=adjusted_vmin, vmax=adjusted_vmax)
            axes[0].set_title(f"XY Plane (Z={center_z})")
            axes[0].set_xlabel("X-axis")
            axes[0].set_ylabel("Y-axis")

        # XZ Plane
        if views >= 2:
            axes[1].imshow(data[:, center_y, :], cmap=colormap, origin='lower', aspect='auto', vmin=adjusted_vmin, vmax=adjusted_vmax) # origin='lower' to match spatial orientation
            axes[1].set_title(f"XZ Plane (Y={center_y})")
            axes[1].set_xlabel("X-axis")
            axes[1].set_ylabel("Z-axis")

        # YZ Plane
        if views >= 3:
            axes[2].imshow(data[:, :, center_x], cmap=colormap, origin='lower', aspect='auto', vmin=adjusted_vmin, vmax=adjusted_vmax) # origin='lower' to match spatial orientation
            axes[2].set_title(f"YZ Plane (X={center_x})")
            axes[2].set_xlabel("Y-axis") # This is Y from the XY plane, but it's the 2nd dim of the slice
            axes[2].set_ylabel("Z-axis")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        plt.show()
        print(f"Successfully visualized '{file_path}'.")

    except Exception as e:
        print(f"An error occurred during visualization of '{file_path}': {e}")


def run_command_line(command: str):
    """
    Runs a shell command and returns its output and error.

    Args:
        command (str): The command string to execute.

    Returns:
        tuple: A tuple containing (stdout, stderr, returncode).
               stdout (str): Standard output from the command.
               stderr (str): Standard error from the command.
               returncode (int): The exit code of the command.
    """
    try:
        result = subprocess.run(command, text=True, capture_output=True, shell=True, check=False)

        print(f"Command executed: '{command}'")
        print(f"Return Code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")

        return result.stdout, result.stderr, result.returncode

    except FileNotFoundError:
        print(f"Error: Command '{command.split()[0]}' not found. Make sure it's in your PATH.")
        return None, f"Command '{command.split()[0]}' not found.", 127
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, str(e), -1

def createExperiment(experiment_name: str, base_experiments_dir: str = "experiments") -> str:
    """
    Creates a new folder for an experiment within a specified base directory.
    The new folder will be named with the experiment_name and the current date and time.

    Args:
        experiment_name (str): The desired name for the experiment.
        base_experiments_dir (str): The base directory where the 'experiments' folder
                                    is located or will be created. Defaults to "experiments".

    Returns:
        str: The full **file path** to the newly created experiment folder, or an empty string if creation failed.
    """
    # Ensure the base 'experiments' directory exists
    full_base_path = os.path.join(os.getcwd(), base_experiments_dir)
    os.makedirs(full_base_path, exist_ok=True)
    print(f"Ensured base experiments directory exists: '{full_base_path}'")

    # Get current date and time and format it
    current_time = datetime.datetime.now()
    # Format: YYYY-MM-DD_HH-MM-SS
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Create the new experiment folder name
    # Sanitize experiment_name for file system (replace spaces, etc.)
    safe_experiment_name = experiment_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    new_folder_name = f"{safe_experiment_name}_{timestamp}"

    # Construct the full path for the new experiment folder
    new_experiment_path = os.path.join(full_base_path, new_folder_name)

    try:
        os.makedirs(new_experiment_path, exist_ok=True)
        print(f"Successfully created experiment folder: '{new_experiment_path}'")
        return new_experiment_path
    except Exception as e:
        print(f"Error creating experiment folder '{new_experiment_path}': {e}")
        return ""
