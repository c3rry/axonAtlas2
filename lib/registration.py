import os
from auxilary import run_command_line
import os

import os
import subprocess

def Registration(
    autof_path: str,
    v1: int,
    v2: int,
    v3: int,
    orientation: str,
    output_dir: str,
    axon_path: str = None,
    atlas: str = "allen_mouse_10um",
    n_free_cpus: int = 8,
    debug: bool = False,
    save_original_orientation: bool = False,
    brain_geometry: str = "full",
    pre_processing: str = "default"
):
    """
    Constructs and runs a brainreg command for image registration.

    Args:
        autof_path (str): Path to the autofluorescence image.
        v1 (int): Voxel size in Z (microns).
        v2 (int): Voxel size in Y (microns).
        v3 (int): Voxel size in X (microns).
        orientation (str): Orientation string for the input data (e.g., 'sal').
        output_dir (str): Base output directory for results.
        axon_path (str, optional): Path to the secondary (e.g., axon) image. Defaults to None.
        atlas (str, optional): Name of the atlas to use. Defaults to "allen_mouse_25um".
        n_free_cpus (int, optional): Number of CPU cores to leave free. Defaults to 2.
        debug (bool, optional): If True, runs in debug mode with verbose logging. Defaults to False.
        save_original_orientation (bool, optional): If True, saves registered data in the
                                                    original orientation. Defaults to False.
        brain_geometry (str, optional): Defines the brain volume to process.
                                        Options: 'full', 'hemisphere_l', 'hemisphere_r'. Defaults to 'full'.
        pre_processing (str, optional): Specifies the preprocessing method.
                                        Options: 'default', 'skip'. Defaults to 'default'.
    """
    # --- Input Validation ---
    if brain_geometry not in ["full", "hemisphere_l", "hemisphere_r"]:
        raise ValueError("brain_geometry must be one of 'full', 'hemisphere_l', or 'hemisphere_r'")
    if pre_processing not in ["default", "skip"]:
        raise ValueError("pre_processing must be one of 'default' or 'skip'")

    # --- Directory Setup ---
    reg_dir = os.path.join(output_dir, "registration")
    os.makedirs(reg_dir, exist_ok=True)

    # --- Command Construction ---
    # Using a list is cleaner for adding optional flags
    command_parts = [
        "brainreg",
        f'"{os.path.normpath(autof_path).replace(os.sep, "/")}"',
        f'"{os.path.normpath(reg_dir).replace(os.sep, "/")}"',
        "--atlas", atlas,
        "-v", str(v1), str(v2), str(v3),
        "--orientation", orientation,
        "--n-free-cpus", str(n_free_cpus),
        "--brain_geometry", brain_geometry,
        "--pre-processing", pre_processing,
    ]

    # Add boolean flags if they are set to True
    if debug:
        command_parts.append("--debug")
    if save_original_orientation:
        command_parts.append("--save-original-orientation")

    # Add the optional axon path
    if axon_path:
        command_parts.extend(["-a", f'"{os.path.normpath(axon_path).replace(os.sep, "/")}"'])

    command = " ".join(command_parts)
    print(f"Constructed brainreg command: {command}")

    # --- Command Execution ---
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    stdout, stderr, returncode = result.stdout, result.stderr, result.returncode

    if returncode != 0:
        print(f"Brainreg command failed with return code {returncode}.")
        print("--- STDOUT ---")
        print(stdout)
        print("\n--- STDERR ---")
        print(stderr)
    else:
        print("Brainreg command executed successfully.")
        return reg_dir

def regExtract(directory):
    target_files = {
        'item1': 'downsampled.tiff',
        'item2': 'registered_atlas.tiff',
        'item3': 'boundaries.tif',
        'item4': 'downsampled_standard.tiff',
        'item5': 'downsampled_standard_'
    }
    
    found = {k: None for k in target_files}
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            lower_file = file.lower()
            # Check items 1-4
            for item, pattern in list(target_files.items())[:4]:
                if file == pattern:
                    found[item] = os.path.join(root, file)
            # Check item5 pattern
            if 'downsampled_standard_' in file and lower_file.endswith(('.tiff', '.tif')):
                found['item5'] = os.path.join(root, file)
                
    return [
        found['item1'],
        found['item2'],
        found['item3'],
        found['item4'],
        found['item5']
    ]
