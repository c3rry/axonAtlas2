# import os
# import subprocess
# from typing import Union, List

# def cellSegmentation(
#     signal_paths: Union[str, List[str]],
#     background_path: str,
#     output_dir: str,
#     v1: int,
#     v2: int,
#     v3: int,
#     orientation: str,
#     debug: bool = False
# ):
#     """
#     Constructs and runs a brainmapper command for cell segmentation.

#     Args:
#         signal_paths (str or list of str): Path(s) to the signal channel image(s). 
#                                            Can be a single string or a list of strings for multiple paths.
#         background_path (str): Path to the background channel images.
#         output_dir (str): Base output directory for results.
#         v1 (int): Voxel size in Z (microns).
#         v2 (int): Voxel size in Y (microns).
#         v3 (int): Voxel size in X (microns).
#         orientation (str): Orientation string for the input data (e.g., 'asl').
#         debug (bool, optional): If True, adds a debug flag (if supported) or can be used 
#                                 to print extra execution info. Defaults to False.
#     """
    
#     # --- Input Validation & Setup ---
#     # Convert a single string to a list so we can iterate over it cleanly
#     if isinstance(signal_paths, str):
#         signal_paths = [signal_paths]

#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # --- Command Construction ---
#     command_parts = ["brainmapper"]

#     # 1. Add Signal Paths (-s)
#     command_parts.append("-s")
#     for path in signal_paths:
#         command_parts.append(f'"{os.path.normpath(path).replace(os.sep, "/")}"')

#     # 2. Add Background Path (-b)
#     command_parts.extend([
#         "-b", 
#         f'"{os.path.normpath(background_path).replace(os.sep, "/")}"'
#     ])

#     # 3. Add Output Directory (-o)
#     command_parts.extend([
#         "-o", 
#         f'"{os.path.normpath(output_dir).replace(os.sep, "/")}"'
#     ])

#     # 4. Add Voxel Sizes (-v) and Orientation
#     command_parts.extend([
#         "-v", str(v1), str(v2), str(v3),
#         "--orientation", orientation
#     ])

#     # Add debug flag if requested (assuming brainmapper supports it like brainreg)
#     if debug:
#         command_parts.append("--debug")

#     command = " ".join(command_parts)
#     print(f"Constructed brainmapper command: {command}")

#     # --- Command Execution ---
#     result = subprocess.run(command, shell=True, capture_output=True, text=True)
#     stdout, stderr, returncode = result.stdout, result.stderr, result.returncode

#     if returncode != 0:
#         print(f"Brainmapper command failed with return code {returncode}.")
#         print("--- STDOUT ---")
#         print(stdout)
#         print("\n--- STDERR ---")
#         print(stderr)
#     else:
#         print("Brainmapper command executed successfully.")
#         return output_dir

# import os
# import subprocess
# from typing import Union, List

# def cellSegmentation(
#     signal_paths: Union[str, List[str]],
#     background_path: str,
#     output_dir: str,
#     v1: int,
#     v2: int,
#     v3: int,
#     orientation: str,
#     debug: bool = False
# ):
#     if isinstance(signal_paths, str):
#         signal_paths = [signal_paths]

#     os.makedirs(output_dir, exist_ok=True)

#     # Construct command as a LIST, not a string
#     command = ["brainmapper"]

#     # 1. Add Signal Paths
#     command.append("-s")
#     for path in signal_paths:
#         command.append(os.path.abspath(path)) # Let subprocess handle the quotes/slashes

#     # 2. Add Background Path
#     command.extend(["-b", os.path.abspath(background_path)])

#     # 3. Add Output Directory
#     command.extend(["-o", os.path.abspath(output_dir)])

#     # 4. Add Voxel Sizes and Orientation
#     command.extend(["-v", str(v1), str(v2), str(v3), "--orientation", orientation])

#     if debug:
#         command.append("--debug")

#     print(f"Running command: {' '.join(command)}")

#     # Execute WITHOUT shell=True for better path handling
#     result = subprocess.run(command, capture_output=True, text=True, shell=False)

#     if result.returncode != 0:
#         print(f"Brainmapper failed. Error:\n{result.stderr}")
#     else:
#         print("Success.")
#         return output_dir

import os
import subprocess
from typing import Union, List

def cellSegmentation(
    signal_paths: Union[str, List[str]],
    background_path: str,
    output_dir: str,
    v1: int,
    v2: int,
    v3: int,
    orientation: str,
    threshold: int = 15,     # Increased from 10 to ignore dim artifacts
    ball_z: int = 5,         # Reduced for 25um spacing (prevents "squashed" cell filtering)
    soma_dia: int = 16,      # Standard cell diameter in microns
    debug: bool = False
):
    if isinstance(signal_paths, str):
        signal_paths = [signal_paths]

    os.makedirs(output_dir, exist_ok=True)

    # Construct command as a LIST
    command = ["brainmapper"]

    # 1. Paths
    command.append("-s")
    for path in signal_paths:
        command.append(os.path.abspath(path))

    command.extend(["-b", os.path.abspath(background_path)])
    command.extend(["-o", os.path.abspath(output_dir)])

    # 2. Voxel Sizes and Orientation
    # NOTE: If you downscaled by 0.5, ensure v1/v2/v3 reflect the NEW micron size
    command.extend(["-v", str(v1), str(v2), str(v3)])
    command.extend(["--orientation", orientation])

    # 3. Detection & Artifact Filtering
    command.extend(["--threshold", str(threshold)])
    command.extend(["--ball-z-size", str(ball_z)])
    command.extend(["--soma-diameter", str(soma_dia)])

    if debug:
        command.append("--debug")

    print(f"Running command: {' '.join(command)}")

    # Execute
    result = subprocess.run(command, capture_output=True, text=True, shell=False)

    if result.returncode != 0:
        print(f"Brainmapper failed. Error:\n{result.stderr}")
    else:
        print("Success.")
        return output_dir