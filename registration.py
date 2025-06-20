import os
from auxilary import run_command_line
def Registration(
    autof_path: str,
    v1: int,
    v2: int,
    v3: int,
    orientation: str,
    output_dir: str,
    axon_path: str = None,
    atlas: str = "allen_mouse_25um"
):
    """
    Constructs and runs a brainreg command for image registration.

    Args:
        autof_path (str): Path to the autofluorescence image.
        v1 (int): Value for the first '-v' argument (e.g., voxel size in Z).
        v2 (int): Value for the second '-v' argument (e.g., voxel size in Y).
        v3 (int): Value for the third '-v' argument (e.g., voxel size in X).
        orientation (str): Orientation string (e.g., 'sar').
        output_dir (str): Base output directory for results.
        axon_path (str, optional): Path to the axon image. Defaults to None.
        atlas (str, optional): Name of the atlas to use. Defaults to "allen_mouse_25um".
    """
    # Ensure output_dir/registration exists
    reg_dir = os.path.join(output_dir, "registration")
    os.makedirs(reg_dir, exist_ok=True)

    # Build the command
    command = (
        f"brainreg {os.path.normpath(autof_path).replace(os.sep, '/')}"
        f" {os.path.normpath(reg_dir).replace(os.sep, '/')}"
        f" --atlas {atlas} -v {v1} {v2} {v3} --orientation {orientation}"
    )
    if axon_path:
        command += f" -a {os.path.normpath(axon_path).replace(os.sep, '/')}"

    print(f"Constructed brainreg command: {command}")
    stdout, stderr, returncode = run_command_line(command)
    if returncode != 0:
        print(f"Brainreg command failed with return code {returncode}.")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
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
