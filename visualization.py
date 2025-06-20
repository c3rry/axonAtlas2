import numpy as np
import pandas as pd
from brainrender import Scene
from brainrender import settings
from brainrender.actors import Volume
from imio import load
from bg_atlasapi import BrainGlobeAtlas
settings.SHOW_AXES = False
import vedo
import shutil
from vedo import embedWindow
from rich import print
from myterial import orange
from pathlib import Path
import nibabel as nib
import tifffile
import os
import datetime
embedWindow(None) 

import os

def modeExtract(directory, mode="threshold"):
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

import os
import shutil
import numpy as np
import tifffile
import nibabel as nib
from brainrender import Scene
from brainrender.actors import Volume
from bg_atlasapi import BrainGlobeAtlas
import gc
import random
import matplotlib.pyplot as plt
import re

def sanitize_filename(name):
    """Replace invalid filename characters with underscores."""
    return re.sub(r'[\\/:"*?<>|]', '_', name)

def volume3D(input_volume, temp_dir, voxel_size, regions, output_dir, colormaps=None):
    """
    Visualize a 3D volume, mask by specified brain regions, and save screenshots.

    Args:
        input_volume (str): Path to TIFF, NIfTI, or .npy file.
        temp_dir (str): Temporary directory for intermediate files.
        voxel_size (int): Atlas voxel size (e.g., 10 or 25).
        regions (list): List of region acronyms (e.g., ["TH", "GPe", "CTX"]).
        output_dir (str): Directory to save screenshots.
        colormaps (list or None): List of colormaps, one per region. If None, random solid color colormaps are assigned.

    Returns:
        None
    """
    print(f"[INFO] Starting volume3D with input_volume: {input_volume}")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"[INFO] Temporary directory created at: {temp_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory created at: {output_dir}")
    screenshots_dir = os.path.join(output_dir, "3dscreenshoots")
    os.makedirs(screenshots_dir, exist_ok=True)
    print(f"[INFO] Screenshots will be saved in: {screenshots_dir}")

    # Handle colormaps
    solidcolor_cmaps = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'Greys']
    available_cmaps = [c for c in solidcolor_cmaps if c in plt.colormaps()]
    if not colormaps:
        print("[INFO] No colormaps provided, generating random solid color colormaps for each region.")
        colormaps = []
        for region in regions:
            cmap = random.choice(available_cmaps)
            colormaps.append(cmap)
            print(f"[INFO] Assigned solid color colormap '{cmap}' to region '{region}'")
    else:
        print(f"[INFO] Using provided colormaps: {colormaps}")

    try:
        # 1. Load Input Volume
        ext = os.path.splitext(input_volume)[1].lower()
        print(f"[INFO] Detected input file extension: {ext}")
        if ext in [".tif", ".tiff"]:
            print(f"[INFO] Loading TIFF file: {input_volume}")
            data = tifffile.imread(input_volume)
        elif ext in [".nii", ".nii.gz"]:
            print(f"[INFO] Loading NIfTI file: {input_volume}")
            data = nib.load(input_volume).get_fdata()
        elif ext == ".npy":
            print(f"[INFO] Loading NumPy file: {input_volume}")
            data = np.load(input_volume)
        else:
            print(f"[ERROR] Unsupported file format: {ext}")
            raise ValueError("Unsupported file format for input_volume")
        print(f"[INFO] Loaded input data with shape: {data.shape} and dtype: {data.dtype}")

        # 2. Save as NIfTI for uniformity
        nifti_path = os.path.join(temp_dir, "image1.nii.gz")
        print(f"[INFO] Saving input data as NIfTI to: {nifti_path}")
        nib.save(nib.Nifti1Image(data, affine=np.eye(4)), nifti_path)

        # 3. Mask by Brain Region and Save
        print(f"[INFO] Loading NIfTI for masking: {nifti_path}")
        neuron_nifti = nib.load(nifti_path)
        neuron = neuron_nifti.get_fdata()
        print(f"[INFO] Loaded NIfTI data with shape: {neuron.shape}")
        print(f"[INFO] Loading BrainGlobe Atlas with voxel size: {voxel_size}")
        bg_atlas = BrainGlobeAtlas(f"allen_mouse_{voxel_size}um", check_latest=False)
        region_paths = []
        
        for i, region in enumerate(regions):
            print(f"[INFO] Processing region: {region}")
            region_id = bg_atlas.structures[region]['id']
            print(f"[INFO] Region '{region}' has ID: {region_id}")
            mask = bg_atlas.get_structure_mask(region_id)
            print(f"[INFO] Mask shape for region '{region}': {mask.shape}")
            masked_neuron = np.copy(neuron)
            masked_neuron[~mask.astype(bool)] = 0
            safe_region = sanitize_filename(region)
            region_path = os.path.join(temp_dir, f"{safe_region}_axonTHINNED.nii")
            print(f"[INFO] Saving masked region NIfTI to: {region_path}")
            region_nifti = nib.Nifti1Image(masked_neuron, affine=neuron_nifti.affine)
            nib.save(region_nifti, region_path)
            region_paths.append(region_path)
        del neuron_nifti, neuron, data
        print("[INFO] Finished masking all regions and saved NIfTI files.")

        # 4. Visualize and Save Screenshots
        print("[INFO] Initializing brainrender Scene.")
        scene = Scene(inset=False)
        for i, region_path in enumerate(region_paths):
            print(f"[INFO] Loading region NIfTI for visualization: {region_path}")
            vol = nib.load(region_path).get_fdata()
            vol = np.swapaxes(vol, 0, 2)
            print(f"[INFO] Adding Volume actor for region '{regions[i]}' with colormap '{colormaps[i]}'")
            actor = Volume(
                vol,
                voxel_size=voxel_size,
                as_surface=False,
                min_value=100,
                c=colormaps[i] if i < len(colormaps) else "Greys",
            )
            scene.add(actor)
            del vol

        views = ["top", "frontal", "sagittal"]
        for view in views:
            try:
                print(f"[INFO] Rendering scene for view: {view}")
                scene.render(interactive=False, camera=view, zoom=1)
                screenshot_path = os.path.join(screenshots_dir, f"screenshot_{view}.png")
                scene.screenshot(screenshot_path)
                print(f"[INFO] Saved screenshot: {screenshot_path}")
            except KeyError:
                print(f"[WARNING] View '{view}' is not supported in this version of brainrender.")

        print("[INFO] Rendering interactive scene (final step).")
        scene.render(interactive=True, camera='top', zoom=1)
        print("[INFO] Interactive rendering complete.")

    finally:
        print("[INFO] Cleaning up temporary files and directories.")
        gc.collect()
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"[INFO] Deleted temporary directory: {temp_dir}")
            except Exception as e:
                print(f"[ERROR] Could not delete temp_dir: {e}")

import os
from PIL import Image
import matplotlib.pyplot as plt

def screenshotExtract(directory):
    """
    Given a directory, look for a folder named '3dscreenshoots' inside it,
    and visualize all PNG files in that folder.

    Args:
        directory (str): Path to the directory to search.

    Returns:
        None
    """
    folder_path = os.path.join(directory, '3dscreenshoots')
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"[ERROR] Folder '3dscreenshoots' not found in {directory}")
        return

    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    if not png_files:
        print(f"[INFO] No PNG files found in {folder_path}")
        return

    print(f"[INFO] Found {len(png_files)} PNG files in {folder_path}. Displaying images...")

    # Display all PNG images in a grid
    n = len(png_files)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(5 * cols, 5 * rows))
    for i, png_file in enumerate(png_files):
        img_path = os.path.join(folder_path, png_file)
        img = Image.open(img_path)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(png_file)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

import pandas as pd

def topNRegions(csv_path, N=10):
    """
    Reads a CSV file and returns the top N acronyms by axon percentage.
    
    Parameters:
        csv_path (str): Path to the CSV file containing at least 'acronym' and 'percentage' columns.
        N (int): Number of top regions to return.
    
    Returns:
        List[str]: List of top N acronyms sorted by descending axon percentage.
    """
    df = pd.read_csv(csv_path)
    top_rows = df.sort_values('percentage', ascending=False).head(N)
    return top_rows['acronym'].tolist()

