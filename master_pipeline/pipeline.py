import subprocess
import json
import os
import sys
import datetime

PROCESSING_CONDA_ENV = r"C:\ProgramData\anaconda3\envs\trailmap_env\python.exe"
QUANTIFICATION_CONDA_ENV = r"C:\ProgramData\anaconda3\envs\bgheatmap-env\python.exe"
VISUALIZATION_CONDA_ENV = r"C:\ProgramData\anaconda3\envs\BrainRender\python.exe"

INPUT_TIFF_PATH = r"F:\102925_smartspim\sample-8\brain8.tif"

EXPERIMENT_NAME = "brain8_test_run_master"
EXPERIMENT_DIR = r"F:\axonAtlas2\experiments"

PROCESSING_PARAMS = {
    # surgicalmask
    "erosion_iterations": 30,
    # clahe
    "clahe_clip_limit": 0.02,
    "clahe_kernel_size": (32, 32, 32),
    # gaussian blurring
    "blur_sigma": 2,
    "subtract_bg": True,
    #post-processing
    "downsample_factor": 0.5,
    "dim_compose_factor": 0.5,
    # registration
    "atlas_name": "allen_mouse_25um",
    "orientation": "sal",
    "voxel1": 10,
    "voxel2": 10,
    "voxel3": 10,
    #skeletonization
    "skel_top_bins": 5
}

QUANTIFICATION_PARAMS = {
    "voxel_size": 25,
    "temp_dir": r"F:\axonAtlas2\tmp",
    # similarity matrix
    "similarity_metric":"cosine",
    "similarity_cmap": "Reds", 
    "similarity_max_depth": 10,
    "similarity_parent_region": "grey",
    # dendrogram
    "dendrogram_cmap": "viridis",
    "dendrogram_max_depth": 10,
    "dendrogram_parent_region": "grey",
    # density heatmap
    "density_heatmap_cmap": "Blues",
    "density_heatmap_orientation": 180,
    "density_heatmap_view": "frontal",
    "density_vmin": 0,
    "density_vmax": 0.2,
    # fraction heatmap
    "fraction_heatmap_cmap": "Reds",
    "fraction_heatmap_orientation": 180,
    "fraction_heatmap_view": "frontal",
    "fraction_vmin": 0,
    "fraction_vmax": 0.1
}

def main():
    print("Starting AxonAtlas2 Master Pipeline...")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_exp_name = f"{EXPERIMENT_NAME}_{timestamp}"
    exp_dir = os.path.join(EXPERIMENT_DIR, full_exp_name)
    
    paths = {
        "root": exp_dir,
        "processed": os.path.join(exp_dir, "processed_data"),
        "registration": os.path.join(exp_dir, "registration_output"),
        "quantification": os.path.join(exp_dir, "quantification_results"),
        "logs": os.path.join(exp_dir, "logs"),
        "config_file": os.path.join(exp_dir, "experiment_config.json")
    }
    
    for k, p in paths.items():
        if k != "config_file": os.makedirs(p, exist_ok=True)
    print(f"Experiment directory created at {exp_dir}")
    
    config = {
        "experiment_info": {"name": full_exp_name, "paths": paths},
        "io": {"input_file": INPUT_TIFF_PATH},
        "processing": PROCESSING_PARAMS,
        "quantification": QUANTIFICATION_PARAMS
    }
    with open(paths["config_file"], "w") as f:
        json.dump(config, f, indent=4)
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    driver_proc = os.path.join(script_dir, "processing_driver.py")
    driver_quant = os.path.join(script_dir, "quantification_driver.py")
    
    print("Running Processing Driver...")
    try:
        subprocess.run([PROCESSING_CONDA_ENV, driver_proc, paths["config_file"]], check=True)
    except subprocess.CalledProcessError:
        print("Processing Step Failed. Aborting.")
        return
    print("Processing Completed Successfully.")
    print("Running Quantification Driver...")
    try:
        subprocess.run([QUANTIFICATION_CONDA_ENV, driver_quant, paths["config_file"]], check=True)
    except subprocess.CalledProcessError:
        print("Quantification Step Failed. Aborting.")
        return
    print("Quantification Completed Successfully.")
    print("AxonAtlas2 Master Pipeline Finished Successfully.")
if __name__ == "__main__":
    main()