import sys
import os
import json
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from lib.processing import * 
from lib.auxilary import * 
from lib.registration import *
from lib.segmentation import *

def run_processing(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    paths = config["experiment_info"]["paths"]
    params = config["processing"]
    input_file = config["io"]["input_file"]
    
    out_proc = paths["processed"]
    out_reg = paths["registration"]
    
    print("Starting Processing Step for: {input_file}")
    
    print("Validating TrailMap Installation...")
    check_trailmap()
    print("TrailMap Installation Validated.")
    
    print("STEP 1: Surgical Masking")
    file_mask = "01_surgical_mask.tif"
    path_mask = surgical_mask(tiff_path_or_array=input_file,
                              output_dir=out_proc,
                              erosion_iterations=params["erosion_iterations"], 
                              output_filename=file_mask)
    print(f"Surgical Mask saved to {path_mask}")
    
    #print("STEP 2:Contrast Limited Adaptive Histogram Equalization (CLAHE)")
    #file_clahe = "02_clahe.tif"
    #path_clahe = CLAHE(input_path=path_mask,
                       #output_dir=out_proc,
                       #output_filename=file_clahe,
                       #clip_limit=params["clahe_clip_limit"],
                       #kernel_size=params["clahe_kernel_size"])
    #print(f"CLAHE saved to {path_clahe}")
    
    #print("STEP 3: Subtractive Gaussian Blurring")
    #file_blur = "03_blur.tif"
    #path_blur = gaussian_blur(input_path=path_clahe,
                              #output_dir=out_proc,
                              #output_filename=file_blur,
                              #sigma=params["blur_sigma"],
                              #subtract_background=params["subtract_bg"])
    #print(f"Gaussian Blur saved to {path_blur}")
    
    print("STEP 4: Axon Segmentation")
    path_seg = axonSegment(tiff_file_path=path_mask,
                                 output_base_dir=out_proc)
    print(f"Axon Segmentation saved to {path_seg}")
    
    print("STEP 5: Downscaling")
    print("Dowscaling raw volume...")
    file_raw_down = "05a_downscaled_raw.tif"
    path_raw_down = downscale(input_path=input_file,
                               output_dir=out_proc,
                               output_filename=file_raw_down,
                               scale_factor=params["downsample_factor"])
    print(f"Downsampled raw volume saved to {path_raw_down}")
    print("Downscaling segmented volume...")
    file_seg_down = "05b_downscaled_seg.tif"
    path_seg_down = downscale(input_path=path_seg,
                               output_dir=out_proc,
                               output_filename=file_seg_down,
                               scale_factor=params["downsample_factor"])
    print(f"Downsampled segmented volume saved to {path_seg_down}")
    
    print("STEP 6: Dimmed Composition")
    file_seg_down_binary = "06a_downsampled_seg_binary.tif"
    path_seg_down_binary = binarize(file_path=path_seg_down,
                                    output_dir=out_proc,
                                    output_filename=file_seg_down_binary)
    print(f"Binarized downsampled segmented volume saved to {path_seg_down_binary}")
    file_compose = "06b_dim_compose.tif"
    path_compose = dimCompose(mask_path=path_seg_down_binary,
                              image_path=path_raw_down,
                                output_dir=out_proc,
                                output_filename=file_compose,
                                dim_factor=params["dim_compose_factor"])
    print(f"Dimmed Composed volume saved to {path_compose}")
    
    print("STEP 7: Registration")
    reg_dir = Registration(autof_path=path_compose,
                           output_dir=out_reg,
                           v1=params["voxel1"],
                           v2=params["voxel2"], 
                           v3=params["voxel3"],
                           atlas=params["atlas_name"],
                           orientation=params["orientation"])
    
    print(f"Registration outputs saved to {reg_dir}")
    reg_list = regExtract(reg_dir)
    print("Registration Outputs:")
    for key, value in reg_list.items():
        print(f"  {key}: {value}")
    
    print("STEP 8: Skeletonization")
    d_strd = reg_list[3]
    path_skel = skeletonize2(input_path=d_strd,
                             output_dir=out_proc,
                             num_top_bins_to_combine=params["skel_top_bins"])
    print(f"Skeletonization output saved to {path_skel}")
    config["handoff"] = {
        "axon_skeleton": path_skel
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
        
    
    print("Processing Steps Completed.")
    
if __name__ == "__main__":
    config_file_path = sys.argv[1]
    run_processing(config_file_path)