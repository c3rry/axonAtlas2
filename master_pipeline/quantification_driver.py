import sys
import os
import json
import shutil
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from lib.quantification import *

def run_quantification(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    paths = config["experiment_info"]["paths"]
    params = config["quantification"]
    print("Loading handoff data...")
    handoff = config.get("handoff", {})
    
    out_quant = paths["quantification"]
    
    if not handoff:
        print("No handoff data found. Aborting Quantification Step.")
        sys.exit(1)
    skeleton_path = handoff.get("skeleton_path")
    print(f"Using axon skeleton path: {skeleton_path}")
    print("Starting Quantification Step for Experiment: {config['experiment_info']['name']}")
    
    print("STEP 1: Axon Quantification")
    print("Per Region Quantification...")
    region_df = calculate_axon_percentage(input_volume = skeleton_path,
                                          output_dir = out_quant,
                                          temp_dir = params["temp_dir"],
                                          voxel_size=params["voxel_size"])
    region_csv_path = os.path.join(out_quant, "axon_percentage_per_region.csv")
    region_df.to_csv(region_csv_path, index=False)
    print(f"Axon Percentage per Region saved to {region_csv_path}")
    
    print("Per Region Per Hemisphere Quantification...")
    hemi_df = calculate_hemisphere_axon_percentage(input_volume = skeleton_path,
                                                       output_dir = out_quant,
                                                       temp_dir = params["temp_dir"],
                                                       voxel_size=params["voxel_size"])
    hemi_csv_path = os.path.join(out_quant, "axon_percentage_per_region_per_hemisphere.csv")
    hemi_df.to_csv(hemi_csv_path, index=False)
    print(f"Axon Percentage per Region per Hemisphere saved to {hemi_csv_path}")    
    
    axonDict = axonDict(hemi_df)
    axonDict_frac = axonDict_fraction(hemi_df)
    
    print("STEP 2: Parent Region Similarity Matrix")
    plot_anatomical_similarity_matrix(df_hemispheres=hemi_df, 
                                      axonDict=axonDict, 
                                      output_dir=out_quant,
                                      cmap=params["similarity_cmap"],
                                      max_depth=params["similarity_max_depth"],
                                      parent_region=params["similarity_parent_region"])
    
    print("STEP 3: Parent Region Dendrogram")
    plot_1d_cluster_heatmap(df_hemispheres=hemi_df,
                            output_dir=out_quant,
                            parent_region=params["dendrogram_parent_region"],
                            cmap=params["dendrogram_cmap"],
                            max_depth=params["dendrogram_max_depth"])
    
    print("STEP 4: Density Heatmap Generation")
    os.path.makedirs(os.path.join(out_quant, "density_heatmaps"), exist_ok=True)
    density_heatmap_path = os.path.join(out_quant, "density_heatmaps")
    bg_heatmap_slices(values=axonDict, 
                      view=params["density_heatmap_view"], 
                      output_dir=density_heatmap_path,
                      orientation=params["density_heatmap_orientation"],
                      cmap=params["density_heatmap_cmap"],
                      vmin=params["density_vmin"],
                      vmax=params["density_vmax"])
    print(f"Density Heatmaps saved to {density_heatmap_path}")
    
    print("STEP 5: Fractional Heatmap Generation")
    os.path.makedirs(os.path.join(out_quant, "fraction_heatmaps"), exist_ok=True)
    fraction_heatmap_path = os.path.join(out_quant, "fraction_heatmaps")
    bg_heatmap_slices(values=axonDict_frac, 
                      view=params["fraction_heatmap_view"], 
                      output_dir=fraction_heatmap_path,
                      orientation=params["fraction_heatmap_orientation"],
                      cmap=params["fraction_heatmap_cmap"],
                      vmin=params["fraction_vmin"],
                      vmax=params["fraction_vmax"])
    
    print(f"Fractional Heatmaps saved to {fraction_heatmap_path}")
    print("Quantification Steps Completed.")
    
if __name__ == "__main__":
    run_quantification(sys.argv[1])