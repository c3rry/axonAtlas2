import os
import pandas as pd
import neurom as nm
import numpy as np
from pathlib import Path
from bg_atlasapi import BrainGlobeAtlas
from collections import defaultdict
import csv # Added for CSV writing

def calculate_metadata_with_bg_atlas(swc_directory, output_csv_file, atlas_name="allen_mouse_10um"):
    """
    Scans a directory of .swc files, calculates their soma coordinates,
    and uses the BrainGlobeAtlas API to determine the brain region
    (acronym) and hemisphere for each soma.
    
    This function creates a new CSV file from scratch.

    Parameters:
    - swc_directory (str): The path to the folder containing your .swc files.
    - output_csv_file (str): The name for the new CSV file to be created.
    - atlas_name (str): The name of the atlas to use (e.g., "allen_mouse_10um").
    """
    
    print(f"Initializing Brain Globe Atlas: '{atlas_name}'...")
    try:
        # 1. Initialize the Atlas
        # This will download the atlas on the first run.
        atlas = BrainGlobeAtlas(atlas_name)
    except Exception as e:
        print(f"Error initializing atlas. Is 'bg-atlasapi' installed?")
        print(f"You might need to run: pip install bg-atlasapi")
        print(f"Error details: {e}")
        return

    swc_dir_path = Path(swc_directory)
    
    # --- 2. Extract Soma Coordinates from all SWC files ---
    
    print(f"Scanning '{swc_directory}' for .swc files...")
    swc_files = list(swc_dir_path.glob("*.swc"))
    
    if not swc_files:
        print(f"Error: No .swc files found in '{swc_directory}'.")
        return

    print(f"Found {len(swc_files)} .swc files. Extracting soma coordinates...")
    
    soma_coords_list = []
    swc_filenames = []
    
    for swc_file in swc_files:
        try:
            # Load the morphology
            morph = nm.load_morphology(str(swc_file))
            
            # Get the soma coordinates (the first point in the file)
            # We assume coordinates are already in atlas microns (um)
            soma_xyz = morph.points[0, :3] 
            
            soma_coords_list.append(soma_xyz)
            swc_filenames.append(swc_file.name)
            
        except Exception as e:
            print(f"  Warning: Could not load {swc_file.name}. Skipping. Error: {e}")

    # Convert list of coordinates to a single Nx3 numpy array for batch lookup
    # This is much faster than looking up one coordinate at a time.
    coordinates_array = np.array(soma_coords_list)

    # --- 3. Get Region and Hemisphere from Atlas ---
    
    print(f"Looking up {len(coordinates_array)} coordinates in atlas...")
    
    # Use lookup_df for efficient batch processing.
    # This returns a DataFrame with 'acronym', 'name', and 'hemisphere'
    atlas_results_df = atlas.lookup_df(coordinates_array)

    # --- 4. Create Final DataFrame ---
    
    print("Assembling final DataFrame...")
    
    # Create a new DataFrame with our file-specific info
    final_df = pd.DataFrame({
        "swc_name": swc_filenames,
        "soma_xyz": [list(coord) for coord in coordinates_array] # Store as list
    })
    
    # Rename atlas columns to be clear
    atlas_results_df = atlas_results_df.rename(columns={
        "acronym": "calculated_structure",
        "hemisphere": "calculated_hemisphere"
    })
    
    # Join our file info with the atlas results
    # We reset indexes to ensure a clean join
    final_df = pd.concat([
        final_df.reset_index(drop=True), 
        atlas_results_df[['calculated_structure', 'calculated_hemisphere']].reset_index(drop=True)
    ], axis=1)

    # --- 5. Create 'structure_hemisphere' Column ---
    
    print("Creating 'structure_hemisphere' column...")
    final_df['calculated_structure_hemisphere'] = (
        final_df['calculated_structure'] + "_" + 
        final_df['calculated_hemisphere']
    )

    # --- 6. Save New CSV File ---
    
    try:
        final_df.to_csv(output_csv_file, index=False)
        print(f"\nSuccessfully created '{output_csv_file}' with {len(final_df)} entries.")
        print("This file was generated *only* from the SWC files and the BrainGlobe Atlas.")
    except Exception as e:
        print(f"Error writing output file: {e}")



# --- Single SWC File Processing Function (Mostly Unchanged) ---
def calculate_axon_length_per_region(swc_filepath, atlas, scale_factor=1000.0):
    """
    Calculates axon length per region for a single SWC file.
    (Helper function for the main directory processing function).

    Args:
        swc_filepath (str): Path to the SWC file.
        atlas (BrainGlobeAtlas): Pre-initialized BrainGlobeAtlas object.
        scale_factor (float): Factor to divide SWC coordinates by.

    Returns:
        dict: Region acronyms mapped to axon length (in scaled units, assumed um),
              or None on critical error, {} if no axons.
    """
    if not os.path.exists(swc_filepath):
        print(f"Error: SWC file not found at {swc_filepath}")
        return None
    try:
        names = ['id', 'type', 'x', 'y', 'z', 'radius', 'parent']
        df = pd.read_csv(swc_filepath, sep='\s+', comment='#', names=names, index_col='id', engine='python')
        df['parent'] = pd.to_numeric(df['parent'], errors='coerce').fillna(-1).astype(int)
    except Exception as e:
        print(f"Error reading SWC file {swc_filepath}: {e}")
        return None

    # Apply scaling
    if scale_factor != 1.0:
        # print(f"  Scaling coordinates by dividing by {scale_factor}...") # Can be verbose
        df[['x', 'y', 'z']] = df[['x', 'y', 'z']] / scale_factor

    axon_df = df[df['type'] == 2].copy()
    if axon_df.empty:
        return {}

    region_lengths = {}
    coords_for_lookup = []
    segment_data = []
    skipped_parents = 0
    skipped_coords = 0

    for point_id, point_data in axon_df.iterrows():
        parent_id = point_data['parent']
        if parent_id == -1 or parent_id not in df.index:
            skipped_parents += 1
            continue
        try:
            p1 = pd.to_numeric(point_data[['x', 'y', 'z']], errors='coerce').values
            p2_series = df.loc[parent_id, ['x', 'y', 'z']]
            p2 = pd.to_numeric(p2_series, errors='coerce').values
            if np.isnan(p1).any() or np.isnan(p2).any():
                 skipped_coords += 1
                 continue
        except KeyError:
            skipped_parents += 1
            continue
        except Exception as e:
             skipped_parents += 1
             continue

        midpoint = (p1 + p2) / 2.0
        length = np.linalg.norm(p1 - p2) # Length is in scaled units (assumed um)
        if not np.all(np.isfinite(midpoint)) or not np.isfinite(length) or length < 0:
            skipped_coords += 1
            continue
        coords_for_lookup.append(midpoint)
        segment_data.append({'length': length})

    if not coords_for_lookup:
        return {}

    coords_array = np.array(coords_for_lookup)
    acronyms = []
    try:
        raw_acronyms = atlas.structures_from_coords(coords_array, as_acronym=True)
        acronyms = [a if a is not None else "Outside_Atlas" for a in raw_acronyms]
    except AttributeError:
         acronyms = []
         for coord in coords_array:
             try:
                 structure = atlas.structure_from_coords(coord, as_acronym=True)
                 acronyms.append(structure if structure else "Outside_Atlas")
             except (IndexError, ValueError, KeyError):
                 acronyms.append("Outside_Atlas")
             except Exception as e_lookup:
                 acronyms.append("Lookup_Error")
    except Exception as e_batch:
         print(f"Unexpected Error during batch atlas lookup for {swc_filepath}: {e_batch}")
         return None

    assert len(acronyms) == len(segment_data), f"Length mismatch in {os.path.basename(swc_filepath)}."
    for i, acronym in enumerate(acronyms):
        length = segment_data[i]['length']
        region_lengths[acronym] = region_lengths.get(acronym, 0.0) + length

    return region_lengths


# --- New Main Function to Process Directory and Output CSV ---
def process_swc_directory_to_csv(directory_path, output_csv_path, atlas_name="allen_mouse_25um", scale_factor=1000.0):
    """
    Processes all SWC files in a directory, calculates axon length per region,
    and saves the results to a CSV file in a matrix format (files x regions).

    Args:
        directory_path (str): Path to the directory containing SWC files.
        output_csv_path (str): Path where the output CSV file will be saved.
        atlas_name (str): Name of the BrainGlobe Atlas to use.
        scale_factor (float): Factor to divide SWC coordinates by.
                               NOTE: 1000.0 is typical for nm -> um. Using 25.0
                               implies SWC coords are in units of atlas resolution * nm.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return

    try:
        print(f"Initializing atlas: {atlas_name}...")
        atlas = BrainGlobeAtlas(atlas_name)
        print(f"Atlas '{atlas_name}' initialized successfully.")
    except Exception as e:
        print(f"Error initializing atlas '{atlas_name}': {e}")
        return

    all_files_results = {} # Store results per file: {filename: {region: length, ...}}
    all_regions = set() # Keep track of all unique regions found
    swc_files_found = []

    print(f"\nSearching for SWC files in: {directory_path}")
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.swc'):
                swc_files_found.append(os.path.join(root, file))

    if not swc_files_found:
        print("No SWC files found in the specified directory.")
        return
    else:
        print(f"Found {len(swc_files_found)} SWC files.")

    processed_count = 0
    error_count = 0
    for swc_file in swc_files_found:
        filename = os.path.basename(swc_file)
        print(f"Processing: {filename}...")
        lengths_single_file = calculate_axon_length_per_region(
            swc_file,
            atlas=atlas,
            scale_factor=scale_factor
        )

        if lengths_single_file is None:
            print(f"  -> Skipped due to error.")
            error_count += 1
            continue
        elif not lengths_single_file:
             print(f"  -> Skipped (no axon data found).")
             all_files_results[filename] = {} # Store empty dict for files with no axons
             processed_count += 1
             continue

        all_files_results[filename] = lengths_single_file
        all_regions.update(lengths_single_file.keys()) # Add new regions found in this file
        processed_count += 1
        print(f"  -> Done.")

    print(f"\n--- Processing Summary ---")
    print(f"Successfully processed: {processed_count - error_count}/{len(swc_files_found)} files.")
    if error_count > 0:
        print(f"Files skipped due to errors: {error_count}")

    if not all_files_results:
        print("No results to write to CSV.")
        return

    # --- Prepare data for CSV output ---
    # Sort regions alphabetically for consistent column order, put Outside_Atlas last if present
    sorted_regions = sorted(list(all_regions - {"Outside_Atlas"}))
    if "Outside_Atlas" in all_regions:
        sorted_regions.append("Outside_Atlas")
    if "Lookup_Error" in all_regions:
         sorted_regions.append("Lookup_Error") # Add lookup error column if needed


    header = [''] + sorted_regions # First column header is empty (for filename column)
    csv_data = [header]

    # Sort filenames alphabetically for consistent row order
    sorted_filenames = sorted(all_files_results.keys())

    for filename in sorted_filenames:
        row = [filename] # Start row with filename
        file_results = all_files_results[filename]
        for region in sorted_regions:
            # Get length for this region, default to 0.0 if not found in this file's results
            length = file_results.get(region, 0.0)
            row.append(f"{length:.4f}") # Format length to 4 decimal places
        csv_data.append(row)

    # --- Write to CSV ---
    try:
        print(f"\nWriting results to: {output_csv_path}")
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)
        print("CSV file written successfully.")
    except Exception as e:
        print(f"Error writing CSV file: {e}")


# --- Example Usage ---
target_directory = r"C:\Users\GangliaGuardian\Downloads\mlnb-export-data" # <-- Your specific path
output_csv = "axon_length_per_region_matrix.csv" # <-- Name for the output file
atlas_resolution = "allen_mouse_25um"

# Set scale factor to 25 as requested by user
# Note: 1000.0 is standard for nm -> um. A factor of 25.0 implies SWC coordinates
# might be in units of (atlas_resolution * nm) or require a different transformation.
coordinate_scale_factor = 25.0

# --- !! Make sure atlas is installed !! ---
# brainglobe install -a allen_mouse_25um

print(f"\n======= Calculating Axon Lengths and Creating CSV =======")
process_swc_directory_to_csv(
    target_directory,
    output_csv,
    atlas_name=atlas_resolution,
    scale_factor=coordinate_scale_factor
)

print("\n======= Script Finished =======")