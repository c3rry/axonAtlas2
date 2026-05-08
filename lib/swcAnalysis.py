import os
import pandas as pd
import neurom as nm
import numpy as np
from pathlib import Path
from bg_atlasapi import BrainGlobeAtlas
from collections import defaultdict
import csv # Added for CSV writing
import random

import numpy as np
import pandas as pd
from pathlib import Path
import neurom as nm
from bg_atlasapi.bg_atlas import BrainGlobeAtlas


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

import numpy as np
import pandas as pd
from pathlib import Path
import neurom as nm
from bg_atlasapi.bg_atlas import BrainGlobeAtlas

import numpy as np
import pandas as pd
from pathlib import Path
import neurom as nm
from bg_atlasapi.bg_atlas import BrainGlobeAtlas

def calculate_metadata_with_bg_atlas(swc_directory, output_csv_file, atlas_name="allen_mouse_25um"):
    """
    Scans a directory of .swc files and calculates:
      - Total Length
      - Start Coordinates, Acronym, Full Name, & Hemisphere
      - End Coordinates, Acronym, Full Name, & Hemisphere
    """
    
    print(f"Initializing Brain Globe Atlas: '{atlas_name}'...")
    try:
        atlas = BrainGlobeAtlas(atlas_name)
    except Exception as e:
        print(f"Error initializing atlas. Is 'bg-atlasapi' installed?")
        print(f"Error details: {e}")
        return

    swc_dir_path = Path(swc_directory)
    swc_files = list(swc_dir_path.glob("*.swc"))
    
    if not swc_files:
        print(f"Error: No .swc files found in '{swc_directory}'.")
        return

    print(f"Found {len(swc_files)} .swc files. Extracting morphology data...")
    
    # Storage lists
    swc_filenames = []
    start_coords = []
    end_coords = []
    lengths = []
    
    for swc_file in swc_files:
        try:
            morph = nm.load_morphology(str(swc_file))
            
            # 1. Get Length
            total_len = nm.get('total_length', morph)
            if isinstance(total_len, (list, np.ndarray)):
                total_len = sum(total_len)
            
            # 2. Get Start Coordinate (Soma)
            soma_xyz = morph.points[0, :3] 
            
            # 3. Get End Coordinate (Farthest point from Soma)
            all_points = morph.points[:, :3]
            distances = np.linalg.norm(all_points - soma_xyz, axis=1)
            farthest_idx = np.argmax(distances)
            end_xyz = all_points[farthest_idx]
            
            # Append to lists
            swc_filenames.append(swc_file.name)
            lengths.append(total_len)
            start_coords.append(soma_xyz)
            end_coords.append(end_xyz)
            
        except Exception as e:
            print(f"  Warning: Could not load {swc_file.name}. Skipping. Error: {e}")

    n_samples = len(start_coords)
    if n_samples == 0:
        print("No valid data found.")
        return

    start_coords_arr = np.array(start_coords)
    end_coords_arr = np.array(end_coords)
    
    # Stack for atlas lookup
    combined_coords = np.vstack([start_coords_arr, end_coords_arr])

    # --- 3. Get Region and Hemisphere from Atlas ---
    print(f"Looking up {len(combined_coords)} coordinates (Start + End) in atlas...")
    
    acronyms = []
    full_names = []  # New list for full names
    hemispheres = []
    z_midpoint = (atlas.shape[2] * atlas.resolution[2]) / 2

    for coord in combined_coords:
        # A. Find Structure Info (ID -> Acronym + Name)
        try:
            # Get the ID first (microns=True for physical units)
            region_id = atlas.structure_from_coords(coord, microns=True)
            
            if region_id is not None and region_id > 0:
                # Look up details in the atlas dictionary
                struct_info = atlas.structures[region_id]
                acronym = struct_info['acronym']
                name = struct_info['name']
            else:
                acronym = "outside atlas"
                name = "outside atlas"
        except Exception:
            print(f"  Warning: Could not find structure for coordinate {coord}")
            #acronym = "outside_brain"
            #name = "outside_brain"
            
        acronyms.append(acronym)
        full_names.append(name)

        # B. Find Hemisphere
        if coord[2] < z_midpoint:
            hemispheres.append("Left")
        else:
            hemispheres.append("Right")

    # Split results back into Start and End
    start_acronyms = acronyms[:n_samples]
    end_acronyms = acronyms[n_samples:]
    
    start_names = full_names[:n_samples] # Split names
    end_names = full_names[n_samples:]
    
    start_hemispheres = hemispheres[:n_samples]
    end_hemispheres = hemispheres[n_samples:]

    # --- 4. Create Final DataFrame ---
    print("Assembling final DataFrame...")
    
    final_df = pd.DataFrame({
        "swc_name": swc_filenames,
        "length_um": lengths,
        
        # Start Metadata (Soma)
        "start_x": start_coords_arr[:, 0],
        "start_y": start_coords_arr[:, 1],
        "start_z": start_coords_arr[:, 2],
        "start_structure": start_acronyms,
        "start_structure_name": start_names, # New Column
        "start_hemisphere": start_hemispheres,
        
        # End Metadata (Tip)
        "end_x": end_coords_arr[:, 0],
        "end_y": end_coords_arr[:, 1],
        "end_z": end_coords_arr[:, 2],
        "end_structure": end_acronyms,
        "end_structure_name": end_names,     # New Column
        "end_hemisphere": end_hemispheres,
    })

    # Optional: Composite columns
    final_df['start_region_hemi'] = final_df['start_structure'].astype(str) + "_" + final_df['start_hemisphere']
    final_df['end_region_hemi'] = final_df['end_structure'].astype(str) + "_" + final_df['end_hemisphere']

    # --- 5. Save New CSV File ---
    try:
        final_df.to_csv(output_csv_file, index=False)
        print(f"\nSuccessfully created '{output_csv_file}' with {len(final_df)} entries.")
    except Exception as e:
        print(f"Error writing output file: {e}")
def generate_point_cloud_metadata(swc_directory, output_csv_file, atlas_name="allen_mouse_25um"):
    """
    Scans a directory of .swc files and creates a row for EVERY single coordinate
    in every file, identifying its brain region acronym, FULL NAME, and hemisphere.
    """
    
    print(f"Initializing Brain Globe Atlas: '{atlas_name}'...")
    try:
        atlas = BrainGlobeAtlas(atlas_name)
    except Exception as e:
        print(f"Error initializing atlas: {e}")
        return

    swc_dir_path = Path(swc_directory)
    swc_files = list(swc_dir_path.glob("*.swc"))
    
    if not swc_files:
        print(f"Error: No .swc files found in '{swc_directory}'.")
        return

    print(f"Found {len(swc_files)} .swc files. Extracting ALL points...")
    
    all_filenames = []
    all_x = []
    all_y = []
    all_z = []
    
    # 1. READ ALL DATA
    for swc_file in swc_files:
        try:
            morph = nm.load_morphology(str(swc_file))
            points = morph.points[:, :3]
            
            n_points = len(points)
            all_filenames.extend([swc_file.name] * n_points)
            all_x.extend(points[:, 0])
            all_y.extend(points[:, 1])
            all_z.extend(points[:, 2])
            
        except Exception as e:
            print(f"  Skipping {swc_file.name}: {e}")

    total_points = len(all_x)
    print(f"\nTotal points to analyze: {total_points}")
    print("Starting atlas lookup... (This may take a while)")

    # 2. PREPARE FOR LOOKUP
    coords_array = np.vstack((all_x, all_y, all_z)).T
    
    acronyms = []
    full_names = [] # New list
    hemispheres = []
    z_midpoint = (atlas.shape[2] * atlas.resolution[2]) / 2

    # 3. RUN LOOKUP LOOP
    for i, coord in enumerate(coords_array):
        if i % 5000 == 0:
            print(f"  Processed {i}/{total_points} points...", end='\r')
            
        # A. Region (ID lookup strategy)
        try:
            region_id = atlas.structure_from_coords(coord, microns=True)
            
            if region_id is not None and region_id > 0:
                struct_info = atlas.structures[region_id]
                acronym = struct_info['acronym']
                name = struct_info['name']
            else:
                acronym = "root"
                name = "root"
        except:
            acronym = "outside_brain"
            name = "outside_brain"
            
        acronyms.append(acronym)
        full_names.append(name)
        
        # B. Hemisphere
        if coord[2] < z_midpoint:
            hemispheres.append("Left")
        else:
            hemispheres.append("Right")
            
    print(f"  Processed {total_points}/{total_points} points. Done.")

    # 4. SAVE
    print("Constructing final DataFrame...")
    df = pd.DataFrame({
        "swc_name": all_filenames,
        "x": all_x,
        "y": all_y,
        "z": all_z,
        "structure": acronyms,
        "structure_name": full_names, # New Column
        "hemisphere": hemispheres
    })
    
    df['structure_hemisphere'] = df['structure'].astype(str) + "_" + df['hemisphere'].astype(str)

    print(f"Saving to '{output_csv_file}'...")
    try:
        df.to_csv(output_csv_file, index=False)
        print(f"Success! Saved {len(df)} rows.")
    except Exception as e:
        print(f"Error saving CSV: {e}")


def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def generate_random_rgb():
    """Generates a random RGB color tuple."""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)

