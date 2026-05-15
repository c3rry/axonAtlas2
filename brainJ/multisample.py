import pandas as pd
import os

def process_brainj_raw_data(input_dir=".", output_dir="./Output_StrictRaw_Python/"):
    """
    Process BrainJ raw CSV files. Includes robust voxel column detection 
    to handle different BrainJ output versions (Voxels vs Volume).
    """
    CHANNEL_NUM = 3
    MOUSE_NUM = 7
    
    ATLAS_FILE = os.path.join(input_dir, 'Atlas_Regions.csv')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(ATLAS_FILE):
        print(f"Error: {ATLAS_FILE} not found.")
        return
        
    atlas_df = pd.read_csv(ATLAS_FILE)
    atlas_df.columns = [c.lower() for c in atlas_df.columns]
    rename_map = {'id': 'Region_ID', 'acronym': 'Acronym', 'name': 'Name'}
    atlas_df = atlas_df.rename(columns=rename_map).set_index('Region_ID')[['Acronym', 'Name']]

    for c in range(1, CHANNEL_NUM + 1):
        print(f"Processing Channel {c}...")
        channel_df = atlas_df.copy()
        
        for m in range(1, MOUSE_NUM + 1):
            f_cells = os.path.join(input_dir, f"M{m}_C{c}_Detected_Cells_Summary.csv")
            f_dens = os.path.join(input_dir, f"M{m}_C{c}_Measured_Projection_Density.csv")
            f_int = os.path.join(input_dir, f"M{m}_C{c}_Measured_Region_Intensity.csv")
            
            # --- Cells ---
            if os.path.exists(f_cells):
                df_c = pd.read_csv(f_cells).set_index('ID')
                channel_df[f'M{m}_Cell_Left'] = df_c['Total_Cells_Left']
                channel_df[f'M{m}_Cell_Right'] = df_c['Total_Cells_Right']

            # --- Density & Voxels ---
            if os.path.exists(f_dens):
                df_d = pd.read_csv(f_dens).set_index('ID')
                
                # Metrics to extract
                metrics_to_map = {
                    'Projection_Density_Left': f'M{m}_ProjDensity_Left',
                    'Projection_Density_Right': f'M{m}_ProjDensity_Right',
                    'Relative_Density_Left': f'M{m}_RelDensity_Left',
                    'Relative_Density_Right': f'M{m}_RelDensity_Right'
                }
                
                # Flexible Voxel Detection: Handles 'Region_Voxels', 'Voxels', or 'Volume'
                v_left_col = next((col for col in df_d.columns if ('Voxel' in col or 'Volume' in col) and 'Left' in col), None)
                v_right_col = next((col for col in df_d.columns if ('Voxel' in col or 'Volume' in col) and 'Right' in col), None)
                
                if v_left_col: metrics_to_map[v_left_col] = f'M{m}_Voxels_Left'
                if v_right_col: metrics_to_map[v_right_col] = f'M{m}_Voxels_Right'

                for src, dest in metrics_to_map.items():
                    if src in df_d.columns:
                        channel_df[dest] = df_d[src]

            # --- Intensity ---
            if os.path.exists(f_int):
                df_i = pd.read_csv(f_int).set_index('ID')
                channel_df[f'M{m}_Intensity_Left'] = df_i['Mean_Intensity_Left']
                channel_df[f'M{m}_Intensity_Right'] = df_i['Mean_Intensity_Right']

        # Fill missing values and save
        output_path = os.path.join(output_dir, f"Channel{c}_StrictRaw.csv")
        channel_df.fillna(0).reset_index().to_csv(output_path, index=False)

def create_source_region_averages(input_dir=".", output_file="Source_Region_Averages.csv"):
    """
    Creates the summary CSV with averages for each metric.
    Includes: Cell, ProjDensity, RelDensity, Voxels, Intensity.
    """
    ch_mapping = {1: 'GPi', 2: 'SNr', 3: 'GPe'}
    # Included 'Voxels' in the metric list
    metrics = ['Cell', 'ProjDensity', 'RelDensity', 'Voxels', 'Intensity']
    
    final_summary = None
    
    for ch_num, source_name in ch_mapping.items():
        fname = os.path.join(input_dir, f"Channel{ch_num}_StrictRaw.csv")
        if not os.path.exists(fname): 
            print(f"File {fname} not found, skipping...")
            continue
        
        df = pd.read_csv(fname)
        
        if final_summary is None:
            # Initialize with primary keys
            final_summary = df[['Region_ID', 'Acronym', 'Name']].copy()
            
        for metric in metrics:
            # Matches columns like M1_Voxels_Left, M2_Voxels_Right, etc.
            cols = [c for c in df.columns if f'_{metric}_' in c]
            
            if cols:
                col_name = f'average_{source_name}_{metric}'
                final_summary[col_name] = df[cols].mean(axis=1)
            else:
                print(f"Note: No columns found for metric '{metric}' in Channel {ch_num}")
                
    if final_summary is not None:
        final_summary.to_csv(output_file, index=False)
        print(f"Successfully saved averages to {output_file}")

# Example usage:
# process_brainj_raw_data(input_dir="path/to/data", output_dir="output/path")
# create_source_region_averages(input_dir="output/path")

def create_source_region_averages(input_dir="./Output_StrictRaw_Python/", output_file="Source_Region_Averages.csv"):
    """
    Takes ChannelX_StrictRaw.csv files and creates a summary file with 
    averages across all mice for each metric, labeled by source region.
    """
    mapping = {1: 'GPi', 2: 'SNr', 3: 'GPe'}
    metrics = ['Cell', 'ProjDensity', 'RelDensity', 'Voxels', 'Intensity']
    
    final_summary = None
    
    for ch_num, source_name in mapping.items():
        file_path = os.path.join(input_dir, f"Channel{ch_num}_StrictRaw.csv")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping {source_name}.")
            continue
            
        df = pd.read_csv(file_path)
        
        # Initialize summary with primary keys on first loop
        if final_summary is None:
            final_summary = df[['Region_ID', 'Acronym', 'Name']].copy()
            
        for metric in metrics:
            # Find all columns for this metric across all mice (Left and Right)
            # Example: M1_Cell_Left, M1_Cell_Right ... M7_Cell_Right
            metric_cols = [c for c in df.columns if f'_{metric}_' in c]
            
            if metric_cols:
                # Calculate the mean across all these columns for each row (region)
                col_name = f"average_{source_name}_{metric}"
                final_summary[col_name] = df[metric_cols].mean(axis=1)
                
    if final_summary is not None:
        # Reorder to ensure primary keys are first
        final_summary.to_csv(output_file, index=False)
        print(f"Successfully created {output_file}")

# ================= EXECUTION =================
# 1. Generate the StrictRaw files (with voxels)
process_brainj_raw_data()

# 2. Generate the Average summary
create_source_region_averages()
def load_csvs_to_dfs(directory):
    dfs = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            dfs.append(df)
    return dfs

def clean_and_merge_region_data(df_list, labels_df):
    """
    Cleans each DataFrame by deleting the initial region acronym and merging 
    with a labels DataFrame to get the correct acronyms.

    It joins the two DataFrames where the 'region_name' in the input DataFrame
    matches the 'name' in the labels DataFrame.

    Parameters:
      df_list (list of pd.DataFrame): A list of DataFrames to process.
      labels_df (pd.DataFrame): The DataFrame from your labels.csv file.

    Returns:
      list of pd.DataFrame: A list of the processed and merged DataFrames.
    """
    cleaned_and_merged_list = []
    for df in df_list:
        # Create a copy to work with
        df_processed = df.copy()

        # 1. Remove the original 'region_acronym' column if it exists
        if 'region_acronym' in df_processed.columns:
            df_processed.drop(columns=['region_acronym'], inplace=True)

        # 2. Perform the left join using the correct columns for the key
        #    left_on -> column from the df_processed DataFrame
        #    right_on -> column from the labels_df DataFrame
        df_merged = pd.merge(
            df_processed, 
            labels_df, 
            left_on='Name', 
            right_on='name', 
            how='left'
        )
        
        # 3. (Optional but recommended) Drop the redundant 'name' column from labels.csv
        if 'name' in df_merged.columns:
            df_merged.drop(columns=['name'], inplace=True)

        cleaned_and_merged_list.append(df_merged)

    return cleaned_and_merged_list

def split_by_mouse(cleaned_df_list):
    """
    Converts a list of cleaned DataFrames into a 2D list of DataFrames.
    
    - Detects 'StrictRaw' wide format (M1_Cell_Left, M1_Cell_Right, etc.).
    - Splits Left and Right data into separate rows, appending '_left' or '_right' to the acronym.
    - Calculates 'mean_intensity_percent' per mouse.
    - Ensures only ONE 'acronym' column exists in the output (prioritizing the merged/last one).

    Returns:
        list of list of pd.DataFrame
    """
    all_split_dfs = []

    for df_orig in cleaned_df_list:
        df = df_orig.copy()
        
        # Normalize columns slightly to ensure we catch 'acronym'
        df.columns = [c if c.startswith('M') else c.lower() for c in df.columns]

        # --- FIX: Deduplicate columns, keeping the LAST one ---
        # This prioritizes the 'acronym' from the labels file (merged as 'right') 
        # and drops the original 'acronym' from the raw file.
        df = df.loc[:, ~df.columns.duplicated(keep='last')]

        # --- 1. Identify Mouse IDs ---
        # Look for columns like 'M1_Cell_Left', 'M2_Intensity_Right'
        mouse_cols = [c for c in df.columns if c.startswith('M') and '_' in c and c[1].isdigit()]
        
        if not mouse_cols:
            all_split_dfs.append([])
            continue

        # Extract IDs (1, 2, 3...)
        mouse_ids = sorted(list(set([c.split('_')[0].replace('M', '') for c in mouse_cols])), key=int)
        
        mouse_dfs = []
        for m in mouse_ids:
            hemisphere_dfs = []
            
            for side in ['Left', 'Right']:
                hemisphere = side.lower() # 'left' or 'right'
                
                # Map CSV Columns -> Standard Internal Names
                col_mapping = {}
                
                # 1. Cells
                if f"M{m}_Cell_{side}" in df.columns:
                    col_mapping[f"M{m}_Cell_{side}"] = 'mouse_cell'
                
                # 2. Projection Density (Occupancy)
                if f"M{m}_ProjDensity_{side}" in df.columns:
                    col_mapping[f"M{m}_ProjDensity_{side}"] = 'projection_density'
                elif f"M{m}_Density_{side}" in df.columns:
                    col_mapping[f"M{m}_Density_{side}"] = 'projection_density'

                # 3. Relative Density
                if f"M{m}_RelDensity_{side}" in df.columns:
                    col_mapping[f"M{m}_RelDensity_{side}"] = 'relative_density'
                
                # 4. Mean Intensity
                if f"M{m}_Intensity_{side}" in df.columns:
                    col_mapping[f"M{m}_Intensity_{side}"] = 'mean_intensity'
                
                if not col_mapping:
                    continue
                
                # Extract ONLY the necessary columns + acronym
                cols_to_select = ['acronym'] + list(col_mapping.keys())
                sub_df = df[cols_to_select].copy()
                
                # Rename to standard names
                sub_df = sub_df.rename(columns=col_mapping)
                
                # Modify the SINGLE acronym column to include hemisphere
                sub_df['acronym'] = sub_df['acronym'].astype(str) + '_' + hemisphere
                
                hemisphere_dfs.append(sub_df)
            
            # Combine Left + Right for this mouse
            if hemisphere_dfs:
                df_mouse = pd.concat(hemisphere_dfs, ignore_index=True)
                
                # Calculate Intensity Percentage
                if 'mean_intensity' in df_mouse.columns:
                    total_int = df_mouse['mean_intensity'].sum()
                    if total_int > 0:
                        df_mouse['mean_intensity_percent'] = (df_mouse['mean_intensity'] / total_int) * 100
                    else:
                        df_mouse['mean_intensity_percent'] = 0.0
                
                # Final Cleanup: Ensure no duplicate columns exist
                df_mouse = df_mouse.loc[:, ~df_mouse.columns.duplicated()]
                
                mouse_dfs.append(df_mouse)
            else:
                mouse_dfs.append(pd.DataFrame(columns=['acronym', 'mouse_cell']))

        all_split_dfs.append(mouse_dfs)

    return all_split_dfs

def dictify_split_data(split_data, type):
    """
    Converts the output of split_by_mouse() into a nested list of dicts.

    Parameters:
        split_data (list of list of pd.DataFrame):
            The 2D list returned by split_by_mouse().
        type (str): Determines which value to extract. Options:
            - "cell": Raw cell counts ('mouse_cell')
            - "projection": Projection Density / Occupancy ('projection_density')
            - "relative": Relative Density ('relative_density')
            - "intensity": Raw Mean Intensity ('mean_intensity')
            - "intensity_percent": Relative Mean Intensity % ('mean_intensity_percent')

    Returns:
        list of list of dict:
            Same structure as split_data, but each DataFrame is replaced by
            a dict mapping region_acronym -> value.
    """
    # Define valid types and their corresponding column names in the DataFrames
    col_map = {
        "cell": "mouse_cell",
        "projection": "projection_density",   # Maps to Occupancy (ProjDensity)
        "relative": "relative_density",       # Maps to Relative Density (RelDensity)
        "intensity": "mean_intensity",        # Maps to Raw Intensity
        "intensity_percent": "mean_intensity_percent" # Maps to Calculated %
    }

    if type not in col_map:
        raise ValueError(f"`type` must be one of: {list(col_map.keys())}")
    
    val_col = col_map[type]
    
    dictified = []
    for mouse_dfs in split_data:
        mouse_dicts = []
        for df in mouse_dfs:
            # Check if the requested column actually exists in the dataframe
            if val_col not in df.columns:
                # Fallback or warning if specific metric is missing for a mouse
                # (Quietly appending empty dict to maintain structure, or print warning)
                # print(f"Warning: Column '{val_col}' missing in one of the dataframes.")
                mouse_dicts.append({})
                continue

            # Build dict: region_acronym -> selected value
            # Drop NaNs to keep the dictionary clean
            clean_df = df.dropna(subset=["acronym", val_col])
            d = clean_df.set_index("acronym")[val_col].to_dict()
            mouse_dicts.append(d)
            
        dictified.append(mouse_dicts)
    
    return dictified

from collections import defaultdict

def average_list_of_dicts(list_of_dicts):
    """
    Averages the values of a list of dictionaries by key.

    Args:
        list_of_dicts: A list of dictionaries, where each dictionary
                       has numerical values for common keys.

    Returns:
        A dictionary containing the average value for each key.
    """
    sums = defaultdict(float)
    counts = defaultdict(int)

    for d in list_of_dicts:
        for key, value in d.items():
            if isinstance(value, (int, float)):  # Only average numerical values
                sums[key] += value
                counts[key] += 1

    averages = {}
    for key in sums:
        if counts[key] > 0:
            averages[key] = sums[key] / counts[key]
        else:
            averages[key] = 0  # Or handle as appropriate for keys with no numerical values

    return averages

    return summary_df, full_dict, thresholded_dict, novel_regions_list, novel_df

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def barplot(df, metric='ProjDensity', top_n_regions=10, color='labels.csv', sem_labels=True, output_path=None):
    """
    Plots the novel_df in descending order and saves the output to the specified path.
    
    Parameters:
    - df: The DataFrame (novel_df) containing results.
    - metric: The metric string ('ProjDensity', 'RelDensity', 'Cell', 'Intensity').
    - top_n_regions: Number of regions to display.
    - color: Path to 'labels.csv' or a specific color string.
    - sem_labels: Boolean to show numerical SEM on bars.
    - output_path: The base filename/path (e.g., 'results/my_plot') without extension.
    """
    val_col = f"average_{metric}"
    mouse_cols = [c for c in df.columns if c.startswith('M') and f"_{metric}" in c]
    
    df_copy = df.copy()
    # Fix: Fill missing Acronyms with Full Name so the plot doesn't crash
    df_copy['Acronym'] = df_copy['Acronym'].fillna(df_copy['Full Name'])
    
    if mouse_cols:
        df_copy['SEM'] = df_copy[mouse_cols].std(axis=1) / np.sqrt(len(mouse_cols))
    else:
        df_copy['SEM'] = 0

    plot_df = df_copy.nlargest(top_n_regions, val_col).sort_values(by=val_col, ascending=False)
    
    # Handle Color Mapping (Supports String Path OR DataFrame)
    labels_df = None
    if isinstance(color, str) and color.endswith('.csv'):
        labels_df = pd.read_csv(color)
    elif isinstance(color, pd.DataFrame):
        labels_df = color

    if labels_df is not None:
        labels_df.columns = [c.lower() for c in labels_df.columns]
        color_map = {
            str(row['acronym']).lower(): f"#{str(row['color_hex_triplet']).strip().replace('#','')}" 
            for _, row in labels_df.iterrows()
        }
        bar_colors = [color_map.get(str(acc).lower(), '#CCCCCC') for acc in plot_df['Acronym']]
    else:
        bar_colors = color

    # Create the Plot
    plt.figure(figsize=(max(12, top_n_regions * 0.45), 9))
    bars = plt.bar(plot_df['Acronym'], plot_df[val_col], yerr=plot_df['SEM'], color=bar_colors, capsize=4)
    
    # Add numerical labels if requested
    if sem_labels:
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            sem_val = plot_df['SEM'].iloc[i]
            plt.text(
                bar.get_x() + bar.get_width()/2, 
                yval + sem_val + (max(plot_df[val_col]) * 0.01),
                f'±{sem_val:.2f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold'
            )

    plt.xticks(rotation=90)
    plt.ylabel(f'{metric} (Mean Sum ± SEM)')
    plt.title(f'Top {top_n_regions} Regions - {metric}')
    plt.tight_layout()

    # SAVE LOGIC
    if output_path:
        # Ensure the directory exists if a folder path is provided
        folder = os.path.dirname(output_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
            
        # Save high-resolution PNG for presentations
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        # Save SVG for vector editing (Adobe Illustrator, etc.)
        plt.savefig(f"{output_path}.svg", bbox_inches='tight')
        print(f"Files saved: {output_path}.png and {output_path}.svg")

    plt.show()
# --- Examples ---
# To see labels:
# barplot(df_novel, metric='RelDensity', top_n_regions=25, sem_labels=True)

# To hide labels:
# barplot(df_novel, metric='RelDensity', top_n_regions=25, sem_labels=False)
import pandas as pd
import numpy as np
import os

def analyze_brain_channel(df_list, labels_df, channel=3, metric='ProjDensity'):
    """
    Consolidated pipeline for BrainJ data analysis.
    Filters: IDs, Acronyms, Keywords, missing labels, missing hex codes, and empty acronyms.
    Bypass: PF (930) and PPN are always preserved.
    """
    # 1. Metric Mapping
    metric_map = {
        'ProjDensity': 'projection',
        'RelDensity': 'relative',
        'Cell': 'cell',
        'Intensity': 'intensity'
    }
    internal_metric_type = metric_map.get(metric, 'projection')
    
    # 2. Clean and Merge
    raw_df = df_list[channel - 1]
    # This merge uses 'left' join with labels.csv
    cleaned_df = clean_and_merge_region_data([raw_df], labels_df)[0]
    id_col = 'Region_ID' if 'Region_ID' in cleaned_df.columns else 'id'

    # 3. Define Global Exclusion Criteria
    known_list = ["STN", "GPi", "SNr", "STR", "VP", "SNc", "VTA"]
    exc_list = ['AIp5', 'COPY', 'CENT2', 'PFL', 'TH', 'SIM', 'ENTl1', 'MEA', 'DEC', 'HY', 
                'AId6a', 'OLF', 'PIR', 'CUL4, 5', 'CENT3', 'PAG', 'PPT', 'CTXsp', 'AId5', 
                'TR', 'ENTl5', 'ENTl2', 'PAA', 'EPd', 'MY', 'OT', 'ANcr1', 'SI', 'P', 'GPe', 
                'COA', 'MB', 'PAL', 'ENTl3', 'EPv', 'PRM', 'COAa', 'GU6b', 'ACB', 'STR', 'BST', "CP", "fiber tracts"]
    exclude_set = {item.lower() for item in (known_list + exc_list)}
    
    excludedIDs = [20,52,56,139,313,342,344,351,354,403,477,549,566,631,639,662,698,
                   703,771,754,783,788,795,803,936,952,961,966,976,984,1007,1022,
                   1025,1033,1041,1056,1061,1091,1097,1101,1121]

    restricted_keywords = ["capsule", "tract", "bundle", "nerve", "arbor vitae", "fundus of striatum", "anterior forceps"]

    # 4. Apply Global Filtering
    def global_filter(row):
        reg_id = row[id_col]
        raw_acronym = row['acronym'] # This is our Primary Key
        hex_code = row.get('color_hex_triplet') 
        
        # PRIMARY KEY VALIDATION: Exclude if acronym is missing, NaN, or empty
        if pd.isna(raw_acronym) or str(raw_acronym).strip() == "":
            return False
            
        # EXCLUDE: If Region ID is missing or hex code is empty
        if pd.isna(reg_id) or pd.isna(hex_code) or str(hex_code).strip() == "":
            return False
            
        acronym_lower = str(raw_acronym).lower()
        name = str(row.get('Name', row.get('safe_name', ''))).lower()
        
        # BYPASS: Never exclude PF (930) or PPN
        if reg_id == 930 or acronym_lower == 'ppn':
            return True
        
        if reg_id in excludedIDs: return False
        if any(acronym_lower.startswith(base) for base in exclude_set): return False
        if any(kw in name for kw in restricted_keywords): return False
        return True

    # Filter the DataFrame immediately
    cleaned_df = cleaned_df[cleaned_df.apply(global_filter, axis=1)].copy()

    # 5. Create Summary DataFrame (Sums and Averages)
    mouse_cols = [c for c in cleaned_df.columns if c.startswith('M') and f"_{metric}_" in c]
    mouse_ids = sorted(list(set([c.split('_')[0] for c in mouse_cols])), key=lambda x: int(x[1:]))
    
    summary_data = []
    for _, row in cleaned_df.iterrows():
        entry = {
            'Region_ID': int(row.get(id_col)),
            'Acronym': row.get('acronym'),
            'Full Name': row.get('Name', row.get('safe_name', 'Unknown'))
        }
        sums = []
        for m in mouse_ids:
            val_l, val_r = row.get(f"{m}_{metric}_Left", 0), row.get(f"{m}_{metric}_Right", 0)
            m_sum = val_l + val_r
            entry[f"{m}_{metric}"] = m_sum
            sums.append(m_sum)
        entry[f"average_{metric}"] = np.mean(sums) if sums else 0
        summary_data.append(entry)
        
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"summary_channel_{channel}_{metric}.csv", index=False)

    # 6. Generate Hemispheric Dictionaries (Cleaned)
    split_data = split_by_mouse([cleaned_df])
    dict_results = dictify_split_data(split_data, type=internal_metric_type)
    dict_list = dict_results[0] 
    full_dict = average_list_of_dicts(dict_list)

    # 7. Thresholding and Novelty Filtering
    ppn_threshold = full_dict.get("PPN_right", 0)
    thresholded_dict = {}
    novel_regions_list = []
    
    for key, val in full_dict.items():
        if key.lower() == 'ppn_right':
             thresholded_dict[key] = val
             novel_regions_list.append(key)
             continue
        
        if "_right" in key.lower() and val > ppn_threshold:
            print(f"Novel Region Detected: {key} with value {val} (PPN threshold: {ppn_threshold})")
            thresholded_dict[key] = val
            novel_regions_list.append(key)
        else:
            thresholded_dict[key] = 0

    # 8. Final Novel CSV
    novel_base_acronyms = {k.replace('_right', '') for k in novel_regions_list}
    novel_df = summary_df[summary_df['Acronym'].isin(novel_base_acronyms)]
    novel_df.to_csv(f"novel_regions_channel_{channel}_{metric}.csv", index=False)

    return summary_df, full_dict, thresholded_dict, novel_regions_list, novel_df

import pandas as pd

def region_condensation(input_dict, max_depth, atlas): 
    """
    Condenses brain regions based on hierarchy tree from BrainGlobeAtlas.
    Expects a pre-loaded BrainGlobeAtlas object passed into the 'atlas' parameter.
    """
    
    # We no longer load the atlas here! It uses the one you pass in.
    lookup_df = atlas.lookup_df
    
    # 1. Create a quick mapping from ID to acronym 
    id_to_acronym = {
        struct_id: data['acronym'].lower() 
        for struct_id, data in atlas.structures.items()
    }
    
    # 2. Build a clean parent map using the built-in atlas lineage
    parent_map = {}
    
    for struct_id, data in atlas.structures.items():
        acronym = data['acronym'].lower()
        path = data.get('structure_id_path', [])
        
        # If no path exists, map to itself
        if not path:
            parent_map[acronym] = acronym
            continue
            
        # 3. Pick the ID at the target max_depth
        # path[0] is root, path[1] is level 1, path[2] is level 2, etc.
        target_index = max_depth
        
        # If the region is shallower than max_depth, cap it at its own deepest level
        if target_index >= len(path):
            target_index = len(path) - 1
            
        target_id = path[target_index]
        
        # Map the current region to its max_depth parent
        parent_map[acronym] = id_to_acronym.get(target_id, acronym)
            
    # Now condense the input dictionary
    condensed_dict = {}
    
    # FIX: Changed 'values.items()' to 'input_dict.items()'
    for key, value in input_dict.items():
        # Parse the key to extract acronym and hemisphere
        if '_left' in key.lower():
            acronym = key.replace('_left', '').replace('_Left', '').lower()
            hemisphere = '_left'
        elif '_right' in key.lower():
            acronym = key.replace('_right', '').replace('_Right', '').lower()
            hemisphere = '_right'
        else:
            acronym = key.lower()
            hemisphere = ''
        
        # Get parent from map
        parent_acr = parent_map.get(acronym, acronym)
        parent_key = f"{parent_acr}{hemisphere}"
        
        # Sum values
        if parent_key in condensed_dict:
            condensed_dict[parent_key] += value
        else:
            condensed_dict[parent_key] = value
    
    # Create summary DataFrame
    unique_acronyms = set()
    for key in condensed_dict.keys():
        acr = key.replace('_left', '').replace('_right', '')
        unique_acronyms.add(acr)
    
    summary_data = []
    for acronym in sorted(unique_acronyms):
        try:
            lookup_row = lookup_df[lookup_df['acronym'].str.lower() == acronym].iloc[0]
            region_id = lookup_row['id']
            full_name = lookup_row.get('name', 'Unknown')
        except:
            region_id = -1
            full_name = 'Unknown'
        
        total_value = condensed_dict.get(f"{acronym}_left", 0) + condensed_dict.get(f"{acronym}_right", 0)
        
        summary_data.append({
            'Region_ID': region_id,
            'Acronym': acronym,
            'Full Name': full_name,
            'average_value': total_value
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('average_value', ascending=False).reset_index(drop=True)
    
    # --- Updated Thresholding and Novelty Section ---

    # 1. Identify the baseline (the bouncer)
    ppn_val = condensed_dict.get("ppn_right", condensed_dict.get("PPN_right", 0))

    thresholded_dict = {}
    novel_list = []

    # 2. Loop through the grouped results
    for key, val in condensed_dict.items():
        # Always keep the PPN baseline value intact
        if key.lower() == 'ppn_right':
            thresholded_dict[key] = val
            novel_list.append(key)
            continue
        
        # Check for novelty: Right hemisphere AND brighter than PPN
        if "_right" in key.lower() and val > ppn_val:
            thresholded_dict[key] = val
            novel_list.append(key)
        else:
            # If it doesn't meet the criteria, set its "glow" to 0
            thresholded_dict[key] = 0

    # 3. Create the novel_df based on the updated novel_list
    novel_acronyms = {k.replace('_right', '') for k in novel_list}
    novel_df = summary_df[summary_df['Acronym'].isin(novel_acronyms)]

    return summary_df, condensed_dict, thresholded_dict, novel_list, novel_df

import os
import glob
from PIL import Image

def generate_horizontal_stitched_tiff(metric_dict, final_output_dir):
    """Stitches BrainGlobe heatmaps side-by-side (left-to-right) for widescreen presentations."""
    import os, glob, time
    from PIL import Image

    os.makedirs(final_output_dir, exist_ok=True)

    for metric_name, param_list in metric_dict.items():
        print(f"Processing {metric_name} (Horizontal Layout)...")
        generated_image_paths = []

        for params in param_list:
            out_dir = os.path.join(params.get('output_dir'), 'heatmap')

            # Snapshot files that already exist BEFORE generating
            existing_files = set(glob.glob(os.path.join(out_dir, '*')))

            # Generate the heatmap
            bg_heatmap_slices(**params)

            # Small sleep to ensure file system has flushed writes
            time.sleep(0.5)

            # Find all image files after generation
            extensions = ['*.png', '*.PNG', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
            all_files = []
            for ext in extensions:
                all_files.extend(glob.glob(os.path.join(out_dir, ext)))

            # Prefer NEW files (not in the pre-generation snapshot)
            new_files = [f for f in all_files if f not in existing_files]
            candidate_files = new_files if new_files else all_files

            if not candidate_files:
                print(f"[ERROR] No images found in {out_dir}")
                return

            # Pick the most recently MODIFIED file (mtime is reliable post-write)
            best_file = max(candidate_files, key=os.path.getmtime)
            print(f"  -> Selected: {os.path.basename(best_file)}")
            generated_image_paths.append(best_file)

        if len(generated_image_paths) != len(param_list):
            print(f"[ERROR] Expected {len(param_list)} images, got {len(generated_image_paths)}")
            return

        # Open and verify images
        images = []
        for path in generated_image_paths:
            img = Image.open(path).convert('RGB')  # force consistent mode
            images.append(img)
            print(f"  Loaded: {os.path.basename(path)} — size {img.size}")

        # Calculate canvas size for HORIZONTAL stitching
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        horizontal_canvas = Image.new('RGB', (total_width, max_height), (255, 255, 255))

        # Paste images left-to-right
        x_offset = 0
        for img in images:
            horizontal_canvas.paste(img, (x_offset, 0))
            x_offset += img.width

        # Save final image
        tiff_path = os.path.join(final_output_dir, f"{metric_name.replace(' ', '_')}_horizontal.tiff")
        horizontal_canvas.save(tiff_path, format='TIFF')
        print(f"Done! Saved horizontal TIFF at: {tiff_path}\n")
        
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, to_rgb, LinearSegmentedColormap
import importlib

# BrainGlobe imports
from brainglobe_atlasapi import BrainGlobeAtlas
import brainglobe_heatmap as bgh
from brainrender.atlas import Atlas 

# =================================================================
# FAIL-SAFE & PATCHING
# =================================================================
def find_plane_class():
    search_paths = ["brainrender.actors.plane", "brainrender.actors", "brainrender.actor"]
    for name in search_paths:
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "Plane"): return getattr(mod, "Plane")
        except: continue
    class MinimalPlane:
        def __init__(self, pos, normal, **kwargs):
            self.center, self.normal = pos, normal
    return MinimalPlane

PlaneClass = find_plane_class()

def fixed_get_plane(self, pos=None, norm=None, **kwargs):
    try:
        atlas_obj = self.atlas
    except AttributeError:
        atlas_obj = self
        
    shape = atlas_obj.reference.shape
    res = atlas_obj.resolution
    
    if pos is None:
        pos = [s * r / 2 for s, r in zip(shape, res)]
    
    if norm is None: norm = [0, 0, 1]
    
    full_dims = [float(s * r) for s, r in zip(shape, res)]
    idx_pair = [i for i in range(3) if norm[i] == 0]
    if len(idx_pair) < 2: idx_pair = [0, 1]
    
    sx = kwargs.get('sx') or full_dims[idx_pair[0]]
    sy = kwargs.get('sy') or full_dims[idx_pair[1]]
    
    try: return PlaneClass(pos=pos, normal=norm, sx=sx, sy=sy, **kwargs)
    except: return PlaneClass(pos=pos, normal=norm, s=(sx, sy), **kwargs)

Atlas.get_plane = fixed_get_plane

# =================================================================
# HEATMAP GENERATION
# =================================================================

import cv2
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, to_rgb, LinearSegmentedColormap
import importlib

# BrainGlobe imports
from brainglobe_atlasapi import BrainGlobeAtlas
import brainglobe_heatmap as bgh
from brainrender.atlas import Atlas 

# =================================================================
# FAIL-SAFE & PATCHING (BrainRender Fixes)
# =================================================================
def find_plane_class():
    search_paths = ["brainrender.actors.plane", "brainrender.actors", "brainrender.actor"]
    for name in search_paths:
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "Plane"): return getattr(mod, "Plane")
        except: continue
    class MinimalPlane:
        def __init__(self, pos, normal, **kwargs):
            self.center, self.normal = pos, normal
    return MinimalPlane

PlaneClass = find_plane_class()

def fixed_get_plane(self, pos=None, norm=None, **kwargs):
    try:
        atlas_obj = self.atlas
    except AttributeError:
        atlas_obj = self
    shape = atlas_obj.reference.shape
    res = atlas_obj.resolution
    if pos is None:
        pos = [s * r / 2 for s, r in zip(shape, res)]
    if norm is None: norm = [0, 0, 1]
    full_dims = [float(s * r) for s, r in zip(shape, res)]
    idx_pair = [i for i in range(3) if norm[i] == 0]
    if len(idx_pair) < 2: idx_pair = [0, 1]
    sx = kwargs.get('sx') or full_dims[idx_pair[0]]
    sy = kwargs.get('sy') or full_dims[idx_pair[1]]
    try: return PlaneClass(pos=pos, normal=norm, sx=sx, sy=sy, **kwargs)
    except: return PlaneClass(pos=pos, normal=norm, s=(sx, sy), **kwargs)

Atlas.get_plane = fixed_get_plane

# =================================================================
# HEATMAP GENERATION
# =================================================================

def bg_heatmap_slices(
    values, output_dir, view, vmin, vmax, cmap='Reds',
    atlas_name='allen_mouse_25um', annotate_regions=False, 
    cbar_label='Value', hor_swap=False, ver_swap=False, 
    labels_csv=None, specific_slice=None, frame_rate=10 
):
    """
    Optimized brain heatmap generator. Avoids GEOSException by pre-filtering 
    problematic regions and uses a safe rendering wrapper.
    """
    use_custom_gradients = (cmap is None)
    base_cmap = cmap if cmap is not None else 'Reds'
    display_cmap = 'Greys' if use_custom_gradients else base_cmap
    
    heatmap_dir = os.path.join(output_dir, "heatmap")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Load Atlas once
    atlas_bg = BrainGlobeAtlas(atlas_name)
    res = atlas_bg.resolution
    shape = atlas_bg.reference.shape 
    n_slices = shape[{"frontal": 0, "horizontal": 1, "sagittal": 2}[view]]

    # --- 1. Save Standalone Scalebars ---
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=display_cmap, norm=norm)

    for orient, fname in [('vertical', 'scalebar_vertical.svg'), ('horizontal', 'scalebar_horizontal.svg')]:
        fig_sb = plt.figure(figsize=(10, 10))
        ax_sb = fig_sb.add_axes([0.1, 0.1, 0.8, 0.8])
        cb = fig_sb.colorbar(sm, cax=ax_sb, orientation=orient)
        cb.set_label(cbar_label)
        fig_sb.savefig(os.path.join(heatmap_dir, fname), format='svg', bbox_inches='tight')
        plt.close(fig_sb)

    # 2. Load Colors for Custom Gradients
    acronym_to_color = {}
    if use_custom_gradients and labels_csv and os.path.exists(labels_csv):
        ldf = pd.read_csv(labels_csv)
        for _, row in ldf.iterrows():
            acronym_to_color[str(row['acronym']).lower()] = str(row['color_hex_triplet']).strip()

    # 3. FAST Filtering (Avoids get_structure_mask which slows down network drives)
    valid_acronyms = set(atlas_bg.lookup_df.acronym.values)
    # Manual blacklist for regions known to have geometry issues in the 25um atlas
    blacklist = {'rspd4', 'rspd2'} 

    def clean_dict(input_dict, suffix):
        cleaned = {}
        for k, v in input_dict.items():
            if k.endswith(suffix) and not pd.isna(v):
                acronym = k.removesuffix(suffix)
                # Fast check: exists in ontology and not in our known-error list
                if acronym in valid_acronyms and acronym.lower() not in blacklist:
                    cleaned[acronym] = float(v)
        return cleaned

    left_dict = clean_dict(values, '_left')
    right_dict = clean_dict(values, '_right')

    def render_slice(idx):
        pos_um = idx * res[{"frontal": 0, "horizontal": 1, "sagittal": 2}[view]]
        fig, ax = plt.subplots(figsize=(12, 10))

        for d, h in [(left_dict, 'left'), (right_dict, 'right')]:
            if not d: continue
            
            try:
                # Use a try-except loop inside to prevent one bad region from killing the slice
                hm = bgh.Heatmap(d, position=pos_um, orientation=view, hemisphere=h,
                                 atlas_name=atlas_name, cmap=base_cmap, vmin=vmin, vmax=vmax, 
                                 annotate_regions=False)
                
                if use_custom_gradients:
                    for acronym, val in d.items():
                        ckey = acronym.lower()
                        if ckey in acronym_to_color:
                            hex_c = acronym_to_color[ckey]
                            base_rgb = to_rgb(hex_c if hex_c.startswith("#") else f"#{hex_c}")
                            region_cm = LinearSegmentedColormap.from_list("custom", [(1, 1, 1), base_rgb])
                            hm.colors[acronym] = list(region_cm(np.clip((val - vmin) / (vmax - vmin), 0, 1))[:3])
                
                hm.plot_subplot(fig=fig, ax=ax, show_cbar=False)
            
            except Exception as e:
                # Catch-all for GEOS geometry errors
                print(f"[RECOVERED] Slice {idx} {h}-hemisphere error: {e}")
                continue

        ax.autoscale(True) 
        if ver_swap: ax.invert_xaxis() 
        if hor_swap: ax.set_ylim(ax.get_ylim()[::-1])
        ax.axis('off')
        plt.tight_layout()
        
        # Internal frame colorbar
        cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5]) 
        fig.colorbar(sm, cax=cbar_ax, orientation='vertical').set_label(cbar_label)
        return fig

    # 4. Processing Run
    indices = [specific_slice] if specific_slice is not None else range(0, n_slices, 1)
    frame_paths = []

    for i in indices:
        print(f"[PROCESS] Slice {i}/{n_slices}...")
        fig = render_slice(i)
        save_path = os.path.join(heatmap_dir, f"{view}_{i:04d}.tif")
        fig.savefig(save_path, dpi=100, bbox_inches='tight') # Reduced DPI for speed
        frame_paths.append(save_path)
        plt.close(fig)

    # 5. Video Generation
    if len(frame_paths) > 1:
        print(f"[VIDEO] Stitching {len(frame_paths)} frames...")
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is not None:
            video_name = os.path.join(output_dir, f"heatmap_{view}.mp4")
            video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (first_frame.shape[1], first_frame.shape[0]))
            for path in frame_paths:
                video.write(cv2.imread(path))
            video.release()

    print(f"\n[SUCCESS] Completed. Files in: {heatmap_dir}")