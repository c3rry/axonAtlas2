import pandas as pd
import plotly.graph_objects as go
import os

def hex_to_rgba(hex_code, alpha=0.4):
    """Converts hex strings to rgba for Plotly link transparency."""
    hex_code = str(hex_code).lstrip('#')
    if len(hex_code) != 6:
        return f"rgba(128, 128, 128, {alpha})"
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

def create_basal_ganglia_sankey(averages_file, labels_file, metric='RelDensity'):
    """
    Creates a Sankey diagram with accurate colors from labels.csv.
    
    Parameters:
    - averages_file: Path to 'Source_Region_Averages.csv'
    - labels_file: Path to 'labels.csv'
    - metric: 'RelDensity', 'ProjDensity', 'Cell', or 'Intensity'
    """
    # 1. Load Data
    if not os.path.exists(averages_file) or not os.path.exists(labels_file):
        print("Error: One or both input files not found.")
        return None
        
    df = pd.read_csv(averages_file)
    labels_df = pd.read_csv(labels_file)

    # 2. Build Color Mapping from labels.csv
    # We map both 'name' and 'acronym' to the hex triplet
    color_map = {}
    for _, row in labels_df.iterrows():
        hex_val = str(row['color_hex_triplet']).zfill(6)
        color_map[row['name']] = f"#{hex_val}"
        color_map[row['acronym']] = f"#{hex_val}"

    # 3. Setup Nodes
    sources = ["GPe", "GPi", "SNr"]
    target_names = df['Name'].unique().tolist()
    all_nodes = sources + target_names
    node_indices = {name: i for i, name in enumerate(all_nodes)}
    
    # Assign colors to nodes based on mapping
    node_colors = []
    for node in all_nodes:
        # Default to a neutral grey if region not found in labels.csv
        node_colors.append(color_map.get(node, "#808080"))

    # 4. Build Links
    source_indices = []
    target_indices = []
    values = []
    link_colors = []

    for src in sources:
        val_col = f'average_{src}_{metric}'
        vox_col = f'average_{src}_Voxels'
        
        if val_col in df.columns:
            # Filter rows with activity
            mask = df[val_col] > 0
            temp_df = df[mask].copy()
            
            # Normalization: Metric / Voxels
            if vox_col in df.columns:
                temp_df['plot_val'] = temp_df[val_col] / temp_df[vox_col].replace(0, 1)
            else:
                temp_df['plot_val'] = temp_df[val_col]
            
            # Get transparent version of the source color for the links
            src_hex = color_map.get(src, "808080")
            link_color_rgba = hex_to_rgba(src_hex, alpha=0.3)

            for _, row in temp_df.iterrows():
                source_indices.append(node_indices[src])
                target_indices.append(node_indices[row['Name']])
                values.append(row['plot_val'])
                link_colors.append(link_color_rgba)

    if not values:
        print(f"No non-zero data found for metric '{metric}'.")
        return None

    # 5. Create Plotly Figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors,
            # Tooltip shows the raw normalized value
            hovertemplate='Source: %{source.label}<br />'+
                          'Target: %{target.label}<br />'+
                          'Value: %{value:.4f}<extra></extra>'
        )
    )])

    fig.update_layout(
        title_text=f"Basal Ganglia Output Connectivity: {metric} (Normalized by Source Volume)", 
        font_size=10,
        height=max(800, len(target_names) * 15), # Scale height by number of targets
        template="plotly_white"
    )
    
    return fig