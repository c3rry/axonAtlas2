import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def cluster_with_voting_system(csv_path, swc_folder, output_root, k_range=(2, 12)):
    # --- Step 0: Clear and Prepare Output Directory ---
    print(f"--- Step 0: Preparing Output Directory ---")
    if os.path.exists(output_root):
        print(f"Clearing existing folder: {output_root}")
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    # --- Step 1: Loading & Cleaning Data ---
    print(f"--- Step 1: Loading & Cleaning Data ---")
    df = pd.read_csv(csv_path)
    
    # Separate Numeric and Categorical data
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Identify XYZ columns
    xyz_cols = [c for c in numeric_df.columns if c.lower() in ['x', 'y', 'z', 'soma_x', 'soma_y', 'soma_z']]
    
    # Identify Structure columns (one-hot encoded 0s and 1s)
    struct_cols = [c for c in numeric_df.columns if c.startswith('structure_') 
                   and c not in ['cluster', 'soma_id']]

    print(f"Features: {len(xyz_cols)} spatial dimensions, {len(struct_cols)} anatomical flags.")

    # Hybrid Scaling Logic
    scaler = StandardScaler()
    xyz_scaled = scaler.fit_transform(df[xyz_cols])
    
    # Combine Scaled XYZ + Original Binary Structures
    X_final = np.hstack((xyz_scaled, df[struct_cols].values))
    print(f"Final matrix shape for clustering: {X_final.shape}")

    # --- Step 2: Evaluating Optimal k ---
    print(f"--- Step 2: Evaluating Optimal k ---")
    ks = range(k_range[0], k_range[1] + 1)
    sil_scores = []
    wcss = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_final)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X_final, labels))
        print(f"   k={k} | Silhouette: {sil_scores[-1]:.4f}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(ks, wcss, 'bo-', linewidth=2)
    ax1.set_title('Elbow Plot (Inertia)')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.grid(True)

    ax2.plot(ks, sil_scores, 'ro-', linewidth=2)
    ax2.set_title('Silhouette Score (Higher is Better)')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Score')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_root, "cluster_optimization_report.png"))
    plt.show()

    # --- Step 3: Finalizing Clusters ---
    opt_k = ks[np.argmax(sil_scores)]
    print(f"--- Step 3: Finalizing Clusters (Optimal k = {opt_k}) ---")
    final_km = KMeans(n_clusters=opt_k, random_state=42, n_init=10)
    df['cluster'] = final_km.fit_predict(X_final)

    # --- Step 4: Organizing Files ---
    print(f"--- Step 4: Creating {opt_k} Folders & Moving SWC Files ---")
    for cluster_id in range(opt_k):
        cluster_subset = df[df['cluster'] == cluster_id]
        
        # Region Voting: find which one-hot column has the most '1's in this cluster
        if len(struct_cols) > 0:
            votes = cluster_subset[struct_cols].sum()
            winner_col = votes.idxmax()
            region_name = winner_col.replace('structure_', '')
        else:
            region_name = "UnknownRegion"

        # Create Folder for this specific cluster
        folder_name = f"{region_name}_Cluster{cluster_id}"
        cluster_dir = os.path.join(output_root, folder_name)
        os.makedirs(cluster_dir, exist_ok=True)
        
        print(f"Populating Cluster {cluster_id} -> '{folder_name}' ({len(cluster_subset)} neurons)")

        # Copy SWC files into their respective cluster folder
        for _, row in cluster_subset.iterrows():
            fname = str(row['swc_name'])
            if not fname.endswith('.swc'): 
                fname += '.swc'
            
            src = os.path.join(swc_folder, fname)
            dst = os.path.join(cluster_dir, fname)
            
            if os.path.exists(src):
                try:
                    shutil.copy(src, dst)
                except Exception as e:
                    print(f"Skipping {fname}: {e}")
                    continue
            else:
                print(f"Warning: SWC file not found at {src}")

    print(f"\nDone! Fresh results saved in: {output_root}")
    return df



import vedo
import os
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from brainrender import Scene
from brainrender.actors import Neuron

# Force 'qt' backend for the pop-up window
vedo.settings.default_backend = 'qt' 

def get_distinct_colormaps(n_clusters):
    """
    Selects unique colormaps from 'Reds', 'Greens', 'Blues' families.
    Avoids light colors near 0.0 for better visibility in brainrender.
    """
    # Define a library of distinct maps within the requested families
    cmap_pool = [
        plt.cm.Reds, plt.cm.Greens, plt.cm.Blues,
        plt.cm.YlOrRd, plt.cm.YlGn, plt.cm.GnBu,
        plt.cm.Purples, plt.cm.PuRd, plt.cm.BuPu
    ]
    
    # Shuffle or select maps, ensuring we have enough for n_clusters
    selected_cmaps = []
    for i in range(n_clusters):
        # Sample the pool, wrapping around if we run out of unique maps
        map_func = cmap_pool[i % len(cmap_pool)]
        selected_cmaps.append(map_func)
        
    return selected_cmaps

def render_clustered_neurons_by_map(clustered_root_dir):
    """
    Renders SWCs where each folder gets a unique colormap.
    All neurons within a folder have a unique color within that map.
    """
    scene = Scene(title="Cluster Colormap Visualization")
    
    # 1. Gather real clusters (exclude noise folder if it exists)
    all_folders = [f for f in os.listdir(clustered_root_dir) 
                   if os.path.isdir(os.path.join(clustered_root_dir, f))]
    
    real_cluster_folders = [f for f in all_folders if 'clusterfolder-1' not in f]
    noise_folder = [f for f in all_folders if 'clusterfolder-1' in f]
    
    print(f"Found {len(real_cluster_folders)} dense clusters and {len(noise_folder)} noise folders.")

    # 2. Pre-assign unique colormaps to the dense clusters
    cluster_cmaps = get_distinct_colormaps(len(real_cluster_folders))

    # --- Render Real Clusters ---
    for i, folder_name in enumerate(real_cluster_folders):
        folder_path = os.path.join(clustered_root_dir, folder_name)
        swc_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.swc')]
        
        if not swc_files: continue

        # The colormap for THIS specific folder
        cmap = cluster_cmaps[i]
        print(f"Assigning colormap '{cmap.name}' to {folder_name} ({len(swc_files)} neurons).")
        
        for idx, swc_file in enumerate(swc_files):
            full_path = os.path.join(folder_path, swc_file)
            try:
                # Generate a unique color WITHIN the colormap.
                # We normalize the index (0 to 1) and restrict it from 0.4 to 1.0
                # to avoid light colors that disappear against the brain regions.
                norm_index = 0.4 + (0.6 * (idx / len(swc_files))) if len(swc_files) > 1 else 0.7
                rgba_color = cmap(norm_index)
                hex_color = mcolors.to_hex(rgba_color)
                
                neuron = Neuron(
                    full_path,
                    color=hex_color,
                    alpha=0.6 
                )
                scene.add(neuron)
            except Exception as e:
                print(f"Could not render {swc_file}: {e}")

    # --- Render Noise (If present) ---
    if noise_folder:
        folder_path = os.path.join(clustered_root_dir, noise_folder[0])
        noise_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.swc')]
        if noise_files:
            print(f"Rendering {len(noise_files)} noise neurons in neutral light grey.")
            for swc_file in noise_files:
                try:
                    neuron = Neuron(os.path.join(folder_path, swc_file), color='#d3d3d3', alpha=0.15)
                    scene.add(neuron)
                except: continue

    # Add relevant brain regions (using grey tones so cluster colors pop)
    scene.add_brain_region('SNr', alpha=0.1, color='grey')
    scene.add_brain_region('GPe', alpha=0.1, color='lightgrey')
    
    print("Rendering scene... check for the QT pop-up window.")
    scene.render()