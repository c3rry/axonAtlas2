import os
import re
import shutil
import tifffile as tf
from pathlib import Path

def consolidate_channels(root_dir):
    root = Path(root_dir)
    # Find all sample folders (case-insensitive "sample")
    sample_folders = [f for f in root.iterdir() if f.is_dir() and f.name.lower().startswith('sample')]

    for sample_path in sample_folders:
        sample_name = sample_path.name.replace(" ", "_") # e.g., "sample_1"
        channels_root = sample_path / "All_Channels"
        
        if not channels_root.exists():
            print(f"Skipping {sample_path}: 'All_Channels' not found.")
            continue

        print(f"\n--- Processing {sample_name} ---")

        # ==========================================
        # PHASE 1: Organize loose files into folders
        # ==========================================
        loose_tifs = list(channels_root.glob("*.tif"))
        if loose_tifs:
            print(f"Found {len(loose_tifs)} loose .tif files. Sorting into channel folders...")
            for tif_path in loose_tifs:
                # Look for 'ch' followed by numbers at the end of the filename (e.g., _ch0.tif)
                match = re.search(r'(ch\d+)\.tif$', tif_path.name.lower())
                
                if match:
                    ch_name = match.group(1) # Extracts exactly 'ch0', 'ch1', etc.
                    
                    # Create the channel directory (e.g., All_Channels/ch0)
                    ch_dir = channels_root / ch_name
                    ch_dir.mkdir(exist_ok=True)
                    
                    # Move the file into the new directory
                    target_path = ch_dir / tif_path.name
                    shutil.move(str(tif_path), str(target_path))
        else:
            print("No loose .tif files found to sort (they might already be in folders).")


        # ==========================================
        # PHASE 2: Run your original stacking logic
        # ==========================================
        # Find all channel subdirectories (ch0, ch1, etc.)
        channel_dirs = [d for d in channels_root.iterdir() if d.is_dir() and d.name.startswith('ch')]
        
        if not channel_dirs:
             print(f"No channel directories to process in {channels_root}.")
             continue

        for ch_dir in channel_dirs:
            ch_name = ch_dir.name # e.g., "ch0"
            output_filename = f"{sample_name}_{ch_name}.tif"
            output_path = sample_path / output_filename
            
            # Get all .tif files and sort them numerically/alphabetically
            tif_files = sorted(list(ch_dir.glob("*.tif")))
            
            if not tif_files:
                print(f"No .tif files found in {ch_dir}")
                continue

            print(f"Stacking {ch_name} ({len(tif_files)} planes) -> {output_filename}...")
            
            # Read and write as a multipage TIFF
            with tf.TiffWriter(output_path, bigtiff=True) as stack:
                for filename in tif_files:
                    data = tf.imread(filename)
                    stack.write(data, contiguous=True)
            
            print(f"Saved: {output_path}")

# Run the function
# consolidate_channels(r"Z:\anatomy_histology\UTSW_cells")