import os
import sys
import re
import subprocess
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.registration import phase_cross_correlation
from skimage.feature import match_template, ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import rescale, EuclideanTransform, warp
from skimage.filters import gaussian
from scipy.stats import mode
from scipy.optimize import least_squares

# =========================================================
# 1. Preprocessing Tools
# =========================================================

def preprocessing(input_dir, output_dir):
    """
    Concatenates TIFF stacks with matching base names and VXHY tags, 
    ordered by the (n) suffix.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dictionary to hold lists of files: { 'base_name': [(index, filepath), ...] }
    groups = {}

    # Regex to capture:
    # Group 1: The unique base name (everything up to the parenthesis)
    # Group 2: The order index inside the parenthesis
    # Matches format: ..._V#H#(n).tif
    pattern = re.compile(r"(.+_V\d+H\d+)\((\d+)\)\.tif$")

    print(f"Scanning {input_path}...")

    for file_path in input_path.glob("*.tif"):
        match = pattern.search(file_path.name)
        if match:
            base_name = match.group(1)
            index = int(match.group(2))
            
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append((index, file_path))

    # Process each group
    for base_name, file_list in groups.items():
        # Sort by the index (n)
        file_list.sort(key=lambda x: x[0])
        
        ordered_files = [f[1] for f in file_list]
        print(f"Processing group: {base_name} ({len(ordered_files)} parts)")

        try:
            # Read all stacks in the group
            # Note: If files are massive, we might need a memory-mapped approach
            stack_parts = [tifffile.imread(f) for f in ordered_files]
            
            # Concatenate along the first axis (Z-axis for typical ZYX stacks)
            full_stack = np.concatenate(stack_parts, axis=0)
            
            # Construct output filename
            output_filename = output_path / f"{base_name}_concat.tif"
            
            # Save the result
            tifffile.imwrite(output_filename, full_stack)
            print(f"  -> Saved to {output_filename}")
            
        except Exception as e:
            print(f"  -> Error processing {base_name}: {e}")

    print("\nProcessing complete.")


def fix_filenames(input_dir):
    """
    Renames microscopy files to a simple linear sequence (tile_000, tile_001...)
    to prevent BigStitcher Manual Loader errors.
    """
    prefix = "2x_532nm_200ms_3.25x3.25x3.25um_"
    suffix = "_concat.tif"
    
    if not os.path.exists(input_dir):
        print(f"Error: Directory not found: {input_dir}")
        return False, 0

    print(f"Scanning: {input_dir} ...\n")
    
    # 1. Collect all valid files and their V, H coordinates
    files_to_rename = []
    for filename in os.listdir(input_dir):
        if not filename.endswith(".tif"):
            continue
            
        # Look for V/H pattern
        match = re.search(r"V(\d+)H(\d+)", filename)
        if match:
            v_val = int(match.group(1))
            h_val = int(match.group(2))
            files_to_rename.append({
                'original': filename,
                'v': v_val,
                'h': h_val
            })
    
    if not files_to_rename:
        print("No matching V...H... files found to rename.")
        return False, 0

    # 2. Sort by V then H to maintain grid order
    # This ensures V1H1 is tile 0, V1H2 is tile 1, etc.
    files_to_rename.sort(key=lambda x: (x['v'], x['h']))
    
    # 3. Rename to linear sequence
    count = 0
    for i, file_info in enumerate(files_to_rename):
        # Create new name: ..._tile_000.tif
        new_name = f"{prefix}tile_{i:03d}{suffix}"
        
        old_path = os.path.join(input_dir, file_info['original'])
        new_path = os.path.join(input_dir, new_name)
        
        if file_info['original'] != new_name:
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {file_info['original']} \n     ->  {new_name}")
                count += 1
            except OSError as e:
                print(f"Error renaming {file_info['original']}: {e}")
        else:
            print(f"Checked: {file_info['original']} (Already correct)")
            
    total_files = len(files_to_rename)
    print(f"\nSuccess! Processed {total_files} files.")
    return True, total_files

# =========================================================
# 2. Stitching Core Logic
# =========================================================

def _scan_directory(input_dir):
    """Internal helper to find tiles and parse V/H coordinates."""
    tiles = []
    pattern = re.compile(r"V(\d+)H(\d+)")
    
    if not os.path.exists(input_dir):
        print(f"Error: Directory {input_dir} not found.")
        return []

    files = sorted([f for f in os.listdir(input_dir) if f.endswith((".tif", ".tiff"))])
    for fname in files:
        match = pattern.search(fname)
        if match:
            v_idx = int(match.group(1))
            h_idx = int(match.group(2))
            path = os.path.join(input_dir, fname)
            tiles.append({'v': v_idx, 'h': h_idx, 'path': path, 'filename': fname})
            
    return tiles

def compute_pseudo_flat_field(tiles, mid_z, sample_size=20):
    """
    Estimates illumination pattern. Robust to varying tile sizes.
    """
    print("  > Computing Pseudo Flat-Field (removing vignetting)...")
    
    shapes = []
    sample_subset = tiles[:min(len(tiles), 50)] 
    for t in sample_subset:
        try:
            with tifffile.TiffFile(t['path']) as tif:
                s = tif.series[0].shape
                if len(s) == 3: shapes.append((s[1], s[2]))
                elif len(s) == 2: shapes.append(s)
        except: pass
    
    if not shapes: return None
    try:
        common_shape = mode(shapes, axis=0).mode[0]
        std_h, std_w = common_shape[0], common_shape[1]
    except:
        std_h, std_w = shapes[0]

    accum = np.zeros((std_h, std_w), dtype=np.float32)
    count = 0
    
    import random
    subset = random.sample(tiles, min(len(tiles), sample_size))
    
    for t in subset:
        try:
            with tifffile.TiffFile(t['path']) as tif:
                if len(tif.series[0].shape) == 3:
                    z_idx = min(mid_z, tif.series[0].shape[0]-1)
                    img = tif.asarray(key=z_idx)
                else:
                    img = tif.asarray()
                
                if img.shape[-2:] != (std_h, std_w): continue
                    
                img = img.astype(np.float32)
                accum += img
                count += 1
        except: continue
            
    if count == 0: return None
    
    flat_field = accum / count
    flat_field = gaussian(flat_field, sigma=30)
    flat_field = flat_field / np.max(flat_field)
    flat_field[flat_field < 0.1] = 0.1
    return flat_field

def midplane(input_dir, rotation_dict=None, overlap=None, vmax=None, cmap='viridis', switch_axes=False):
    """
    Advanced Stitching Preview with Interactive Mode.
    
    Args:
        overlap: 
            - 'interactive': Activates Sliders in Notebook.
            - 'auto' / 'hybrid': Automatic detection.
            - dict {'x':...}: Manual fixed overlap.
    """
    if rotation_dict is None: rotation_dict = {}
    
    tiles = _scan_directory(input_dir)
    if not tiles: return

    min_v = min(t['v'] for t in tiles)
    min_h = min(t['h'] for t in tiles)
    
    grid_rows_max = 0
    grid_cols_max = 0
    max_y, max_x, max_z = 0, 0, 0

    processed_tiles = []
    
    print("Scanning metadata...")
    for i, t in enumerate(tiles):
        v, h = t['v'] - min_v, t['h'] - min_h
        if switch_axes: v, h = h, v
        
        k_rot = rotation_dict.get((t['v'], t['h']), 0)
        
        try:
            with tifffile.TiffFile(t['path']) as tif:
                series = tif.series[0]
                shape = series.shape
                if len(shape) == 2: z, y, x = 1, shape[0], shape[1]
                elif len(shape) == 3: z, y, x = shape
                else: continue
                
                max_z = max(max_z, z)
                if k_rot % 2 != 0: y, x = x, y
                max_y = max(max_y, y)
                max_x = max(max_x, x)
                
                t_data = {
                    'id': i, 'grid_v': v, 'grid_h': h, 
                    'path': t['path'], 'k_rot': k_rot, 
                    'h': y, 'w': x
                }
                processed_tiles.append(t_data)
        except: pass

    mid_z_temp = max_z // 2
    flat_field = compute_pseudo_flat_field(processed_tiles, mid_z_temp)

    def apply_correction(img):
        if flat_field is None: return img
        if img.shape == flat_field.shape:
            return (img / flat_field).astype(img.dtype)
        return img

    # ---------------------------------------------------------
    # INTERACTIVE MODE
    # ---------------------------------------------------------
    if overlap == 'interactive':
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError:
            print("Error: ipywidgets not installed. Run 'pip install ipywidgets'.")
            return

        print("Interactive Mode: Pre-loading data for smooth scrubbing (this takes a moment)...")
        
        # Pre-load cache to RAM for speed
        tile_cache = []
        for t in processed_tiles:
            try:
                vol = tifffile.imread(t['path'])
                if t['k_rot'] > 0: vol = np.rot90(vol, k=t['k_rot'], axes=(1,2)) if vol.ndim==3 else np.rot90(vol, k=t['k_rot'])
                
                # XY Midplane
                if vol.ndim == 3:
                    img_xy = vol[min(mid_z_temp, vol.shape[0]-1)].astype(np.float32)
                else:
                    img_xy = vol.astype(np.float32)
                
                img_xy = apply_correction(img_xy)
                
                t['img_xy'] = img_xy
                t['z_dim'] = vol.shape[0] if vol.ndim == 3 else 1
                tile_cache.append(t)
            except: pass

        rows = grid_rows_max + 1
        cols = grid_cols_max + 1
        
        # Function to render the plot
        def render_interactive(ov_x, ov_y, ov_z, int_max):
            total_y = rows * max_y - (rows - 1) * ov_y
            total_x = cols * max_x - (cols - 1) * ov_x
            
            # Limit canvas size for display performance if needed, but let's try full first
            canvas = np.zeros((total_y, total_x), dtype=np.float32)
            
            for t in tile_cache:
                r, c = t['grid_v'], t['grid_h']
                # Simple linear grid logic
                y_start = r * (max_y - ov_y)
                x_start = c * (max_x - ov_x)
                
                img = t['img_xy']
                h, w = img.shape
                
                # Bounds check
                y_end = min(y_start + h, total_y)
                x_end = min(x_start + w, total_x)
                h_crop = y_end - y_start
                w_crop = x_end - x_start
                
                if h_crop > 0 and w_crop > 0:
                    canvas[y_start:y_end, x_start:x_end] = img[:h_crop, :w_crop]
            
            # Plot
            plt.figure(figsize=(10, 10))
            vis = np.log1p(np.maximum(canvas, 0))
            if int_max > 0: vis = np.clip(vis, 0, np.log1p(int_max))
            
            plt.imshow(vis, cmap=cmap)
            plt.title(f"Interactive Grid: {rows}x{cols}\nOverlap: X={ov_x}, Y={ov_y}")
            plt.axis('off')
            plt.show()

        # Widgets
        style = {'description_width': 'initial'}
        
        # Ranges: Overlap can be negative (gap) to almost full width
        w_ov_x = widgets.IntSlider(min=-500, max=max_x-100, value=0, step=10, description='X Overlap (px)', style=style, continuous_update=False)
        w_ov_y = widgets.IntSlider(min=-500, max=max_y-100, value=0, step=10, description='Y Overlap (px)', style=style, continuous_update=False)
        # Z shift currently doesn't affect XY plane view in simple mode, but we keep for API consistency or expand later
        w_ov_z = widgets.IntSlider(min=-100, max=100, value=0, step=1, description='Z Shift (ignored in 2D)', style=style, disabled=True) 
        w_vmax = widgets.FloatSlider(min=100, max=65535, value=vmax if vmax else 3000, step=100, description='Max Intensity', style=style, continuous_update=False)

        ui = widgets.VBox([
            widgets.HBox([w_ov_x, w_ov_y]),
            widgets.HBox([w_vmax, w_ov_z])
        ])

        out = widgets.interactive_output(render_interactive, {'ov_x': w_ov_x, 'ov_y': w_ov_y, 'ov_z': w_ov_z, 'int_max': w_vmax})
        
        display(ui, out)
        return

    # ---------------------------------------------------------
    # STATIC / AUTO MODES (Existing Logic)
    # ---------------------------------------------------------
    def get_slab(pt, plane='xy'):
        try:
            vol = tifffile.imread(pt['path'])
            if pt['k_rot'] > 0: vol = np.rot90(vol, k=pt['k_rot'], axes=(1,2))
            if plane == 'xy':
                img = vol[min(mid_z_temp, vol.shape[0]-1)]
                return apply_correction(img.astype(np.float32))
            elif plane == 'xz': return np.max(vol, axis=1) 
            elif plane == 'yz': return np.max(vol, axis=2)
        except: return None

    # --- Pairwise Registration ---
    def register_pair_orb(img1, img2, axis):
        descriptor_extractor = ORB(n_keypoints=500)
        descriptor_extractor.detect_and_extract(img1)
        k1 = descriptor_extractor.keypoints
        d1 = descriptor_extractor.descriptors
        descriptor_extractor.detect_and_extract(img2)
        k2 = descriptor_extractor.keypoints
        d2 = descriptor_extractor.descriptors
        if len(k1) < 10 or len(k2) < 10: return None
        matches = match_descriptors(d1, d2, cross_check=True)
        if len(matches) < 5: return None
        src, dst = k1[matches[:, 0]], k2[matches[:, 1]]
        try:
            model_robust, inliers = ransac((src, dst), EuclideanTransform, min_samples=3, residual_threshold=2, max_trials=100)
            if not inliers.any() or np.sum(inliers) < 5: return None
            shift_y, shift_x = model_robust.translation
            return (-shift_y, -shift_x)
        except: return None

    def register_pair_hybrid(img1, img2, axis):
        try:
            # Crop center to avoid edge artifacts
            h, w = img1.shape
            cy, cx = int(h*0.05), int(w*0.05)
            sub1, sub2 = img1[cy:-cy, cx:-cx], img2[cy:-cy, cx:-cx]
            shift, _, _ = phase_cross_correlation(sub1, sub2, upsample_factor=1)
            return shift
        except: return (0, 0)

    # [FIX] Define grid_map before calculating shifts!
    grid_map = { (t['grid_v'], t['grid_h']): t for t in processed_tiles }

    # Collect constraints
    constraints = []
    print("Calculating Pairwise Shifts...")
    use_orb = (overlap == 'features')
    
    for t in processed_tiles:
        v, h = t['grid_v'], t['grid_h']
        # Right Neighbor
        if (v, h+1) in grid_map:
            n = grid_map[(v, h+1)]
            img1, img2 = get_slab(t, 'xy'), get_slab(n, 'xy')
            if img1 is not None and img2 is not None:
                res = register_pair_orb(img1, img2, axis=1) if use_orb else register_pair_hybrid(img1, img2, axis=1)
                if res is not None and (use_orb or (abs(res[0]) < img1.shape[0]*0.1)): # Sanity check for hybrid
                    constraints.append((t['id'], n['id'], res[0], res[1]))

        # Bottom Neighbor
        if (v+1, h) in grid_map:
            n = grid_map[(v+1, h)]
            img1, img2 = get_slab(t, 'xy'), get_slab(n, 'xy')
            if img1 is not None and img2 is not None:
                res = register_pair_orb(img1, img2, axis=0) if use_orb else register_pair_hybrid(img1, img2, axis=0)
                if res is not None and (use_orb or (abs(res[1]) < img1.shape[1]*0.1)):
                    constraints.append((t['id'], n['id'], res[0], res[1]))

    # --- Global Optimization ---
    print(f"Global Optimization on {len(constraints)} constraints...")
    num_tiles = len(processed_tiles)
    
    def residuals(params):
        coords = {0: (0, 0)}
        for i in range(num_tiles - 1): coords[i+1] = (params[2*i], params[2*i+1])
        res = []
        for (id_a, id_b, dy, dx) in constraints:
            idx_a = next(i for i, x in enumerate(processed_tiles) if x['id'] == id_a)
            idx_b = next(i for i, x in enumerate(processed_tiles) if x['id'] == id_b)
            pos_a = coords[idx_a] if idx_a != 0 else (0,0)
            pos_b = coords[idx_b] if idx_b != 0 else (params[2*(idx_b-1)], params[2*(idx_b-1)+1])
            res.append(pos_b[0] - pos_a[0] - dy)
            res.append(pos_b[1] - pos_a[1] - dx)
        return np.array(res)

    # Initial Guess
    guess_y_step = max_y * 0.9 if overlap not in ['auto', 'hybrid', 'features'] else max_y
    guess_x_step = max_x * 0.9
    
    # Check if manual overlap provided
    if isinstance(overlap, dict):
        guess_y_step = max_y - overlap.get('y', 0)
        guess_x_step = max_x - overlap.get('x', 0)

    x0 = []
    for i in range(1, num_tiles):
        t = processed_tiles[i]
        x0.extend([t['grid_v'] * guess_y_step, t['grid_h'] * guess_x_step])
        
    if constraints and overlap in ['auto', 'hybrid', 'features']:
        res_opt = least_squares(residuals, x0, loss='soft_l1', f_scale=1.0)
        optimized_params = res_opt.x
    else:
        optimized_params = x0

    final_coords = {} 
    final_coords[processed_tiles[0]['id']] = (0, 0)
    for i in range(num_tiles - 1):
        real_id = processed_tiles[i+1]['id']
        final_coords[real_id] = (optimized_params[2*i], optimized_params[2*i+1])

    # Normalize
    min_y = min(c[0] for c in final_coords.values())
    min_x = min(c[1] for c in final_coords.values())
    for tid in final_coords:
        y, x = final_coords[tid]
        final_coords[tid] = (int(y - min_y), int(x - min_x))

    # --- Rendering with Blending ---
    print("Rendering with Linear Blending...")
    canvas_h = int(max(final_coords[t['id']][0] + t['h'] for t in processed_tiles))
    canvas_w = int(max(final_coords[t['id']][1] + t['w'] for t in processed_tiles))
    
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    weights = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    
    def create_weight_mask(shape):
        h, w = shape
        wy = np.sin(np.pi * np.arange(h) / (h-1)) if h > 1 else np.ones(h)
        wx = np.sin(np.pi * np.arange(w) / (w-1)) if w > 1 else np.ones(w)
        return np.outer(wy, wx)

    for t in processed_tiles:
        img = get_slab(t, 'xy')
        if img is None: continue
        y_off, x_off = final_coords[t['id']]
        h, w = img.shape
        y_end = min(y_off + h, canvas_h)
        x_end = min(x_off + w, canvas_w)
        h_crop, w_crop = y_end - y_off, x_end - x_off
        if h_crop <= 0 or w_crop <= 0: continue
        
        img_crop = img[:h_crop, :w_crop]
        mask = create_weight_mask(img_crop.shape)
        canvas[y_off:y_end, x_off:x_end] += img_crop * mask
        weights[y_off:y_end, x_off:x_end] += mask

    valid_mask = weights > 1e-5
    canvas[valid_mask] /= weights[valid_mask]
    
    def prepare_plot(img):
        if vmax is not None: img = np.clip(img, 0, vmax)
        return np.log1p(np.maximum(img, 0))

    plt.figure(figsize=(10, 10))
    plt.imshow(prepare_plot(canvas), cmap=cmap)
    plt.title("Optimized & Blended XY Midplane")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()
    
def rotations(input_dir, output_dir, rotation_dict):
    """
    Step 2: Applies rotations to full 3D stacks and saves them to a new directory.
    This prepares the data for stitching.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    tiles = _scan_directory(input_dir)
    print(f"Rotating stacks from {input_dir} -> {output_dir}...")
    
    for t in tiles:
        k_rot = rotation_dict.get((t['v'], t['h']), 0)
        fname = t['filename']
        save_path = os.path.join(output_dir, fname)
        
        # Skip if already exists? Remove this check if you want to overwrite.
        # if os.path.exists(save_path): continue 

        try:
            img = tifffile.imread(t['path'])
            
            if k_rot > 0:
                if img.ndim == 3:
                    # Rotate axes 1 and 2 (Y and X), keeping Z (0) intact
                    img = np.rot90(img, k=k_rot, axes=(1, 2))
                else:
                    img = np.rot90(img, k=k_rot)
            
            tifffile.imwrite(save_path, img)
            print(f"Saved {fname} (Rot: {k_rot})")
            
        except Exception as e:
            print(f"Failed to rotate {fname}: {e}")
            
    print("Rotation complete.")

def stitch(input_dir, output_dir, switch_axes=False):
    """
    Step 3: Stitches the tiles found in `input_dir`.
    NOTE: input_dir should be the folder containing the ROTATED stacks (from Step 2).
    """
    tiles = _scan_directory(input_dir)
    if not tiles:
        print("No files to stitch.")
        return

    # Normalize coordinates
    min_v = min(t['v'] for t in tiles)
    min_h = min(t['h'] for t in tiles)
    
    max_z, max_y, max_x = 0, 0, 0
    grid_rows_max, grid_cols_max = 0, 0
    dtype = None

    print("Scanning dimensions for stitching...")
    for t in tiles:
        v, h = t['v'] - min_v, t['h'] - min_h
        if switch_axes:
            v, h = h, v
        
        grid_rows_max = max(grid_rows_max, v)
        grid_cols_max = max(grid_cols_max, h)
        
        # Store adjusted grid coordinates for later
        t['grid_v'] = v
        t['grid_h'] = h

        with tifffile.TiffFile(t['path']) as tif:
            series = tif.series[0]
            if dtype is None: dtype = series.dtype
            shape = series.shape
            
            if len(shape) == 2: z, y, x = 1, shape[0], shape[1]
            elif len(shape) == 3: z, y, x = shape
            else: continue
            
            max_z = max(max_z, z)
            max_y = max(max_y, y)
            max_x = max(max_x, x)

    rows = grid_rows_max + 1
    cols = grid_cols_max + 1
    
    stitched_shape = (max_z, rows * max_y, cols * max_x)
    total_bytes = np.prod(stitched_shape) * np.dtype(dtype).itemsize
    print(f"Allocating {total_bytes / 1e9:.2f} GB for final image...")
    
    stitched_img = np.zeros(stitched_shape, dtype=dtype)
    
    print("Stitching...")
    for t in tiles:
        img = tifffile.imread(t['path'])
        if img.ndim == 2: img = img[np.newaxis, :, :]
        
        cur_z, cur_y, cur_x = img.shape
        r, c = t['grid_v'], t['grid_h']
        
        y_start = r * max_y
        x_start = c * max_x
        
        z_end = min(max_z, cur_z)
        y_end = min(y_start + cur_y, stitched_shape[1])
        x_end = min(x_start + cur_x, stitched_shape[2])
        
        src_y = y_end - y_start
        src_x = x_end - x_start
        
        stitched_img[:z_end, y_start:y_end, x_start:x_end] = img[:z_end, :src_y, :src_x]
        print(f"Placed {t['filename']}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, "stitched_final.tif")
    print(f"Saving to {save_path}...")
    tifffile.imwrite(save_path, stitched_img)
    print("Done.")