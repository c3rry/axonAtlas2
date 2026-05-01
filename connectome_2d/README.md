# 2D Connectome Pipeline

This project generates and visualizes region-level brain connectivity from SWC-derived data.

Pipeline:
**analyze → build → visualize**

---

## Setup

Place region-specific datasets in `data/`:

- `gpeswc_region_summary.csv`
- `gpi_swc_region_summary.csv`
- `stn_swc_region_summary.csv`
- `snr_swc_region_summary.csv` (optional)

---

## Usage

### 1. Analyze
python src\1_analyze_data.py --region GPe --side Left --exclude_outside_brain

Modes:

Left: --side Left
Right: --side Right
Whole brain: omit --side

### 2. Build
python src\2_build_circuit.py --region GPe --side Left

### 3. Visualize
python src\3_visualize_graph.py --region GPe --side Left

### Examples
# GPe Left
python src\1_analyze_data.py --region GPe --side Left --exclude_outside_brain
python src\2_build_circuit.py --region GPe --side Left
python src\3_visualize_graph.py --region GPe --side Left

# GPi Right
python src\1_analyze_data.py --region GPi --side Right --exclude_outside_brain
python src\3_visualize_graph.py --region GPi --side Right

# STN Whole Brain
python src\1_analyze_data.py --region STN --exclude_outside_brain
python src\3_visualize_graph.py --region STN
