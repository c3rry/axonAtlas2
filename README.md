# AxonAtlas2

## Overview
**AxonAtlas2** is a large-scale computational pipeline designed to analyze the axonal content and cellular architecture of high-throughput 3D light-sheet imaged mouse brains. The ultimate goal of this project is to accurately quantify the rate of neurodegeneration in Parkinson's Disease (PD) ridden mice by mapping axon density, individual cell coordinates, and complex mesoscale connectivity. 

By transforming raw light-sheet imaged volumes into Allen Atlas registered segmentation masks and running sophisticated clustering algorithms, AxonAtlas2 allows for detailed region-based quantification, 3D circuitry visualization, and 2D connectome mapping.

---

## Pipeline Components & Notebooks

The repository is modularized into several Jupyter Notebooks and directories, each handling a distinct phase of the volumetric analysis workflow:

### `AxonAtlas2_post_stitch.ipynb`
This notebook serves as an early data preparation step following raw image acquisition. It handles the separation of distinct imaging channels (e.g., separating the fluorescent signal channel from the autofluorescence background channel) and condenses each isolated channel into its corresponding 3D TIFF stack for memory-efficient downstream processing.

### `AxonAtlas2_processing.ipynb` *(Primary Pipeline for Axon Data)*
This is the core preprocessing and segmentation engine for axon data. It sequentially handles:
* **Surgical Masking:** Utilizes SciPy to perform subtractive binary erosion, which removes unnecessary border signals and imaging artifacts that often decrease the performance of downstream segmentation.
* **Downscaling:** Reduces the initially massive raw volumes by a factor of 0.5 across the X, Y, and Z dimensions to make computation feasible.
* **Axon Segmentation:** Deploys a pre-trained UNET model (TrailMap: https://github.com/albert597/TRAILMAP) to generate a pixel-by-pixel segmentation mask where pixel intensity represents the probability of axon classification (from black for "no axon" to white for "100% axon").
* **Image Binary Dim Composition:** Overlays a binarized version of the segmentation mask onto a scaled-down, dimmed raw volume. This creates a composite where the axonal tracts brightly "pop out" against the structural outline of the brain.
* **Atlas Registration:** Uses BrainReg (https://brainglobe.info/documentation/brainreg/index.html) to apply both affine (global) and b-spline (local deformation) transformations, aligning the raw sample volume precisely to the 25µm Allen Mouse Brain Atlas coordinate space.
* **Binned Skeletonization:** Extracts axonal content by segmenting pixel probabilities into 10 distinct volumetric bins, ascendingly scaling and heavily weighting high-probability bins to isolate the final structural paths.

### `AxonAtlas2_quantification.ipynb` *(Primary Analysis for Axon Data)*
This notebook calculates quantitative metrics from the registered pipeline outputs. It utilizes tools like BrainGlobe to perform region condensation—collapsing hierarchical brain structure trees into parent structures based on specified depths. This allows for the calculation of relative fractional and projection axon densities for each target region. The notebook also generates distribution plots, data summaries, and thresholded heatmaps.

### `AxonAtlas2_cells.ipynb`
Dedicated to cellular-level mapping, this notebook handles 3D cell detection via BrainMapper (https://brainglobe.info/documentation/brainglobe-workflows/index.html). By comparing the fluorescently labeled cell signal channel against the background autofluorescence channel, it extracts raw coordinates for millions of individual cells and projects them into the standardized Allen Atlas space.

### `AxonAtlas2_connections.ipynb` & `AxonAtlas2_clustering.ipynb`
These notebooks analyze the complex networks of simple neurite tracer data. They ingest CSV files containing extracted numeric neuron features and utilize advanced clustering methods, specifically K-Means and DBSCAN. The algorithms autonomously determine optimal cluster counts (maximizing the Silhouette Score) and subsequently organize the `.swc` files into localized directories categorized by their structural group and anatomical region.

### `AxonAtlas2_swcAnalysis.ipynb`
This notebook processes and visualizes the structured `.swc` tracer files post-clustering. It prepares the 3D coordinates and topological data of the reconstructed individual neurons to allow for detailed morphological evaluation. Also creates swc_coordinate.csv and swc_region_summary.csv, which are nescasary for connectome_2D and clustering tasks.

### `AxonAtlas2_visualization.ipynb`
Responsible for the spatial rendering of processed TIFF stacks within the Allen Atlas space. This notebook generates interactive 3D visualizations using tools like BrainRender to showcase whole-brain axon pathways and overlay pairwise subtractive comparison heatmaps directly onto brain models.

### `AxonAtlas2_sankey.ipynb`
A specialized visualization notebook that ingests analyzed `.swc` projection data to generate interactive Sankey diagrams. These flow diagrams illustrate mesoscale connectivity, displaying how output pathways diverge from specific source regions (like the GPe, GPi, or SNr) toward various downstream cortical and subcortical targets.

### `/brainJ`
This directory contains scripts and data specific to the mesoscale analysis of a 7-mouse dataset originally processed using the BrainJ pipeline. It focuses on multi-sample aggregation and calculating relative regional densities across groups to identify overarching neuroanatomical trends.

### `/conectome_2D`
This directory houses a comprehensive pipeline for translating complex 3D neuronal structures into 2D mesoscale connectomes. It includes scripts to generate visually dense analytical graphics, including circular chord/ribbon plots, grouped scatter/dot plots for cluster comparison, and precise pie charts that map the distributed output architecture of regions like the basal ganglia.

### `AxonAtlas2_stitching.ipynb` *(DEPRECATED)*
A legacy notebook initially designed to handle the localized preprocessing and X/Y/Z alignment of raw image tiles specifically generated by the Ding Lab microscope. This step has been deprecated in favor of external stitching software protocols.
