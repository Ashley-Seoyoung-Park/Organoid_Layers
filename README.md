# Organoid Layer Analysis

This repository contains Python scripts for analyzing the layer structure of organoids, specifically focusing on Neural, Skeletal Muscle, and Smooth Muscle layers. The tools allow for individual organoid analysis (segmentation and quantification) and integrated analysis of multiple samples.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- Pandas (`pandas`)
- Matplotlib (`matplotlib`)
- Tifffile (`tifffile`)
- Roifile (`roifile`)
- Tkinter (usually included with Python)

Install dependencies via pip:
```bash
pip install opencv-python numpy pandas matplotlib tifffile roifile
```

## Scripts

### 1. `v3_overlaying_two_layers.py`
**Purpose:** Analyzes organoids with two distinct layers (Neural and Skeletal Muscle).
- **Inputs:** A folder containing channel images (e.g., `*CH01*`, `*CH02*`...).
- **Key Features:**
  - Automatic detection of Neural and Skeletal channels based on filename keywords.
  - Independent polygon mask generation for each layer.
  - Supports manual ROI correction via `neural_inner.roi` files.
  - **Outputs:**
    - Individual masks (PNG).
    - Overlay figures (Fig 1: Raw Merge, Fig 2: Mask Overlay).
    - Quantitative diameter profile data (CSV) and plot (Fig 3).

### 2. `v3_overlaying_three_layers.py`
**Purpose:** Analyzes organoids with three layers (Neural, Skeletal Muscle, and Smooth Muscle).
- **Inputs:** A folder containing channel images.
- **Key Features:**
  - Dynamic channel mapping based on folder name (e.g., `D132.SM22a.MyHC.MAP2`).
  - Enhanced neural gap filling logic.
  - Supports Smooth Muscle analysis via ROI zip files.
  - **Outputs:**
    - Individual masks for all three layers.
    - Comprehensive overlay figures and profile plots.

### 3. `integrated_analysis_v4.py`
**Purpose:** Aggregates results from multiple analysis folders.
- **Inputs:** A parent directory containing multiple subfolders processed by the scripts above.
- **Key Features:**
  - Recursively finds `Comprehensive_Analysis_Report` folders.
  - Aggregates `Diameter_Profile_Data.csv` files.
  - Generates a 5x5 visual grid of all analyzed samples for quality control.
  - Calculates Mean ± SEM for each layer across all samples.
  - **Outputs:**
    - `Final_Integrated_Raw_Data.csv` (all samples + statistics).
    - `Total_Combined_Profile_Graph.png` (Summary Graph).
    - `Alignment_Validation_Grid_5x5.png`.

## Usage

1. **Prepare your data:**
   - Organize your images into separate folders for each organoid.
   - Ensure filenames contain channel markers (e.g., `DAPI`, `MAP2`, `MyHC`, `SM22`).

2. **Run Single Analysis:**
   - Execute `v3_overlaying_two_layers.py` or `v3_overlaying_three_layers.py`.
   - Select the specific organoid folder when prompted.
   - Results will be saved in a `Comprehensive_Analysis_Report` subfolder.

3. **Run Integrated Analysis:**
   - After analyzing all individual samples, run `integrated_analysis_v4.py`.
   - Select the parent directory containing all the analyzed organoid folders.
   - The script will generate a summary of all processed data.

## Notes
- The scripts handle Unicode paths (e.g., Korean characters) using custom file I/O wrappers.
- If automatic segmentation fails, you can provide manual ImageJ ROIs (`.roi` or `.zip`) in the folder to override specific masks.
