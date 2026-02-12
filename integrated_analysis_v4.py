
# Integrated Analysis V4: Aggregation & Visualization Only (No Segmentation)
# 2026-02-12

import os
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff

# --- [0] Unicode-safe Image Loading ---
def safe_imread(path):
    """
    Reads an image from a path that may contain non-ASCII characters.
    """
    try:
        # Use np.fromfile to read binary data, then decode with OpenCV
        stream = np.fromfile(path, np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

# --- [3] Main Integration Pipeline ---
def run_integrated_analysis():
    # 1. Select Parent Directory
    root = tk.Tk()
    root.withdraw() # Hide the main window
    root.attributes('-topmost', True) # Force the dialog to appear on top
    root.lift()
    root.focus_force()
    
    parent_dir = filedialog.askdirectory(title="Select the parent folder containing analysis results")
    root.destroy() # Clean up
    if not parent_dir: return

    # Definitions
    profiles = {'Neu': [], 'Ske': [], 'SM': []}
    # Track valid samples for averaging (Indices of valid masks)
    valid_samples = {'Neu': [], 'Ske': [], 'SM': []}
    
    sample_names = []
    # We will read these from the first CSV
    common_x = None 
    
    # Visualization Grid Setup (5x5)
    # We will load existing PNGs instead of creating new plots
    fig_grid, axes_grid = plt.subplots(5, 5, figsize=(25, 25), facecolor='white')
    axes_flat = axes_grid.flatten()
    grid_idx = 0
    print("\n[Integrated Analysis V4.9] Aggregating results with Fuzzy Mask Search & 5x5 Grid...")
    
    num_pts = 200 # Define resolution for profile line

    # Helper function for fuzzy keyword file finding
    def find_file_fuzzy(directory, keywords, debug=False):
        if not os.path.exists(directory):
            if debug: print(f"    (Debug) Path missing: {directory}")
            return None
        
        files = os.listdir(directory)
        candidates = []
        for f in files:
            lower_name = f.lower()
            # Check if ALL keywords are present
            if all(k.lower() in lower_name for k in keywords):
                # prioritize png/tif
                if lower_name.endswith('.png') or lower_name.endswith('.tif'):
                    candidates.append(f)
        
        if candidates:
            # Filter out "graph" just in case
            valid_candidates = [c for c in candidates if "graph" not in c.lower()]
            if valid_candidates:
                full = os.path.join(directory, valid_candidates[0])
                if debug: print(f"    (Debug) Found Fuzzy Match: {valid_candidates[0]}")
                return full
        
        if debug: print(f"    (Debug) No fuzzy match for {keywords} in {directory}")
        return None

    # Update safe_imread to support color
    def safe_imread(path, color=False):
        try:
            stream = np.fromfile(path, np.uint8)
            flags = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
            img = cv2.imdecode(stream, flags)
            return img
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return None

    # 2. Walk through folders
    sample_idx = 0 # Track global index
    folders_scanned = 0
    
    for root_path, dirs, files in os.walk(parent_dir):
        folders_scanned += 1
        
        # Match logic: target 'Comprehensive_Analysis_Report' folder
        if "Comprehensive_Analysis_Report" in os.path.basename(root_path):
            
            # 1. Find the Data CSV (Statistical Source)
            csv_files = [f for f in files if f.lower().endswith('.csv')]
            if not csv_files:
                print(f"  [Skip] {os.path.basename(root_path)}: No CSV found.")
                continue
                
            csv_path = os.path.join(root_path, csv_files[0]) # Use the first one found
            
            # 2. Find the Raw Masks (Visualization Source)
            # Try Parent AND Report Folder
            result_dir = os.path.dirname(root_path)
            report_dir = root_path
            
            search_locs = [result_dir, report_dir]
            
            vis_method = "None"
            neu_m = None; ske_m = None; sm_m = None

            # Debug output for first sample to trace issue
            enable_debug = (sample_idx == 0) 
            if enable_debug: print(f"  [Debug Probe] Checking masks for first sample...")

            # Robust Search for Masks (Fuzzy)
            neu_path = None; ske_path = None; sm_path = None
            
            for loc in search_locs:
                if not neu_path: neu_path = find_file_fuzzy(loc, ["mask", "neural"], debug=enable_debug)
                if not ske_path: ske_path = find_file_fuzzy(loc, ["mask", "skeletal"], debug=enable_debug)
                if not sm_path:  sm_path  = find_file_fuzzy(loc, ["mask", "smooth"], debug=False) # Optional
            
            if neu_path and ske_path:
                # Load as Grayscale for Mask Logic
                neu_m = safe_imread(neu_path, color=False)
                ske_m = safe_imread(ske_path, color=False)
                if sm_path: sm_m = safe_imread(sm_path, color=False)
                
                if neu_m is not None and ske_m is not None:
                    vis_method = "Masks"
                elif enable_debug:
                    print(f"    (Debug) Found paths but imread returned None.")

            # Fallback: Find the Visual Image (Fig2) if Masks fail
            img_path = None
            if vis_method == "None":
                possible_imgs = ["Fig2_Mask_Overlay.png", "Fig2.png", "Overlay.png"]
                for img_name in possible_imgs:
                    # Check in report folder (root_path)
                    candidates = find_file_fuzzy(root_path, [img_name.split('.')[0]]) # Loose check
                    if candidates:
                         img_path = candidates
                         vis_method = "Fig2"
                         break
            
            # 3. Process Data
            try:
                # Use relative path of the PARENT of the report folder for sample name
                # distinct from just the report folder itself
                report_parent = os.path.dirname(root_path)
                rel_path = os.path.relpath(report_parent, parent_dir)
                sample_name = rel_path.replace(os.sep, "_").replace("Result_", "")
                
                print(f"Processing: {sample_name} (Vis: {vis_method})")

                df = pd.read_csv(csv_path)
                
                # Standardize Column Names if needed
                if common_x is None and 'Relative_Radius' in df.columns:
                    common_x = df['Relative_Radius'].values
                
                # --- Store Data ---
                has_neu = False; has_ske = False; has_sm = False
                
                if 'Neu_Occupancy' in df.columns:
                    profiles['Neu'].append(df['Neu_Occupancy'].values)
                    has_neu = True
                    valid_samples['Neu'].append(sample_idx)
                else: 
                    profiles['Neu'].append(np.zeros_like(common_x) if common_x is not None else [])

                if 'Ske_Occupancy' in df.columns:
                    profiles['Ske'].append(df['Ske_Occupancy'].values)
                    has_ske = True
                    valid_samples['Ske'].append(sample_idx)
                else:
                    profiles['Ske'].append(np.zeros_like(common_x) if common_x is not None else [])

                if 'SM_Occupancy' in df.columns:
                    sm_data = df['SM_Occupancy'].values
                    profiles['SM'].append(sm_data)
                    if np.max(sm_data) > 0:
                        has_sm = True
                        valid_samples['SM'].append(sample_idx)
                else:
                    profiles['SM'].append(np.zeros_like(common_x) if common_x is not None else [])
                
                sample_names.append(sample_name)
                sample_idx += 1
                
                # --- Add to Visual Grid ---
                if grid_idx < 25:
                    ax = axes_flat[grid_idx]
                    
                    if vis_method == "Masks":
                        # Re-calculate Moments & Line
                        h, w = neu_m.shape
                        M_n = cv2.moments(neu_m)
                        if M_n['m00'] != 0:
                            cx, cy = int(M_n['m10']/M_n['m00']), int(M_n['m01']/M_n['m00'])
                            M_s = cv2.moments(ske_m)
                            if M_s['m00'] != 0:
                                sx, sy = int(M_s['m10']/M_s['m00']), int(M_s['m01']/M_s['m00'])
                                peak_angle = np.arctan2(sy - cy, sx - cx)
                                
                                # Profile Line
                                y_idx, x_idx = np.indices((h, w))
                                dists = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
                                max_r_val = np.max(dists[neu_m > 0]) * 1.1 if np.any(neu_m > 0) else w/2
                                line_r = np.linspace(-max_r_val, max_r_val, num_pts) if num_pts else np.linspace(-100,100,100)
                                sample_x = cx + line_r * np.cos(peak_angle)
                                sample_y = cy + line_r * np.sin(peak_angle)
                                
                                # Draw Composite
                                comp = np.zeros((h, w, 3), dtype=np.uint8)
                                comp[neu_m > 0] = [0, 200, 0]   # Green
                                comp[ske_m > 0] = [200, 0, 0]   # Red
                                if sm_m is not None: comp[sm_m > 0] = [128, 0, 128] # Purple
                                
                                ax.imshow(comp)
                                ax.plot(sample_x, sample_y, 'w--', lw=1.5, alpha=0.8)
                                ax.plot(sample_x[0], sample_y[0], 'yo', markersize=5)
                            else:
                                ax.text(0.5, 0.5, "Moments Fail", ha='center', color='black')
                        else:
                             ax.text(0.5, 0.5, "Empty Neu", ha='center', color='black')

                    elif vis_method == "Fig2":
                         if os.path.exists(img_path):
                            # Safe Load WITH COLOR
                            img = safe_imread(img_path, color=True) 
                            if img is not None:
                                # imread loads BGR, convert to RGB
                                if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                ax.imshow(img)
                            else:
                                ax.text(0.5, 0.5, "Img Load Fail", ha='center')
                    else:
                        ax.text(0.5, 0.5, "No Vis (Masks Missing)", ha='center')
                        
                    ax.set_title(sample_name[:20], fontsize=8)
                    ax.axis('off')
                    grid_idx += 1
                    
            except Exception as e:
                print(f"  ✗ Error processing {sample_name}: {e}")
                continue

    # 3. Final Outputs
    # A. Grid Image
    for i in range(grid_idx, 25): axes_flat[i].axis('off')
    plt.tight_layout()
    grid_path = os.path.join(parent_dir, "Alignment_Validation_Grid_5x5.png")
    fig_grid.savefig(grid_path, dpi=200)
    plt.close(fig_grid)
    print(f"\n[Output] Validation Grid Saved: {grid_path}")
    
    # B. CSV Aggregation
    if common_x is not None:
        export_dict = {'Relative_Radius': common_x}
        for key in ['Neu', 'Ske', 'SM']:
            data_list = profiles[key]
            if not data_list: continue
            
            data_array = np.array(data_list)
            valid_indices = valid_samples[key]
            
            if len(valid_indices) > 0:
                # Use only VALID samples for Mean/SEM
                valid_data = data_array[valid_indices]
                mean_vals = np.mean(valid_data, axis=0)
                sem_vals = np.std(valid_data, axis=0) / np.sqrt(len(valid_data))
                
                export_dict[f'{key}_Mean'] = mean_vals
                export_dict[f'{key}_SEM'] = sem_vals
            else:
                print(f"[Info] No valid masks found for {key}. Skipping Mean/SEM.")

            # Individual Traces (Still output all, even zeros)
            for i, s_name in enumerate(sample_names):
                if i < len(data_array):
                    export_dict[f'{key}_{s_name}'] = data_array[i]
        
        # Save Master CSV
        df_export = pd.DataFrame(export_dict)
        csv_path = os.path.join(parent_dir, "Final_Integrated_Raw_Data.csv")
        df_export.to_csv(csv_path, index=False)
        print(f"[Output] Master CSV Saved: {csv_path}")
        
        # C. Summary Graph
        try:
            fig_sum, ax_sum = plt.subplots(figsize=(10, 7), facecolor='white')
            
            colors = {'Neu': 'green', 'Ske': 'red', 'SM': 'purple'}
            plots_drawn = False
            for key in ['Neu', 'Ske', 'SM']:
                if f'{key}_Mean' in df_export.columns:
                    x = df_export['Relative_Radius']
                    y = df_export[f'{key}_Mean']
                    err = df_export[f'{key}_SEM']
                    
                    ax_sum.plot(x, y, label=f'{key} Layer', color=colors[key], linewidth=3)
                    ax_sum.fill_between(x, y-err, y+err, color=colors[key], alpha=0.15)
                    plots_drawn = True
            
            if plots_drawn:
                ax_sum.axvline(0, color='black', lw=1, ls='--')
                ax_sum.set_xlabel("Normalized Radius")
                ax_sum.set_ylabel("Occupancy (%)")
                ax_sum.set_title("Integrated Spatial Profile (Mean ± SEM)")
                ax_sum.legend()
                ax_sum.grid(True, alpha=0.2)
                
                graph_path = os.path.join(parent_dir, "Total_Combined_Profile_Graph.png")
                fig_sum.savefig(graph_path, dpi=300)
                print(f"[Output] Summary Graph Saved: {graph_path}")
            else:
                print("[Info] No aggregated data available to plot.")
            plt.close(fig_sum)
            
        except Exception as e:
            print(f"Error generating summary plot: {e}")
            
    else:
        print("No valid data found to aggregate (No 'Relative_Radius' column found).")
        
    print(f"\n[Done] Scanned {folders_scanned} folders, Processed {sample_idx} samples.")

if __name__ == "__main__":
    run_integrated_analysis()
