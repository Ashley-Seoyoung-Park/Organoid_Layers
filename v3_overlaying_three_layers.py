
# 2026-02-10 Diameter profile, tilting organoids

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import tifffile as tiff
import pandas as pd
import roifile # Required for ROI processing

# --- [1] Utility: Unicode Path Support & Geometry ---
def imwrite_unicode(path, img):
    try:
        ret, buf = cv2.imencode(".png", img)
        if ret:
            with open(path, "wb") as f: f.write(buf)
            return True
    except Exception as e: print(f"  ✗ Save failed: {path} ({e})")
    return False

def get_angle(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0: return 180
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)))

def smooth_polygon(poly_pts, n, angle_threshold):
    for _ in range(3):
        new_pts = poly_pts.copy().astype(np.float32)
        for i in range(n):
            p_prev, p_curr, p_next = poly_pts[i-1], poly_pts[i], poly_pts[(i+1) % n]
            if get_angle(p_prev, p_curr, p_next) < angle_threshold:
                new_pts[i] = (p_prev + p_next) / 2
        poly_pts = new_pts.astype(np.int32)
    return poly_pts

def get_n_gon_points(contour, cx, cy, n):
    pts = contour.reshape(-1, 2)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    target_angles = np.linspace(-np.pi, np.pi, n + 1)[:-1]
    return np.array([pts[np.argmin(np.abs(angles - ta))] for ta in target_angles])

def get_display_image(img):
    if img is None: return None
    p2, p98 = np.percentile(img, (2, 98))
    rescaled = np.clip((img - p2) / (p98 - p2 + 1e-5), 0, 1)
    return (rescaled * 255).astype(np.uint8)

# --- [3] Independent Triple Mask Generation (Enhanced Neural Gap Filling) ---
def generate_masks(img_data, h, w, folder_path, files, n_vertices=40, save_dir=None):
    # A. Neural Mask: Outer (Auto) - Inner (Manual ROI 'neural_inner.roi' or Auto 40-polygon)
    neu_mask = np.zeros((h, w), dtype=np.uint8)
    if img_data['NEURAL'] is not None:
        norm_n = cv2.normalize(img_data['NEURAL'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Thresholding and connection logic
        _, bin_n = cv2.threshold(norm_n, 85, 255, cv2.THRESH_BINARY)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        bin_n_closed = cv2.morphologyEx(bin_n, cv2.MORPH_CLOSE, kernel_close)
        
        dist_n = cv2.distanceTransform(cv2.bitwise_not(bin_n_closed), cv2.DIST_L2, 5)
        mask_n = (dist_n < 15).astype(np.uint8) * 255
        
        # Use RETR_CCOMP to detect holes in hierarchy
        cnts, hier = cv2.findContours(mask_n, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if cnts and hier is not None:
            idx = np.argmax([cv2.contourArea(c) for c in cnts])
            full = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(full, [cnts[idx]], -1, 255, -1)
            # Contour Smoothing
            smooth_outer = cv2.threshold(cv2.GaussianBlur(full, (41, 41), 0), 127, 255, cv2.THRESH_BINARY)[1]
            
            # Try to load manual Inner ROI first
            h_mask = np.zeros((h, w), dtype=np.uint8)
            roi_inner_file = next((f for f in files if f.lower() in ["neural_inner.roi", "neural_inner.zip"]), None)
            
            if roi_inner_file:
                # Method 1: Use manual ROI file
                try:
                    roi_path = os.path.join(folder_path, roi_inner_file)
                    roi_data = roifile.ImagejRoi.fromfile(roi_path)
                    if isinstance(roi_data, list): roi_data = roi_data[0]
                    pts_inner = roi_data.coordinates().astype(np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(h_mask, [pts_inner], 255)
                    neu_mask = cv2.subtract(smooth_outer, h_mask)
                    print(f"  ✓ Neural Inner ROI (manual) applied: {roi_inner_file}")
                except Exception as e:
                    print(f"  ✗ Failed to load Neural ROI: {e}")
                    neu_mask = smooth_outer
            else:
                # Method 2: Automatic 40-vertex polygon from hierarchy
                holes = [(cv2.contourArea(cnts[i]), i) for i, h_info in enumerate(hier[0]) if h_info[3] == idx]
                if holes:
                    holes.sort(key=lambda x: x[0], reverse=True)
                    inner_cnt = cnts[holes[0][1]].reshape(-1, 2)
                    M = cv2.moments(inner_cnt)
                    if M['m00'] > 0:
                        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                        # Generate 40-vertex polygon from inner contour
                        inner_poly = get_n_gon_points(inner_cnt, cx, cy, n_vertices)
                        angle_threshold = ((n_vertices - 2) * 180 / n_vertices) - 30
                        inner_poly = smooth_polygon(inner_poly, n_vertices, angle_threshold)
                        cv2.fillPoly(h_mask, [inner_poly.reshape((-1, 1, 2))], 255)
                        neu_mask = cv2.subtract(smooth_outer, h_mask)
                        print(f"  ✓ Neural Inner Hole (automatic 40-polygon) applied")
                    else:
                        neu_mask = smooth_outer
                        print(f"  ⚠ Failed to calculate inner hole center, keeping outer contour only")
                else:
                    neu_mask = smooth_outer
                    print(f"  ⚠ Could not find inner hole, keeping outer contour only.")

    # B. Skeletal Mask (Polygon)
    ske_mask = np.zeros((h, w), dtype=np.uint8)
    if img_data['SKELETAL'] is not None:
        norm_s = cv2.normalize(img_data['SKELETAL'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, bin_s = cv2.threshold(norm_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. STANDARD SELECTION with SENSITIVE THRESHOLD
        # Problem: Otsu was too strict, making the bulk sparse.
        # Solution: Low fixed threshold to capture all dots, then Closing to merge.
        
        # Fixed Threshold instead of Otsu (User observed "erosion", so let's be generous)
        _, bin_s = cv2.threshold(norm_s, 30, 255, cv2.THRESH_BINARY)
        
        # A. Open to remove tiny noise (small speckles)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bin_s_opened = cv2.morphologyEx(bin_s, cv2.MORPH_OPEN, kernel_open)
        
        # B. Close to merge the dots into a single mass
        # Kernel 25 is strong enough if the mask is dense (which it is now)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        bin_s_closed = cv2.morphologyEx(bin_s_opened, cv2.MORPH_CLOSE, kernel_close)
        
        # C. Find Connected Components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_s_closed, connectivity=8)
        
        # D. Select Largest Component (The Bulk)
        # We need to filter out background (0)
        largest_mask = np.zeros((h, w), dtype=np.uint8)
        
        valid_components = []
        img_cx, img_cy = w // 2, h // 2
        max_radius = np.sqrt(img_cx**2 + img_cy**2)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            comp_cx, comp_cy = centroids[i]
            
            # Distance Filter (Relaxed to 95% just in case)
            dist = np.sqrt((comp_cx - img_cx)**2 + (comp_cy - img_cy)**2)
            if dist > max_radius * 0.95:
                continue
                
            valid_components.append((i, area))
        
        # Pick the winner
        if valid_components:
            largest_idx, _ = max(valid_components, key=lambda x: x[1])
            largest_mask[labels == largest_idx] = 255
            
        # E. Convex Hull for smoothness
        # Use the dots inside the selected region
        cnts_s, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = None
        if cnts_s:
            # Combine all contours if multiple (though largest_mask should imply one)
            all_pts = np.vstack([c for c in cnts_s])
            hull = cv2.convexHull(all_pts)
            
        # Save debug images
        if save_dir:
            imwrite_unicode(os.path.join(save_dir, "Debug_Skeletal_Binary.png"), bin_s)
            imwrite_unicode(os.path.join(save_dir, "Debug_Skeletal_Closed.png"), bin_s_closed)
            imwrite_unicode(os.path.join(save_dir, "Debug_Skeletal_Largest.png"), largest_mask)
            
        # F. Refined Polygon Generation from Hull
        if hull is not None:
            M = cv2.moments(hull)
            if M['m00'] > 0:
                cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                # Create smooth 40-gon from the convex hull
                s_pts = smooth_polygon(get_n_gon_points(hull, cx, cy, n_vertices), n_vertices, ((n_vertices-2)*180/n_vertices)-30)
                cv2.fillPoly(ske_mask, [s_pts.reshape((-1, 1, 2))], 255)
            else: 
                cv2.fillPoly(ske_mask, [hull], 255)

    # C. Smooth Muscle Mask (ROI-based)
    sm_mask = np.zeros((h, w), dtype=np.uint8)
    roi_zip = [f for f in files if f.lower().endswith('.zip')]
    if roi_zip:
        try:
            rois = roifile.ImagejRoi.fromfile(os.path.join(folder_path, roi_zip[0]))
            if not isinstance(rois, list): rois = [rois]
            if len(rois) >= 2:
                rois_s = sorted(rois, key=lambda r: cv2.contourArea(r.coordinates().astype(np.int32)), reverse=True)
                out_m, in_m = np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(out_m, [rois_s[0].coordinates().astype(np.int32).reshape((-1, 1, 2))], 255)
                cv2.fillPoly(in_m, [rois_s[1].coordinates().astype(np.int32).reshape((-1, 1, 2))], 255)
                sm_mask = cv2.subtract(out_m, in_m)
        except Exception as e: print(f"  ✗ ROI Error: {e}")
            
    return ske_mask, neu_mask, sm_mask

# --- [3] Main Pipeline ---
def main():
    root = Tk(); root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Organoid Folder")
    if not folder_path: return
    save_dir = os.path.join(folder_path, "Comprehensive_Analysis_Report")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Parse folder name to detect marker-to-channel mapping
    folder_name = os.path.basename(folder_path)
    print(f"\nAnalyzing Folder: {folder_name}")
    
    # Define marker categories
    neural_markers = ['MAP2', 'NEUN', 'NFM']
    skeletal_markers = ['MYHC', 'MYH3', 'MYH']
    smooth_markers = ['SM22A', 'SM22', 'ASM22', 'ACTA2', 'SMA']
    
    # Detect which markers are in folder name
    marker_map = {}  # marker_type -> detected_marker_name
    for marker in neural_markers:
        if marker.upper() in folder_name.upper():
            marker_map['NEURAL'] = marker
            print(f"  - NEURAL Marker Detected: {marker}")
            break
    for marker in skeletal_markers:
        if marker.upper() in folder_name.upper():
            marker_map['SKELETAL'] = marker
            print(f"  - SKELETAL Marker Detected: {marker}")
            break
    for marker in smooth_markers:
        if marker.upper() in folder_name.upper():
            marker_map['SMOOTH'] = marker
            print(f"  - SMOOTH Marker Detected: {marker}")
            break
    
    # 2. Build channel mapping based on FOLDER NAME ORDER
    # Logic: DAPI is always CH00.
    # The order of markers in the folder name determines CH01, CH02, CH03.
    # Example: "D132.SM22a.MyHC.MAP2" -> SM22a=CH01, MyHC=CH02, MAP2=CH03
    
    img_data = {'NEURAL': None, 'SKELETAL': None, 'SMOOTH': None, 'DAPI': None}
    
    # Check marker presence and order in folder name
    detected_markers = [] # List of (index_in_name, marker_type, marker_name)
    
    for m in neural_markers:
        if m.upper() in folder_name.upper():
            detected_markers.append((folder_name.upper().find(m.upper()), 'NEURAL', m))
            break
    for m in skeletal_markers:
        if m.upper() in folder_name.upper():
            detected_markers.append((folder_name.upper().find(m.upper()), 'SKELETAL', m))
            break
    for m in smooth_markers:
        if m.upper() in folder_name.upper():
            detected_markers.append((folder_name.upper().find(m.upper()), 'SMOOTH', m))
            break
            
    # Sort by position in folder name
    detected_markers.sort(key=lambda x: x[0])
    
    # Assign Channels
    channel_map = {}
    channel_map['DAPI'] = ['CH00', 'CH0', 'DAPI'] # Always CH00
    
    if len(detected_markers) > 0:
        print(f"  [Auto-Map] Folder Name Dynamic Mapping:")
        for i, (idx, m_type, m_name) in enumerate(detected_markers):
            ch_num = f"CH0{i+1}" # CH01, CH02, CH03...
            channel_map[m_type] = [ch_num, f"CH{i+1}"]
            print(f"    - {ch_num} -> {m_type} ({m_name})")
    else:
        print("  [Auto-Map] ⚠ Could not find marker names in folder name. Using default settings.")
        channel_map = {
            'NEURAL': ['CH03', 'CH3'],
            'SKELETAL': ['CH01', 'CH1'],
            'SMOOTH': ['CH02', 'CH2'],
            'DAPI': ['CH00', 'CH0', 'DAPI']
        }

    # Load Files with Strict Checking
    files = os.listdir(folder_path)
    for target in ['NEURAL', 'SKELETAL', 'SMOOTH', 'DAPI']:
        if target not in channel_map: continue
        
        search_keywords = channel_map[target]
        found_file = None
        
        # 1. Try Specific Channel Number detected from mapping
        for k in search_keywords:
            # Filter: Check keyword AND exclude "OVERLAY"/"COMPOSITE"
            # Allow "MERGE" because "Merging" is common in TileScan filenames
            candidates = [f for f in files if k.upper() in f.upper() and f.lower().endswith(('.tif', '.tiff'))]
            valid_candidates = [f for f in candidates if "OVERLAY" not in f.upper() and "COMPOSITE" not in f.upper()]
            
            if valid_candidates:
                # Prefer shortest name usually? Or first?
                # User's file: TileScan_001_Merging_Crop_Processed001_ch01.tif
                found_file = valid_candidates[0] 
                break
                
        # 2. Fallback: Search for Marker Name (only if no channel number found)
        # (Only if we know the marker name from typical lists)
        if not found_file and target != 'DAPI':
            ref_list = neural_markers if target == 'NEURAL' else (skeletal_markers if target == 'SKELETAL' else smooth_markers)
            for m in ref_list:
                if m.upper() in folder_name.upper(): # Only search if this marker is relevant
                    candidates = [f for f in files if m.upper() in f.upper() and f.lower().endswith(('.tif', '.tiff'))]
                    valid_candidates = [f for f in candidates if "OVERLAY" not in f.upper() and "COMPOSITE" not in f.upper()]
                    if valid_candidates:
                        found_file = valid_candidates[0]
                        break
        
        if found_file:
            path = os.path.join(folder_path, found_file)
            try:
                raw = tiff.imread(path)
                img_data[target] = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY) if len(raw.shape) == 3 else np.squeeze(raw)
                print(f"  ✓ {target} Loaded successfully: {found_file}")
            except Exception as e:
                print(f"  ✗ {target} Load failed ({found_file}): {e}")
        elif target == 'DAPI':
             # DAPI is critical for shape, but maybe not strictly required if other images define shape?
             # But generating masks usually needs a base size.
             print(f"  ⚠ {target} Channel file not found (Keywords: {search_keywords})")
        else:
             print(f"  ⚠ {target} Channel file not found (Keywords: {search_keywords})")

    # Safety Check: Did we get anything at all?
    if img_data['DAPI'] is None:
        # Fallback: Try to get shape from ANY loaded image if DAPI failed
        for key in ['NEURAL', 'SKELETAL', 'SMOOTH']:
            if img_data[key] is not None:
                h, w = img_data[key].shape[:2]
                print(f"  ⚠ Failed to load DAPI. Proceeding with '{key}' image size ({w}x{h}).")
                # Create empty DAPI placeholder
                img_data['DAPI'] = np.zeros((h, w), dtype=np.uint8)
                break
        else:
             print("  ✗ [Critical Error] Could not load any channel images. Aborting.")
             return

    h, w = img_data['DAPI'].shape[:2]
    ske_m, neu_m, sm_m = generate_masks(img_data, h, w, folder_path, files, save_dir=save_dir)

    # 2. Save results (Individual Masks and Figures)
    for m, name, color in [(neu_m, "NEURAL_Green", [0, 255, 0]), (ske_m, "SKELETAL_Red", [0, 0, 255]), (sm_m, "SMOOTH_Purple", [128, 0, 128])]:
        tmp = np.zeros((h, w, 3), dtype=np.uint8); tmp[m > 0] = color
        imwrite_unicode(os.path.join(save_dir, f"Individual_Mask_{name}.png"), tmp)

    # Fig 1: Raw Merge (Ske-R, Neu-G, SM-Purple)
    ch_n, ch_s, ch_sm, ch_d = get_display_image(img_data['NEURAL']), get_display_image(img_data['SKELETAL']), get_display_image(img_data['SMOOTH']), get_display_image(img_data['DAPI'])
    fig1 = np.zeros((h, w, 3), dtype=np.uint8)
    if ch_s is not None: fig1[:,:,2] = ch_s # R
    if ch_n is not None: fig1[:,:,1] = ch_n # G
    if ch_sm is not None: fig1[:,:,2] = cv2.add(fig1[:,:,2], ch_sm//2); fig1[:,:,0] = cv2.add(fig1[:,:,0], ch_sm//2) # Purple (R+B)
    imwrite_unicode(os.path.join(save_dir, "Fig1_Raw_Merge.png"), fig1)

    # Fig 2: Translucent Mask Overlay (Neu-G, Ske-R, SM-Purple)
    ov_n, ov_s, ov_sm = np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w, 3), dtype=np.uint8)
    ov_n[neu_m > 0] = [0, 255, 0]; ov_s[ske_m > 0] = [0, 0, 255]; ov_sm[sm_m > 0] = [128, 0, 128]
    fig2 = cv2.addWeighted(ov_n, 0.6, ov_s, 0.6, 0); fig2 = cv2.addWeighted(fig2, 1.0, ov_sm, 0.6, 0)
    imwrite_unicode(os.path.join(save_dir, "Fig2_Mask_Overlay.png"), fig2)

    # Fig 3: CSV & Plot
    M_n = cv2.moments(neu_m); cx, cy = (int(M_n['m10']/M_n['m00']), int(M_n['m01']/M_n['m00'])) if M_n['m00'] > 0 else (w//2, h//2)
    M_s = cv2.moments(ske_m); angle = np.arctan2(M_s['m01']/M_s['m00'] - cy, M_s['m10']/M_s['m00'] - cx) if M_s['m00'] > 0 else 0
    max_r = np.max(np.sqrt((np.indices((h, w))[1]-cx)**2 + (np.indices((h, w))[0]-cy)**2)[neu_m>0]) if np.any(neu_m > 0) else w/2
    l_r = np.linspace(-max_r, max_r, 256); csv_data = {'Relative_Radius': np.linspace(-1, 1, 256)}
    
    plt.figure(figsize=(8, 6), facecolor='white')
    for key, m, c in [('Neu', neu_m, 'green'), ('Ske', ske_m, 'red'), ('SM', sm_m, 'purple')]:
        vals = [100 if (0 <= int(cy+r*np.sin(angle)) < h and 0 <= int(cx+r*np.cos(angle)) < w and m[int(cy+r*np.sin(angle)), int(cx+r*np.cos(angle))] > 0) else 0 for r in l_r]
        csv_data[f'{key}_Occupancy'] = vals
        plt.plot(csv_data['Relative_Radius'], vals, color=c, lw=2.5, label=key)
    pd.DataFrame(csv_data).to_csv(os.path.join(save_dir, "Diameter_Profile_Data.csv"), index=False)
    plt.axvline(0, color='black', lw=1, ls='--'); plt.legend(); plt.savefig(os.path.join(save_dir, "Fig3_Profile_Plot.png"), dpi=300); plt.close()

    # Final Report
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='white')
    axes[0].imshow(cv2.cvtColor(fig1, cv2.COLOR_BGR2RGB)); axes[0].axis('off'); axes[0].set_title("Figure 1: Raw Merge")
    axes[1].imshow(cv2.cvtColor(fig2, cv2.COLOR_BGR2RGB)); axes[1].axis('off'); axes[1].set_title("Figure 2: Mask Overlay")
    axes[2].plot(csv_data['Relative_Radius'], csv_data['Neu_Occupancy'], 'g-', label='Neu')
    axes[2].plot(csv_data['Relative_Radius'], csv_data['Ske_Occupancy'], 'r-', label='Ske')
    axes[2].plot(csv_data['Relative_Radius'], csv_data['SM_Occupancy'], color='purple', label='SM')
    axes[2].axvline(0, color='black', lw=1, ls='--'); axes[2].legend(); axes[2].set_title("Figure 3: Diameter Profile")
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "Final_Integrated_Report.png"), dpi=300); plt.show()

if __name__ == "__main__":
    main()