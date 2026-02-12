
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import tifffile as tiff
import pandas as pd
import roifile # Required for ROI processing

# --- [1] Utility for Unicode Path Support ---
def imwrite_unicode(path, img):
    """Save image using buffer to prevent errors with Unicode paths in OpenCV"""
    try:
        ret, buf = cv2.imencode(".png", img)
        if ret:
            with open(path, "wb") as f:
                f.write(buf)
            return True
    except Exception as e:
        print(f"  ✗ Save failed: {path} ({e})")
    return False

# --- [2] Geometry and Polygon Logic ---
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

# --- [3] Independent Polygon Mask Generation ---
def generate_independent_masks(img_data, h, w, folder_path, files, n_vertices=40):
    # A. Neural Mask (Polygon Donut)
    neu_mask = np.zeros((h, w), dtype=np.uint8)
    if img_data['NEURAL'] is not None:
        norm_n = cv2.normalize(img_data['NEURAL'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, bin_n = cv2.threshold(norm_n, 130, 255, cv2.THRESH_BINARY)
        dist_n = cv2.distanceTransform(cv2.bitwise_not(bin_n), cv2.DIST_L2, 5)
        mask_n = (dist_n < 12).astype(np.uint8) * 255
        cnts, hier = cv2.findContours(mask_n, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if cnts and hier is not None:
            idx = np.argmax([cv2.contourArea(c) for c in cnts])
            full = np.zeros((h, w), dtype=np.uint8); cv2.drawContours(full, [cnts[idx]], -1, 255, -1)
            outer = cv2.threshold(cv2.GaussianBlur(full, (51, 51), 0), 127, 255, cv2.THRESH_BINARY)[1]
            
            # --- START: Neural Inner ROI Check ---
            # Try to load manual Inner ROI first
            h_mask = np.zeros((h, w), dtype=np.uint8)
            # Find neural_inner.roi or .zip (case-insensitive)
            roi_inner_file = next((f for f in files if f.lower() in ["neural_inner.roi", "neural_inner.zip"]), None)
            
            if roi_inner_file:
                # Method 1: Use manual ROI file
                try:
                    roi_path = os.path.join(folder_path, roi_inner_file)
                    roi_data = roifile.ImagejRoi.fromfile(roi_path)
                    # roifile might return a list if zip or single object if roi
                    if isinstance(roi_data, list): roi_data = roi_data[0]
                    
                    # Get coordinates
                    coords = roi_data.coordinates()
                    if coords is not None:
                        pts_inner = coords.astype(np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(h_mask, [pts_inner], 255)
                        neu_mask = cv2.subtract(outer, h_mask)
                        print(f"  ✓ Neural Inner ROI (manual) applied: {roi_inner_file}")
                    else:
                        print(f"  ⚠ No ROI coordinates found: {roi_inner_file}")
                        neu_mask = outer
                except Exception as e:
                    print(f"  ✗ Failed to load Neural ROI: {e}")
                    neu_mask = outer
            else:
                # Method 2: Automatic Hole Detection (Original Logic)
                holes = [(cv2.contourArea(cnts[i]), i) for i, hi in enumerate(hier[0]) if hi[3] == idx]
                if holes:
                    holes.sort(key=lambda x: x[0], reverse=True)
                    i_cnt = cnts[holes[0][1]].reshape(-1, 2)
                    M = cv2.moments(i_cnt)
                    if M['m00'] > 0:
                        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                        pts = smooth_polygon(get_n_gon_points(i_cnt, cx, cy, n_vertices), n_vertices, ((n_vertices-2)*180/n_vertices)-30)
                        h_mask = np.zeros((h, w), dtype=np.uint8); cv2.fillPoly(h_mask, [pts.reshape((-1, 1, 2))], 255)
                        neu_mask = cv2.subtract(outer, h_mask)
                    else: neu_mask = outer
                else: neu_mask = outer

    # B. Skeletal Mask (Polygon)
    ske_mask = np.zeros((h, w), dtype=np.uint8)
    if img_data['SKELETAL'] is not None:
        norm_s = cv2.normalize(img_data['SKELETAL'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, bin_s = cv2.threshold(norm_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dist_s = cv2.distanceTransform(cv2.bitwise_not(bin_s), cv2.DIST_L2, 5)
        mask_s = (dist_s < 25).astype(np.uint8) * 255
        cnts_s, _ = cv2.findContours(mask_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_s:
            main_s = max(cnts_s, key=cv2.contourArea)
            M = cv2.moments(main_s)
            if M['m00'] > 0:
                cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                s_pts = smooth_polygon(get_n_gon_points(main_s, cx, cy, n_vertices), n_vertices, ((n_vertices-2)*180/n_vertices)-30)
                cv2.fillPoly(ske_mask, [s_pts.reshape((-1, 1, 2))], 255)
            else: cv2.fillPoly(ske_mask, [main_s], 255)
    return ske_mask, neu_mask

# --- [4] Main Pipeline ---
def main():
    root = Tk(); root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Organoid Folder")
    if not folder_path: return
    save_dir = os.path.join(folder_path, "Comprehensive_Analysis_Report")
    os.makedirs(save_dir, exist_ok=True)
    
    # Priority Logic
    img_data = {'NEURAL': None, 'SKELETAL': None, 'DAPI': None}
    neural_keywords = ['NEUN', 'CH01', 'CH1', 'CH03', 'CH3', 'MAP2', 'NFM']
    skeletal_keywords = ['MYHC', 'MYH3', 'MYH', 'CH02', 'CH2']
    dapi_keywords = ['CH00', 'CH0', 'DAPI']
    
    all_files = os.listdir(folder_path)
    for target, kw_list in [('NEURAL', neural_keywords), ('SKELETAL', skeletal_keywords), ('DAPI', dapi_keywords)]:
        for k in kw_list:
            match = next((f for f in all_files if k.upper() in f.upper() and f.lower().endswith(('.tif', '.tiff'))), None)
            if match:
                raw = tiff.imread(os.path.join(folder_path, match))
                img_data[target] = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY) if len(raw.shape) == 3 else np.squeeze(raw)
                print(f"  ✓ {target} Channel Selected: {match}")
                break

    if img_data['DAPI'] is None: return
    h, w = img_data['DAPI'].shape[:2]
    ske_mask, neu_mask = generate_independent_masks(img_data, h, w, folder_path, all_files)

    # 1. Save individual masks (Unicode path compatible)
    for m, name, color in [(neu_mask, "NEURAL_Green", [0, 255, 0]), (ske_mask, "SKELETAL_Red", [0, 0, 255])]:
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        tmp[m > 0] = color # BGR order (Skeletal: Red=BGR[0,0,255])
        imwrite_unicode(os.path.join(save_dir, f"Individual_Mask_{name}.png"), tmp)

    # 2. Save individual figures (Fig 1, 2)
    ch_n, ch_s = get_display_image(img_data['NEURAL']), get_display_image(img_data['SKELETAL'])
    fig1 = np.zeros((h, w, 3), dtype=np.uint8)
    if ch_s is not None: fig1[:,:,2] = ch_s # R
    if ch_n is not None: fig1[:,:,1] = ch_n # G
    imwrite_unicode(os.path.join(save_dir, "Fig1_Raw_Merge.png"), fig1)

    ov_n, ov_s = np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w, 3), dtype=np.uint8)
    ov_n[neu_mask > 0] = [0, 255, 0]; ov_s[ske_mask > 0] = [0, 0, 255]
    fig2 = cv2.addWeighted(ov_n, 0.7, ov_s, 0.7, 0)
    imwrite_unicode(os.path.join(save_dir, "Fig2_Mask_Overlay.png"), fig2)

    # 3. Save quantitative data and Fig 3
    M_n = cv2.moments(neu_mask); cx, cy = (int(M_n['m10']/M_n['m00']), int(M_n['m01']/M_n['m00'])) if M_n['m00'] > 0 else (w//2, h//2)
    M_s = cv2.moments(ske_mask); angle = np.arctan2(M_s['m01']/M_s['m00'] - cy, M_s['m10']/M_s['m00'] - cx) if M_s['m00'] > 0 else 0
    max_r = np.max(np.sqrt((np.indices((h, w))[1]-cx)**2 + (np.indices((h, w))[0]-cy)**2)[neu_mask>0])
    
    l_r = np.linspace(-max_r, max_r, 256)
    csv_data = {'Relative_Radius': np.linspace(-1, 1, 256)}
    
    plt.figure(figsize=(8, 6), facecolor='white')
    for key, m, c in [('Neu', neu_mask, 'green'), ('Ske', ske_mask, 'red')]:
        vals = []
        for r in l_r:
            px, py = int(cx + r*np.cos(angle)), int(cy + r*np.sin(angle))
            if 0 <= py < h and 0 <= px < w:
                vals.append(100 if m[py, px] > 0 else 0)
            else: vals.append(0)
        csv_data[f'{key}_Occupancy'] = vals
        plt.plot(csv_data['Relative_Radius'], vals, color=c, lw=2.5, label=key)
    
    pd.DataFrame(csv_data).to_csv(os.path.join(save_dir, "Diameter_Profile_Data.csv"), index=False)
    plt.axvline(0, color='black', lw=1, ls='--'); plt.legend(); plt.title("Fig 3: Diameter Profile")
    plt.savefig(os.path.join(save_dir, "Fig3_Diameter_Profile_Plot.png"), dpi=300); plt.close()

    # 4. Integrated Report
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='white')
    axes[0].imshow(cv2.cvtColor(fig1, cv2.COLOR_BGR2RGB)); axes[0].axis('off'); axes[0].set_title("Figure 1: Raw Merge")
    axes[1].imshow(cv2.cvtColor(fig2, cv2.COLOR_BGR2RGB)); axes[1].axis('off'); axes[1].set_title("Figure 2: Mask Overlay")
    axes[2].plot(csv_data['Relative_Radius'], csv_data['Neu_Occupancy'], 'g-', lw=2, label='Neu')
    axes[2].plot(csv_data['Relative_Radius'], csv_data['Ske_Occupancy'], 'r-', lw=2, label='Ske')
    axes[2].axvline(0, color='black', lw=1, ls='--'); axes[2].legend(); axes[2].set_title("Figure 3: Diameter Profile")
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "Final_Integrated_Report.png"), dpi=300); plt.show()
    print(f"  ✓ All results saved successfully: {save_dir}")

if __name__ == "__main__":
    main()