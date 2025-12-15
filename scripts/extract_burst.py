""" This code cut real solar radio bursts type III for using for CUT-PAST augumentation. """
from pathlib import Path

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt


def get_metadata_bursts(metadata_csv: Path) -> pd.DataFrame:
    """ Get metadata of bursts from CSV file. """    
    metadata_df = pd.read_csv(metadata_csv)
    burst_metadata = metadata_df[metadata_df['has_burst'] == True].reset_index(drop=True)
    return burst_metadata


def extract_radio_bursts(metadata_csv: Path, spec_burst_dir: Path, label_dir: Path) -> list:
    """ Cut bursts with background. """
    extracted_bursts = []
    burst_metadata = get_metadata_bursts(metadata_csv)
    
    # Get bursts from real data.
    for fname in burst_metadata['filename']:
        spec_path = spec_burst_dir / fname
        label_path = (label_dir / fname).with_suffix('.txt')                 

        spec = np.load(spec_path)
        spec_h, spec_w = spec.shape
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)
                
                # Convert to pixel coordinates.
                x_center_px = int(x_center * spec_w)
                y_center_px = int(y_center * spec_h)
                width_px = int(width * spec_w)
                height_px = int(height* spec_h)
                
                # Define bounding box.
                x1 = max(0, x_center_px - width_px)
                y1 = max(0, y_center_px - int(height_px))
                # y1 = max(0, y_center_px - int(height_px * 1.5))
                x2 = min(spec_w, x_center_px + width_px)
                y2 = min(spec_h, y_center_px + int(height_px))
                # y2 = min(spec_h, y_center_px + int(height_px * 1.2))
                
                # Check the size of bbox.
                burst_h = y2 - y1
                burst_w = x2 - x1
                
                if burst_h > 10 and burst_w > 10:
                    burst = spec[y1:y2, x1:x2].copy()
                    
                    extracted_bursts.append({
                                    'burst': burst,
                                    'original_bbox': (x1, y1, x2, y2),
                                    'original_size': (spec_h, spec_w),
                                    'source_file': fname,
                                    'burst_idx': line_idx,
                                    'shape': burst.shape
                                })
                    
                
    print(f"Extracted {len(extracted_bursts)} bursts.")
    
    return extracted_bursts


def transform_radio_burst(burst: np.ndarray) -> np.ndarray:
    """ Apply transformations to the burst. """
    transformed = burst.copy()
    h, w = transformed.shape
    
    # Vertical stretching / compression (imitating frequency band).
    v_scale = np.random.uniform(0.8, 1.2)
    if v_scale != 1.0:
        new_h = int(h * v_scale)
        transformed = cv2.resize(transformed, (w, new_h), interpolation=cv2.INTER_LINEAR)
        
        if new_h > h:
            start = (new_h - h) // 2
            transformed = transformed[start:start+h, :]
        else:
            pad_top = (h - new_h) // 2
            pad_bottom = h - new_h - pad_top
            transformed = np.pad(transformed, ((pad_top, pad_bottom), (0, 0)), mode='edge')
    
    # Horizontal stretching / compression (imitating time duration).
    h_scale = np.random.uniform(0.9, 1.1)
    if h_scale != 1.0:
        new_w = int(w * h_scale)
        transformed = cv2.resize(transformed, (new_w, h), interpolation=cv2.INTER_LINEAR)
        
        # Приводим к исходному размеру
        if new_w > w:
            start = (new_w - w) // 2
            transformed = transformed[:, start:start+w]
        else:
            pad_left = (w - new_w) // 2
            pad_right = w - new_w - pad_left
            transformed = np.pad(transformed, ((0, 0), (pad_left, pad_right)), mode='edge')
    
    # Add slop.
    if np.random.random() > 0.7:
        shear_factor = np.random.uniform(-0.1, 0.1)
        M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        transformed = cv2.warpAffine(transformed, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Change contrast.
    contrast = np.random.uniform(0.9, 1.1)
    transformed = np.clip(0.5 + contrast * (transformed - 0.5), 0, 1)
            
    return transformed

def blur_burst_edges(result, bbox_px, kernel_size=5):
    """ Smooth edges of the pasted burst."""
    x1, y1, x2, y2 = bbox_px
    
    mask = np.zeros(result.shape, dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    
    mask_blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigmaX=2)
    blended = mask_blurred * result + (1 - mask_blurred) * result
    
    return blended

def paste_radio_burst(background: np.ndarray, burst: np.array, paste_position: tuple, match_background=True) -> tuple:
    """ Paste radio burst onto background at specific position."""
    bg_h, bg_w = background.shape
    burst_h, burst_w = burst.shape
    paste_y, paste_x = paste_position
    
    if paste_x < 0 or paste_y < 0 or paste_x + burst_w > bg_w or paste_y + burst_h > bg_h:
        paste_x = max(0, min(paste_x, bg_w - burst_w))
        paste_y = max(0, min(paste_y, bg_h - burst_h))
    
    result = background.copy()
    
    bg_region = background[paste_y:paste_y+burst_h, paste_x:paste_x+burst_w]
    
    # burst_mask = burst > np.percentile(burst, 40)  # threshold mask.
    # alpha_map = np.zeros_like(burst)
    
    # for i in range(burst_h):
    #     for j in range(burst_w):
    #         if burst_mask[i, j]:
    #             burst_intensity = burst[i, j]
                
    #             bg_intensity = bg_region[i, j]
    #             contrast = abs(burst_intensity - bg_intensity)
                
    #             alpha = 0.3 + 0.7 * min(contrast / 0.5, 1.0)
    #             alpha_map[i, j] = alpha
    # blended = alpha_map * burst +(1 - alpha_map) * bg_region
    
    # blended = np.clip(bg_region + burst * 0.7, 0, 1)
    
    alpha = 0.4
    blended = alpha * burst + (1 - alpha) * bg_region
    
    if match_background:
        # Выравниваем среднюю яркость с фоном
        burst_mean = np.mean(burst[burst > 0.1]) if np.any(burst > 0.1) else 0.5
        bg_mean = np.mean(bg_region)
        if burst_mean > 0:
            brightness_factor = bg_mean / burst_mean * 0.8
            blended = np.clip(blended * brightness_factor, 0, 1)
    
    result[paste_y:paste_y+burst_h, paste_x:paste_x+burst_w] = blended
    
    # Bbox for YOLO.
    bbox_x_center = (paste_x + burst_w/2) / bg_w
    bbox_y_center = (paste_y + burst_h/2) / bg_h
    bbox_width = burst_w / bg_w
    bbox_height = burst_h / bg_h
    
    result = blur_burst_edges(result, (paste_x, paste_y, paste_x + burst_w, paste_y + burst_h))
    
    bbox = [0, bbox_x_center, bbox_y_center, bbox_width, bbox_height]
    
    return result, bbox, (paste_x, paste_y, paste_x + burst_w, paste_y + burst_h)
    

if __name__ == "__main__":
    metadata = Path('data/temp/metadata.csv')
    spec_burst_dir = Path('/mnt/d/Work/NSSC/type_III_detection/two_channel')
    label_dir = Path('data/temp/labels')
    
    extracted_bursts = extract_radio_bursts(metadata, spec_burst_dir, label_dir)
    
    for burst in extracted_bursts[:2]:
        transform_radio_burst(burst['burst'])
            
    
    
    
    
