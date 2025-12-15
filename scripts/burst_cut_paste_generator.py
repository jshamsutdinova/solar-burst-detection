import numpy as np
from pathlib import Path

from extract_burst import *


class BurstCutPasteGenerator:
    """
    Generator for cut-and-paste augmentation of radio bursts.
    """
    def __init__(self, spec_dir, label_dir, metadata_csv):
        self.real_bursts = extract_radio_bursts(metadata_csv, spec_dir, label_dir)
        
    def generate_synthetic_sample(self, height=384, width=3000, min_bursts=1, max_bursts=2):
        """ Genereate a synthetic spectrum. """
        background = self._generate_background(height, width)
        num_bursts = np.random.randint(min_bursts, max_bursts + 1)
        
        all_bboxes = []
        burst_positions = []
        
        for _ in range(num_bursts):
            if not self.real_bursts:
                break
            
            burst_data = self.real_bursts[np.random.randint(len(self.real_bursts))]
            burst = burst_data['burst'].copy()
            transform_burst = transform_radio_burst(burst)
            
            paste_position = self._choose_burst_position(burst.shape, height, width, burst_positions)
            
            if paste_position:
                background, bbox, bbox_px = paste_radio_burst(background, burst, paste_position)
                all_bboxes.append(bbox)
                burst_positions.append(bbox_px)
        
        background = np.clip(background, 0, 1)
        
        return {
            'spectrum': background,
            'bboxes': all_bboxes
        }
            
        
    def _generate_background(self, height, width):
        """ Generate a random background for the spectrum."""
        background = np.random.normal(0.5, 0.15, (height, width)).astype(np.float32)
        
        # Add horizontal RFI lines.
        for _ in range(np.random.randint(3, 8)):
            freq = np.random.randint(0, height)
            thickness = np.random.randint(1, 3)
            intensity = np.random.uniform(0.1, 0.2)
            background[freq:min(freq+thickness, height), :] += intensity
        
        # Add noise spikes.
        # for _ in range(np.random.randint(10, 30)):
        #     y, x = np.random.randint(0, height), np.random.randint(0, width)
        #     background[y:y+2, x:x+2] += np.random.uniform(0.3, 0.6)
        
        # freq_gradient = np.linspace(-0.05, 0.05, height).reshape(-1, 1)
        # background += freq_gradient
        
        return np.clip(background, 0, 1)

    def _choose_burst_position(self, burst_shape, height, width, existing_positions,
                               min_distance=20):
        """ Choose a position to paste the burst avoiding overlaps."""
        burst_h, burst_w = burst_shape
        max_attempts = 20
        
        for attempt in range(max_attempts):
            # Random position.
            # paste_y = np.random.randint(50, height - burst_h - 50)
            paste_x = np.random.randint(100, width - burst_w - 100)
            
            if height - burst_h - 50 > 50:
                paste_y = np.random.randint(50, height - burst_h - 50)
            else:
                paste_y = max(0, (height - burst_h) // 2)  # center
            
            # Check for burst overlap.
            too_close = False
            for (x1, y1, x2, y2) in existing_positions:
                if not (paste_x + burst_w < x1 - min_distance or 
                        paste_x > x2 + min_distance or
                        paste_y + burst_h < y1 - min_distance or 
                        paste_y > y2 + min_distance):
                    too_close = True
                    break
            if not too_close:
                return (paste_y, paste_x)
            
        # paste_y = np.random.randint(50, height - burst_h - 50)
        paste_x = np.random.randint(100, width - burst_w - 100)
        
        low = 50
        high = height - burst_h - 50
        if high > low:
            paste_y = np.random.randint(low, high)
        else:
            paste_y = np.random.randint(0, max(1, height - burst_h))
        
        return (paste_y, paste_x)
        
        
    
    def generate_dataset(self, num_samples=10, height=384, width=3000):
        """ Generate a dataset of synthetic spectra with bursts. """
        synthetic_dataset = []
        
        print(f'Starting generation of {num_samples} synthetic samples.')
        
        for i in range(num_samples):
            sample = self.generate_synthetic_sample(height, width)
            synthetic_dataset.append(sample)
            
            if (i + 1) % 50 == 0:
                print(f'Generated {i + 1} / {num_samples} samples.')
        print('Generation completed.')
        
        return synthetic_dataset
    