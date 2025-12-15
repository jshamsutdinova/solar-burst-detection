from pathlib import Path
 
import numpy as np
import pandas as pd
import albumentations as A

import matplotlib.pyplot as plt


def generate_synthetic_bursts(metadata_csv, num_synthetic=10):
    """ Generate synthetic burst images from images with bursts. """
    synthetic_images = []
    metadata_df = pd.read_csv(metadata_csv)
    print(type(metadata_df))
    filtered_df = metadata_df[metadata_df['has_burst'] == True].reset_index(drop=True)
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.ElasticTransform(p=0.3),
        A.RandomGamma(p=0.3),
    ])
    
    for _ in range(num_synthetic):
        random_idx = np.random.randint(0, len(filtered_df) - 1)
        base_image_name = filtered_df['filename'][random_idx]
        
        base_image = np.load(two_channel_dir / base_image_name)
        augmented = transform(image=base_image)
        synthetic_image = augmented['image']
        synthetic_images.append(synthetic_image)
        
    return synthetic_images

def create_synthetic_burst(num_images=10):
    """ Create synthetic burst images from scratch."""
    synthetic_images = []
    
    for i in range(num_images):
        # Create a random noise image
        background = generate_background()
        spec_with_burst = generate_burst(background)
        
        save_syntetic_spec(i, spec_with_burst)         
        

def generate_background(height=384, width=3000):
    """ Generate a random background for the spectrum."""
    background = np.random.normal(0.5, 0.1, (height, width)).astype(np.float32)
    background = np.clip(background, 0, 1)
    
    # Add horizontal RFI lines
    for _ in range(np.random.randint(5, 12)):
        freq_channel = np.random.randint(0, height)
        thickness = np.random.randint(1, 3)
        intensity = np.random.uniform(0.15, 0.25)
        background[freq_channel:min(freq_channel+thickness, height), :] += intensity
    
    # Add vertical RFI lines
    for _ in range(np.random.randint(3, 6)):
        time_sample = np.random.randint(0, width)
        width_artifact = np.random.randint(1, 2)
        intensity = np.random.uniform(0.08, 0.15)
        start = max(0, time_sample - width_artifact)
        end = min(width, time_sample + width_artifact)
        background[:, start:end] += intensity
    
    # Add frequency gradient
    freq_gradient = np.linspace(0, 0.08, height).reshape(-1, 1)
    background += freq_gradient
    
    # find out is it usefull to use transforms images where position of burst doesn't change?
            
    return np.clip(background, 0, 1)

def generate_burst(background, min_duration=150, max_duration=400):
    """ Generate a synthetic burst and overly it on the background."""
    height, width = background.shape
    bboxes = []
    
    # Parameters of type III bursts.
    burst_duration = np.random.randint(min_duration, max_duration)
    burst_freq_band = np.random.randint(30, 100)
    start_time = np.random.randint(150, width - burst_duration - 150)
    start_freq = np.random.randint(80, height - burst_freq_band - 80)
    
    # Parameters of the drift (burst slop).
    drift_rate = np.random.uniform(-0.4, 0.4)
    drift_variation = np.random.uniform(0.05, 0.15)  # Change drift over time.
    
    # Parameters of curvature.
    curvature = np.random.uniform(-0.2, 0.2)
    wiggle_frequency = np.random.uniform(0.01, 0.05)  # дрожание
    wiggle_amplitude = np.random.uniform(0.5, 2.0)
    
    # Intensity changes.
    base_intensity = np.random.uniform(0.3, 0.7)
    intensity_variation = np.random.uniform(0.1, 0.3)
    
    min_freq = height
    max_freq = 0
    
    for t in range(burst_duration):
        drif = curvature * (t / burst_duration - 0.5) ** 2
        wiggle = wiggle_amplitude * np.sin(wiggle_frequency *t)
        current_drift = drift_rate + drift_variation * (t / burst_duration) +\
                        drif + wiggle
                
        current_freq_offset = int(drift_rate * t)
        current_start_freq = start_freq + current_freq_offset
        
        if current_start_freq < min_freq:
            min_freq = current_start_freq
        if current_start_freq + burst_freq_band > max_freq:
            max_freq = current_start_freq + burst_freq_band
        
        
        if current_start_freq < 0 or current_start_freq >= height:
            continue
        for f in range(burst_freq_band):
            freq_pos = current_start_freq + f
            time_pos = start_time + t
            
            if 0 <= freq_pos < height and 0 <= time_pos < width:
                # Intensity gradient
                time_factor = 1.0 - abs(t - burst_duration / 2) / (burst_duration / 2)
                freq_factor = 1.0 - abs(t - burst_freq_band / 2) / (burst_freq_band / 2)
                
                random_variation = 1.0 + intensity_variation * (np.random.random() - 0.5)
                intensity = time_factor * freq_factor * base_intensity * random_variation
                
                if np.random.random() > 0.3:
                    background[freq_pos, time_pos] = min(background[freq_pos, time_pos] + intensity, 1.0)
                
                # Add a burst
                # background[freq_pos, time_pos] = min(background[freq_pos, time_pos] + intensity, 1.0)
        
        #  Save to bboxes!!!       
    return background

def plot_spec(images):    
    for img in images:
        plt.figure(figsize=(10,4))
        plt.imshow(img, cmap='gray', aspect='auto')
        plt.title('Synthetic Burst Image - H Spec')
        
    plt.show()

def plot_real_spec(filtered_df):
    for _, row in filtered_df.iterrows():
        fname = row['filename']
        fpath = two_channel_dir / fname
        spec = np.load(fpath)
        
        plt.figure(figsize=(10,4))
        plt.imshow(spec, cmap='gray', aspect='auto')
        plt.title(f"Real Burst Image - Combined polarizations: {fname}")
        plt.savefig(f"data/temp/img_with_bursts/{fname.replace('.npy', '.png')}", dpi=150)
    print('Saved images with real bursts.')

def save_syntetic_spec(idx, synthetic_spec):
    plt.figure()
    plt.figure(figsize=(10,4))
    plt.imshow(synthetic_spec, cmap='gray', aspect='auto')
    plt.savefig(f'data/temp/synthetic_background/synthetic_{idx}.png', dpi=150)
            
            
if __name__ == "__main__":
    metadata = 'data/temp/metadata.csv'
    two_channel_dir = Path('data/temp/two_channel/')
    synthetic_images = generate_synthetic_bursts(metadata)
    
    # plot_spec(synthetic_images)
    create_synthetic_burst()
