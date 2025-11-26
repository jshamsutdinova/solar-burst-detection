from pathlib import Path
from random import randint

import numpy as np
import pandas as pd
import albumentations as A

import matplotlib.pyplot as plt


def generate_synthetic_bursts(metadata_csv, num_synthetic=10):
    """ Generate synthetic burst images from images with bursts. """
    synthetic_images = []
    
    metadata_df = pd.read_csv(metadata_csv)
    filtered_df = metadata_df[metadata_df['has_burst'] == True].reset_index(drop=True)
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.ElasticTransform(p=0.3),
        A.RandomGamma(p=0.3),
        # A.ShiftScaleRotate(
        #     shift_limit=0.1,
        #     scale_limit=0.2,
        #     rotate_limit=15,
        #     p=0.5
        # ),
    ])
    
    for _ in range(num_synthetic):
        random_idx = randint(0, len(filtered_df) - 1)
        base_image_name = filtered_df['filename'][random_idx]
        
        base_image = np.load(two_channel_dir / base_image_name)
        print(base_image.shape)
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

def generate_background(height=384, width=3000, pols=2):
    """ Generate a random background for two polarizations."""
    background = np.zeros((height, width, pols), dtype=np.float32)
    
    for pol in range(pols):
        # Random noise
        base_noise = np.random.normal(0, 0.1, (height, width)).astype(np.float32)
        
        # Если для двух поляризаций  создавать разные фоны, при создании двухканальных данных это не повлияет
        # Почему получился такой размер? Разве он не должне быть двухканальным? Две подяризации наложены друг на друга???
    pass

def generate_burst():
    pass


def plot_spec(images):    
    for img in images:
        plt.figure(figsize=(10,4))
        plt.imshow(img[:,:,0], cmap='gray', aspect='auto')
        plt.title('Synthetic Burst Image - H Spec')
        
    plt.show()
            
            
if __name__ == "__main__":
    metadata = 'data/temp/metadata.csv'
    two_channel_dir = Path('data/temp/two_channel/')
    synthetic_images = generate_synthetic_bursts(metadata)
    
    plot_spec(synthetic_images)


# def plot_real_spec(filtered_df):
#     for _, row in filtered_df.iterrows():
#         fname = row['filename']
#         fpath = two_channel_dir / fname
#         spec = np.load(fpath)
        
#         plt.figure(figsize=(10,4))
#         plt.imshow(spec[:,:,0], cmap='gray', aspect='auto')
#         plt.title(f"Real Burst Image - H Spec: {fname}")
#         plt.savefig(f"data/temp/img_with_bursts/{fname.replace('.npy', '.png')}", dpi=150)
#     print('Saved images with real bursts.')