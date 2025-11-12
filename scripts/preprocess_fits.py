import numpy as np
import cv2
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'notebook', 'grid'])


def adaptive_normalization(data):
    """ Apply adaptive normalization to the data. """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    clipped = np.clip(data, median - 2*mad, median + 8*mad)
    norm_data = (clipped - clipped.min()) / (clipped.max() - clipped.min())
    
    return norm_data

def prepare_for_labeling(two_channel_data):
    """ Convert two-channel data to three-channel for convinient labeling."""
    h_channel = two_channel_data[:, :, 0]
    v_channel = two_channel_data[:, :, 1]
    third_channel = (h_channel + v_channel) / 2
    
    three_channel_data = np.stack([h_channel, v_channel, third_channel], axis=-1)
    
    return three_channel_data

def save_as_png(data, save_path):
    """ Save the data as a PNG image. """
    img_data = (data * 255).astype(np.uint8)
    cv2.imwrite(save_path, img_data)
    

def prepare_dataset(fits_path):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
    
    h_spec = adaptive_normalization(data[:384, :])
    v_spec = adaptive_normalization(data[384:, :])
    
    # for training
    two_channel_data = np.stack([h_spec, v_spec], axis=-1)    
    
    # for labeling
    three_channel_data = prepare_for_labeling(two_channel_data)
    
    np.save(f"data/temp/two_channel/two_channel_{fits_path.stem}.npy", two_channel_data)
    save_as_png(three_channel_data, f"data/temp/three_channel/three_channel_{fits_path.stem}.png")
    
    
    # fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12, 10))
    
    
    # for ax, method in zip(axes.flat, methods):
    #     if method == 'percentile_rgb' or method == 'percentile':
    #         p_low, p_high = np.percentile(data, [5, 95])
    #         clipped = np.clip(data, p_low, p_high)
    #         norm_data = (clipped - p_low) / (p_high - p_low)
    #     elif method == 'adaptive_rgb' or method == 'adaptive':
    #         median = np.median(data)
    #         mad = np.median(np.abs(data - median))
    #         clipped = np.clip(data, median - 2*mad, median + 8*mad)
    #         norm_data = (clipped - clipped.min()) / (clipped.max() - clipped.min())
        
        # img_data = (norm_data * 255).astype(np.uint8)
        # if method == 'percentile_rgb' or method == 'adaptive_rgb':
        #     ax.imshow(img_data, cmap='jet', aspect='auto')
        # elif method == 'percentile' or method == 'adaptive':
        #     ax.imshow(img_data, cmap='gray', aspect='auto')
        # ax.set_title(f'Method: {method}')
        # ax.axis('off')
    
    # plt.tight_layout()
    # plt.savefig(f'data/images/temp/normalization_comparison/{fits_path.stem}.png', dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    base_dir = Path("/mnt/d/Work/NSSC/MUSER-L/data/5min_pngfits")
    dir_to_save = Path("data/images/temp/normalization_comparison")
    
    dirs = [dir for dir in base_dir.iterdir() if dir.is_dir()]
    
    for dir in dirs[:1]:
        files = list(dir.glob("*.fits"))
        print(f"Processing directory: {dir}, found {len(files)} fits files")

        for fits_file in files:
            prepare_dataset(idx, fits_file)
            # file_exist = list(dir_to_save.glob(f'{fits_file.stem}*.png'))        
            # if not file_exist:
                # compare_normalization_methods(fits_file)
            