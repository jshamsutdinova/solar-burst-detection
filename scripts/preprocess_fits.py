import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits


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

def save_as_png(save_path, data):
    """ Save the data as a PNG image. """
    img_data = (data * 255).astype(np.uint8)
    img_data = cv2.applyColorMap(img_data, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, img_data)
    
def generate_name(num_file, prefix, file_extension):
    """ Generate a file name using a pattern with leading zeros. """
    number = str(num_file).zfill(6)
    filename = f"{prefix}{number}{file_extension}"
    return filename

def add_metadata(file, output_csv):
    """ Write metadata of files into a CSV file."""
    pass

def prepare_dataset(fits_path):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
    print(hdul[0].header)
    h_spec = adaptive_normalization(data[:384, :])
    v_spec = adaptive_normalization(data[384:, :])
    
    two_channel_data = np.stack([h_spec, v_spec], axis=-1) # for training    
    three_channel_data = prepare_for_labeling(two_channel_data) # for labeling
    
    fname_two_ch = generate_name(idx_file, prefix='', file_extension='.npy')
    fname_three_ch = generate_name(idx_file, prefix='3ch_', file_extension='.png')

    np.save(f"data/temp/two_channel/{fname_two_ch}", two_channel_data)
    save_as_png(f"data/temp/three_channel/{fname_three_ch}", three_channel_data)
    

if __name__ == "__main__":
    base_dir = Path("/mnt/d/Work/NSSC/MUSER-L/data/5min_pngfits")
    csv_path = Path("data/temp/metadata.csv")
    
    if csv_path.is_file():
        df_meta = pd.read_csv(csv_path)
        print('here')
    else:
        columns = ['filename', 'has_burst', 'date', 'start_t' 'fits']
        df_meta = pd.DataFrame(columns=columns)

    dirs = [dir for dir in base_dir.iterdir() if dir.is_dir()]
    idx_file = 0
    for dir in dirs[:1]:
        files = list(dir.glob("*.fits"))
        print(f"Processing directory: {dir}, found {len(files)} fits files")

        for fits_file in files[:1]:
            if fits_file.stem in df_meta['fits'].values:
                print(f"File {fits_file} already processed, skipping.")
            else:
                prepare_dataset(fits_file)
                                     
            idx_file += 1
            print(f"Proccesed {fits_file}, total files: {idx_file}")
            