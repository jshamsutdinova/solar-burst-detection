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
    
def generate_name(num_file, file_extension):
    """ Generate a file name using a pattern with leading zeros. """
    number = str(num_file).zfill(6)
    filename = f"{number}{file_extension}"
    return filename

def read_MUSER_fits(fits_path):
    """ Read MUSER FITS files. Check problem with headers."""
    try:
        with fits.open(fits_path, verify='fix') as hdul:
            header = hdul[0].header
            data = hdul[0].data            
            return header, data

    except fits.VerifyError as e:
        print(f"Problem with header: {e}")
        with fits.open(fits_path, verify='fix') as hdul:
            header = hdul[0].header
            data = hdul[0].data
            return header, data

def prepare_dataset(fits_path, num_file):
    """ Save two and three channel data and return metadata record."""
    header, data = read_MUSER_fits(fits_path)

    h_spec = adaptive_normalization(data[:384, :])
    v_spec = adaptive_normalization(data[384:, :])
    
    two_channel_data = np.stack([h_spec, v_spec], axis=-1) # for training    
    three_channel_data = prepare_for_labeling(two_channel_data) # for labeling
    
    fname_two_ch = generate_name(num_file, file_extension='.npy')
    fname_three_ch = generate_name(num_file, file_extension='.png')
    
    new_record = {
        'filename': fname_two_ch,
        'has_burst': False,
        'date': header['DATE'],
        'start_t': str(fits_path.stem.split('_')[3]),
        'fits': fits_path.name
    }
    
    np.save(f"data/temp/two_channel/{fname_two_ch}", two_channel_data)
    save_as_png(f"data/temp/three_channel/{fname_three_ch}", three_channel_data)
    
    return new_record
    

if __name__ == "__main__":
    base_dir = Path("/mnt/d/Work/NSSC/MUSER-L/data/5min_pngfits")
    csv_path = Path("data/temp/metadata.csv")
    
    if csv_path.is_file():
        df_meta = pd.read_csv(csv_path)
        print('here')
    else:
        columns = ['filename', 'has_burst', 'date', 'start_t', 'fits']
        df_meta = pd.DataFrame(columns=columns)
        df_meta.to_csv(csv_path, index=False)

    dirs = [dir for dir in base_dir.iterdir() if dir.is_dir()]
    num_file = 1
    for dir in dirs:
        files = list(dir.glob("*.fits"))
        print(f"Processing directory: {dir}, found {len(files)} fits files")

        for fits_file in files:
            if fits_file.stem in df_meta['fits'].values:
                print(f"File {fits_file} already processed, skipping.")
            else:
                new_record = prepare_dataset(fits_file, num_file)
                
                new_df = pd.DataFrame([new_record])
                df_meta = pd.concat([df_meta, new_df], ignore_index=True)
                df_meta.to_csv(csv_path, index=False)
                                     
            num_file += 1
            print(f"Proccesed {fits_file}, total files: {num_file}")
            