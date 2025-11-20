from pathlib import Path
import pandas as pd


def add_metadata_label(csv_path: Path, labels_dir: Path):
    """ This method updates the csv file with metadata labels. """
    df = pd.read_csv(csv_path)
    
    for idx, row in df.iterrows():
        fname = Path(row['filename'])
        label_path = labels_dir / f"{fname.stem}.txt"
        
        if label_path.exists():
            with open(label_path, 'r') as ff:
                content = ff.read().strip()
                if content:
                    print('Write label for', fname)
                    df.at[idx, 'has_burst'] = True
        
    new_csv_name = csv_path.parent / f"{csv_path.stem}_update.csv"
    df.to_csv(new_csv_name, index=False)


csv_path = Path('data/temp/metadata.csv')
label_dir = Path('data/temp/labels')

add_metadata_label(csv_path, label_dir)
