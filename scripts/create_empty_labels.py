from pathlib import Path


def create_empty_label_files(save_dir):
    """ Create empty label txt files for images without bursts. """
    image_dir = Path('data/temp/three_channel')
    label_dir = Path('data/temp/labels')
    empty_label_dir = Path(save_dir)
    
    images = image_dir.glob('*.png')
    for img in images:
        label_file = label_dir / f"{img.stem}.txt"
        if not label_file.exists():
            empty_label_file = empty_label_dir / f"{img.stem}.txt"
            empty_label_file.touch()
            print(f"Created empty label file: {empty_label_file}")

create_empty_label_files('data/temp/empty_labels')