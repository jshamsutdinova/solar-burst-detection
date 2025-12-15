from pathlib import Path
import matplotlib.pyplot as plt

from burst_cut_paste_generator import BurstCutPasteGenerator


generator = BurstCutPasteGenerator(
    spec_dir = Path('/mnt/d/Work/NSSC/type_III_detection/two_channel'),
    label_dir = Path('data/temp/labels'),
    metadata_csv = Path('data/temp/metadata.csv')  
)

cutpaste_dataset = generator.generate_dataset(
    num_samples=10,
    height=384,
    width=3000
)


def visualize_cutpaste_results(dataset, num_samples=4):
    """
    Визуализирует результаты cut-paste
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, sample in enumerate(dataset[:num_samples]):
        spectrogram = sample['spectrum']
        bboxes = sample['bboxes']
        
        # Отображаем спектрограмму
        im = axes[idx].imshow(spectrogram[::-1, :], aspect='auto', cmap='gray', 
                             origin='lower', vmin=0, vmax=1)
        axes[idx].set_title(f'Cut-Paste sample #{idx} ({len(bboxes)} bursts)')
        
        # Рисуем bounding boxes
        for bbox in bboxes:
            class_id, x_center, y_center, width, height = bbox
            x1 = (x_center - width/2) * spectrogram.shape[1]
            y1 = (y_center - height/2) * spectrogram.shape[0]
            # rect = plt.Rectangle((x1, y1), width * spectrogram.shape[1], 
            #                    height * spectrogram.shape[0],
            #                    linewidth=2, edgecolor='cyan', facecolor='none')
            # axes[idx].add_patch(rect)
        
        axes[idx].set_xlabel('Time')
        axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Визуализация
visualize_cutpaste_results(cutpaste_dataset, 4)