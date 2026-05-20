import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataSets')
IMAGES_DIR   = os.path.join(DATASETS_DIR, 'images')
LABELS_FILE  = os.path.join(DATASETS_DIR, 'labels.txt')


class VisionSample:
    def __init__(self, filename, shape, color, x, y):
        img_path        = os.path.join(IMAGES_DIR, filename)
        self.pixels     = np.array(Image.open(img_path).convert('RGB'))  # (H, W, 3) uint8
        self.shape      = shape    # string e.g. "torus"
        self.color      = color    # string e.g. "yellow"
        self.x          = x        # float, pixel x coordinate
        self.y          = y        # float, pixel y coordinate

    def __repr__(self):
        h, w, _ = self.pixels.shape
        return f"VisionSample(shape={self.shape!r}, color={self.color!r}, x={self.x}, y={self.y}, image={w}x{h})"


class VisionDataset(Dataset):
    def __init__(self, labels_file=LABELS_FILE):
        self.samples = []
        with open(labels_file, 'r') as f:
            lines = f.readlines()
        # skip the 2-line header (column names + separator)
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # index, filename, shape, color, x, y
            _, filename, shape, color, x, y = parts
            self.samples.append((filename, shape, color, float(x), float(y)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, shape, color, x, y = self.samples[idx]
        return VisionSample(filename, shape, color, x, y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = VisionDataset()
    print(f"Loaded {len(dataset)} samples")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, ax in enumerate(axes):
        sample = dataset[i]
        ax.imshow(sample.pixels)
        ax.set_title(f"{sample.shape} ({sample.color})\nx={sample.x}, y={sample.y}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
