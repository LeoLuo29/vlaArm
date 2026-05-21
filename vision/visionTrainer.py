import os
import torch
from torch import nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


DATASETS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataSets')
IMAGES_DIR    = os.path.join(DATASETS_DIR, 'images')
LABELS_FILE   = os.path.join(DATASETS_DIR, 'labels.txt')
SHAPE_CLASSES = {"torus": 0, "cube": 1, "cylinder": 2, "cone": 3, "sphere": 4}

transform = transforms.Compose([
    transforms.ToTensor(),  # (H,W,3) uint8 → (3,H,W) float32 in [0,1]
])


class VisionSample:
    """Used for visualization only — not for training."""
    def __init__(self, filename, shape, color, x, y):
        img_path    = os.path.join(IMAGES_DIR, filename)
        self.pixels = np.array(Image.open(img_path).convert('RGB'))  # (H, W, 3) uint8
        self.shape  = shape
        self.color  = color
        self.x      = x
        self.y      = y

    def __repr__(self):
        h, w, _ = self.pixels.shape
        return f"VisionSample(shape={self.shape!r}, color={self.color!r}, x={self.x}, y={self.y}, image={w}x{h})"


class VisionDataset(Dataset):
    def __init__(self, labels_file=LABELS_FILE):
        self.samples = []
        with open(labels_file, 'r') as f:
            lines = f.readlines()
        for line in lines[2:]:  # skip 2-line header
            line = line.strip()
            if not line:
                continue
            _, filename, shape, color, x, y = line.split()
            self.samples.append((filename, shape, color, float(x), float(y)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, shape, *_ = self.samples[idx]
        img   = Image.open(os.path.join(IMAGES_DIR, filename)).convert('RGB')
        img_t = transform(img)                  # (3, 360, 640) float32
        label = SHAPE_CLASSES[shape]            # int 0–3
        return img_t, label

    def get_sample(self, idx):
        """Returns a VisionSample for visualization."""
        filename, shape, color, x, y = self.samples[idx]
        return VisionSample(filename, shape, color, x, y)


class ImageNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: 640×360×3 → 158×88×96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            # MaxPool1: 158×88×96 → 78×43×96
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv2: 78×43×96 → 78×43×256
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # MaxPool2: 78×43×256 → 38×21×256
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv3: 38×21×256 → 38×21×384
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Conv4: 38×21×384 → 38×21×384
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Conv5: 38×21×384 → 38×21×256
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # → 1×1×256

        self.classifier = nn.Sequential(
            nn.Linear(256, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
            nn.LogSoftmax(dim=1),  # use with NLLLoss
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (no GPU available)")

    ## load dataset and split 80/20 train/val
    dataset    = VisionDataset()
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    print(f"Loaded {len(dataset)} samples — train: {train_size}, val: {val_size}")

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=16)

    model     = ImageNet(num_classes=5).to(device)
    loss_fn   = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def train(dataloader):
        model.train()
        total_loss = 0
        for batch, (x, y) in enumerate(dataloader):
            current = batch * dataloader.batch_size
            print(f"  image {current:>4d} / {len(dataloader.dataset)}", end="\r")
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print()
        return total_loss / len(dataloader)

    def validate(dataloader):
        model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                total_loss += loss_fn(pred, y).item()
                correct    += (pred.argmax(1) == y).sum().item()
        return total_loss / len(dataloader), correct / len(dataloader.dataset)

    import matplotlib.pyplot as plt

    epochs = 20
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss        = train(train_loader)
        val_loss, val_acc = validate(val_loader)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch:>2}/{epochs}  train loss: {train_loss:.4f}  val loss: {val_loss:.4f}  val acc: {val_acc:.1%}")

    torch.save(model, "vision/vision_model.pt")
    print("Model saved to vision/vision_model.pt")

    ## plot training curves
    ep = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(ep, history["train_loss"], label="train loss")
    ax1.plot(ep, history["val_loss"],   label="val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train vs Val Loss")
    ax1.legend()

    ax2.plot(ep, [a * 100 for a in history["val_acc"]], color="green", label="val acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.show()
