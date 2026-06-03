import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from visionTrainer import VisionDataset, IMAGE_W, IMAGE_H

VISION_DIR = os.path.dirname(os.path.abspath(__file__))


class ImageNetV2(nn.Module):
    def __init__(self, num_classes, freeze_features=True):
        super().__init__()

        alexnet        = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.features  = alexnet.features  # pretrained, output: (B, 256, H', W')
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if freeze_features:
            for p in self.features.parameters():
                p.requires_grad = False

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(256, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
            nn.LogSoftmax(dim=1),
        )

        self.regressor = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Sigmoid(),  # output in [0,1] — multiply by IMAGE_W/H to get pixels
        )

    def freeze_features(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_features(self):
        for p in self.features.parameters():
            p.requires_grad = True

    def forward(self, x):
        x    = self.normalize(x)              # apply ImageNet normalisation in-model
        x    = self.features(x)
        flat = torch.flatten(self.gap(x), 1)  # (B, 256)
        return self.classifier(flat), self.regressor(flat)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (no GPU available)")

    dataset    = VisionDataset()
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    print(f"Loaded {len(dataset)} samples — train: {train_size}, val: {val_size}")

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=16)

    model       = ImageNetV2(num_classes=4, freeze_features=True).to(device)
    cls_loss_fn = nn.NLLLoss()
    reg_loss_fn = nn.MSELoss()
    lambda_reg  = 1.0

    def train_epoch(dataloader, optimizer):
        model.train()
        total_loss = 0
        for batch, (x, labels, coords) in enumerate(dataloader):
            print(f"  image {batch * dataloader.batch_size:>4d} / {len(dataloader.dataset)}", end="\r")
            x, labels, coords = x.to(device), labels.to(device), coords.to(device)
            cls_pred, reg_pred = model(x)
            loss = cls_loss_fn(cls_pred, labels) + lambda_reg * reg_loss_fn(reg_pred, coords)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print()
        return total_loss / len(dataloader)

    def validate(dataloader):
        model.eval()
        total_cls, total_reg, correct, coord_mae = 0, 0, 0, 0
        with torch.no_grad():
            for x, labels, coords in dataloader:
                x, labels, coords = x.to(device), labels.to(device), coords.to(device)
                cls_pred, reg_pred = model(x)
                total_cls += cls_loss_fn(cls_pred, labels).item()
                total_reg += reg_loss_fn(reg_pred, coords).item()
                correct   += (cls_pred.argmax(1) == labels).sum().item()
                reg_px    = reg_pred * torch.tensor([IMAGE_W, IMAGE_H], device=device)
                coords_px = coords   * torch.tensor([IMAGE_W, IMAGE_H], device=device)
                coord_mae += (reg_px - coords_px).abs().mean().item()
        n = len(dataloader)
        return total_cls / n, total_reg / n, correct / len(dataloader.dataset), coord_mae / n

    history = {"train_loss": [], "val_cls_loss": [], "val_reg_loss": [], "val_acc": [], "val_mae_px": []}

    def run_phase(epochs, lr, label):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
        print(f"\n=== {label} (lr={lr}, {epochs} epochs) ===")
        for epoch in range(1, epochs + 1):
            train_loss                         = train_epoch(train_loader, optimizer)
            val_cls, val_reg, val_acc, val_mae = validate(val_loader)
            history["train_loss"].append(train_loss)
            history["val_cls_loss"].append(val_cls)
            history["val_reg_loss"].append(val_reg)
            history["val_acc"].append(val_acc)
            history["val_mae_px"].append(val_mae)
            print(f"  Epoch {epoch:>2}/{epochs}  train: {train_loss:.4f}  "
                  f"cls: {val_cls:.4f}  reg: {val_reg:.4f}  acc: {val_acc:.1%}  mae: {val_mae:.1f}px")

    # Phase 1 — backbone frozen, only heads learn
    run_phase(epochs=20, lr=1e-4, label="Phase 1 — heads only (features frozen)")

    # Phase 2 — unfreeze backbone, fine-tune end-to-end at lower lr
    model.unfreeze_features()
    run_phase(epochs=30, lr=1e-5, label="Phase 2 — full fine-tune")

    torch.save(model, os.path.join(VISION_DIR, "vision_model_V2.pt"))
    print("\nModel saved to vision_model_V2.pt")

    # Plot training curves with phase boundary marker
    ep         = range(1, len(history["train_loss"]) + 1)
    phase1_end = 20

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    for ax in (ax1, ax2, ax3):
        ax.axvline(phase1_end, color='gray', linestyle='--', linewidth=1, label='unfreeze')

    ax1.plot(ep, history["train_loss"],   label="train loss")
    ax1.plot(ep, history["val_cls_loss"], label="val cls")
    ax1.plot(ep, history["val_reg_loss"], label="val reg")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Losses"); ax1.legend()

    ax2.plot(ep, [a * 100 for a in history["val_acc"]], color="green")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Classification Accuracy"); ax2.legend()

    ax3.plot(ep, history["val_mae_px"], color="orange")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("MAE (pixels)")
    ax3.set_title("Coordinate Regression MAE"); ax3.legend()

    plt.tight_layout()
    plt.show()
