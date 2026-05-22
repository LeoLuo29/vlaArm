import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from visionTrainer import ImageNet

VISION_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(VISION_DIR, 'vision_model.pt')

CLASS_NAMES = {0: "torus", 1: "cube", 2: "cylinder", 3: "cone", 4: "sphere"}

transform = transforms.Compose([
    transforms.ToTensor(),  # (H,W,3) uint8 → (3,H,W) float32 in [0,1]
])

image_files = [
    os.path.join(VISION_DIR, 'img_00000.jpg'),
    os.path.join(VISION_DIR, 'img_00038.jpg'),
    os.path.join(VISION_DIR, 'IMG_9342.jpeg'),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()
print(f"Model loaded from {MODEL_PATH}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, img_path in zip(axes, image_files):
    img_pil = Image.open(img_path).convert('RGB')
    img_t   = transform(img_pil).unsqueeze(0).to(device)  # (1, 3, H, W)

    with torch.no_grad():
        feat_maps = model.features(img_t)                    # (1, 256, H, W)
        output    = model.classifier(torch.flatten(model.gap(feat_maps), 1))
        probs     = output.exp().squeeze(0)                  # log-softmax → probabilities
        class_idx = probs.argmax().item()
        category  = CLASS_NAMES[class_idx]

    print(f"{os.path.basename(img_path)} → {category}")
    for idx, name in CLASS_NAMES.items():
        print(f"  {name:<10} {probs[idx].item() * 100:6.2f}%")

    ax.imshow(img_pil)
    ax.set_title(category, fontsize=16)
    ax.axis('off')

    # Plot first 32 feature channels (4×8 grid) after all conv layers
    feat = feat_maps.squeeze(0).cpu()  # (256, H, W)
    fig_f, axes_f = plt.subplots(4, 8, figsize=(16, 8))
    fig_f.suptitle(f"{os.path.basename(img_path)} — feature maps after conv layers ({category})", fontsize=13)
    for i, fax in enumerate(axes_f.flat):
        fax.imshow(feat[i], cmap='viridis')
        fax.axis('off')
    fig_f.tight_layout()

plt.tight_layout()
plt.show()
