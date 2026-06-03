import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import alexnet, AlexNet_Weights
from PIL import Image

VISION_DIR = os.path.dirname(os.path.abspath(__file__))

# Load pretrained AlexNet — no saved .pt file needed
weights    = AlexNet_Weights.IMAGENET1K_V1
categories = weights.meta['categories']   # list of 1000 ImageNet class names

model = alexnet(weights=weights)
model.eval()
print("Pretrained AlexNet loaded (ImageNet, 1000 classes)")

# AlexNet expects specific normalisation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")
model.to(device)

image_files = [
    os.path.join(VISION_DIR, 'img_00000.jpg'),
    os.path.join(VISION_DIR, 'IMG_9341.jpeg'),
    os.path.join(VISION_DIR, 'IMG_9342.jpeg'),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, img_path in zip(axes, image_files):
    img_pil = Image.open(img_path).convert('RGB')
    img_t   = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)                        # (1, 1000)
        probs  = logits.softmax(dim=1).squeeze(0)    # (1000,)

    # Top 5 predictions
    top5_probs, top5_idx = probs.topk(5)
    top5 = [(categories[i], p.item() * 100) for i, p in zip(top5_idx, top5_probs)]

    print(f"\n{os.path.basename(img_path)}")
    for name, pct in top5:
        print(f"  {name:<30} {pct:6.2f}%")

    ax.imshow(img_pil)
    ax.set_title(top5[0][0], fontsize=12)
    ax.axis('off')

    # Feature maps after all conv layers (first 32 of 256 channels)
    with torch.no_grad():
        feat_maps = model.features(img_t)            # (1, 256, H', W')
    feat = feat_maps.squeeze(0).cpu()

    fig_f, axes_f = plt.subplots(4, 8, figsize=(16, 8))
    fig_f.suptitle(f"{os.path.basename(img_path)} — pretrained AlexNet feature maps  |  top: {top5[0][0]}", fontsize=12)
    for i, fax in enumerate(axes_f.flat):
        fax.imshow(feat[i], cmap='viridis')
        fax.axis('off')
    fig_f.tight_layout()

plt.tight_layout()
plt.show()
