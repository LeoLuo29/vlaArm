import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import matplotlib.pyplot as plt
from cameraImageLoader import ImageClass, CAMERA_DIR
from PIL import Image


cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)
print("GPU Device Name:", torch.cuda.get_device_name(0) if cuda_available else "No GPU (CPU only)")
print(torch.__version__)


myImage = ImageClass(Image.open(os.path.join(CAMERA_DIR, 'test.jpg')))

plt.imshow(myImage.imageArray)
plt.show()
