from fileinput import filename
import torch
import sys
import random
import os
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.io import decode_image
import numpy as np
from tempfile import TemporaryFile
outfile = TemporaryFile()
from torch_train_test import Dataset
from torch_train_test import Stair_UNet
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from camera import Picamera2

CAMERA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'camera')


class ImageClass:
    def __init__(self,rawImage):
        self.rawImage = rawImage
        self.width = self.rawImage.width
        self.height = self.rawImage.height
        self.imageArray = np.array(Image.open(os.path.join(CAMERA_DIR, 'test.jpg')))
    
    
