import numpy as np

import torch
from torchvision.transforms import v2

# Base transform, works for MNIST and CIFAR10 (not for CELEBA, images need to be reshaped to 64x64)
default_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

celeba_64_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((64, 64), antialias=True),
    v2.ToDtype(torch.float32, scale=True)
])

celeba_64_untransform = v2.Compose([
    v2.Resize((218, 178), antialias=True),
])

celeba_128_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((128, 128), antialias=True),
    v2.ToDtype(torch.float32, scale=True)
])

celeba_128_untransform = v2.Compose([
    v2.Resize((218, 178), antialias=True),
])
