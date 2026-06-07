"""
Dataset utilities
=================
Provides:
  - RotatedMNIST  : MNIST with random rotations U(0, 2pi)
  - SRMnist       : MNIST with random rotation + random scale (Table 3)
  - get_loaders() : convenience factory

These match the evaluation datasets in Section 5.3 of the paper.
"""

import math
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import random


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class RandomRotationFull:
    """Rotate by a uniformly sampled angle in [0, 360)."""

    def __call__(self, img):
        angle = random.uniform(0, 360)
        return TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR,
                         fill=0)


class RandomIsotropicScale:
    """Scale by s ~ U(scale_min, scale_max) with centre-crop / pad."""

    def __init__(self, scale_min: float = 0.5, scale_max: float = 2.5,
                 output_size: int = 32):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.output_size = output_size

    def __call__(self, img):
        s = random.uniform(self.scale_min, self.scale_max)
        h, w = img.size[-1], img.size[-2] if hasattr(img, 'size') else \
               (img.shape[-2], img.shape[-1])
        # Use resize then centre-crop/pad
        new_h = int(round(self.output_size * s))
        new_w = new_h
        img = TF.resize(img, (new_h, new_w),
                        interpolation=TF.InterpolationMode.BILINEAR)
        # Pad or crop to output_size
        img = TF.center_crop(img, self.output_size) if \
              new_h >= self.output_size else \
              TF.pad(img, (self.output_size - new_w) // 2)
        return img


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class RotatedMNIST(Dataset):
    """
    MNIST with on-the-fly random full rotation.
    Matches the rotation setting in Table 4 and Section 5.4.
    """

    def __init__(self, root: str = './data', train: bool = True,
                 image_size: int = 32, download: bool = True):
        base_transform = transforms.Compose([
            transforms.Resize(image_size),
            RandomRotationFull(),
            transforms.ToTensor(),
        ])
        self.dataset = torchvision.datasets.MNIST(
            root=root, train=train, download=download,
            transform=base_transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class SRMnist(Dataset):
    """
    Roto-Scale MNIST: rotation ~ U(0, 2pi) + scale ~ U(0.5, 2.5).
    Used for autoencoding evaluation in Table 3 (SR-MNIST).
    """

    def __init__(self, root: str = './data', train: bool = True,
                 image_size: int = 32, download: bool = True):
        base_transform = transforms.Compose([
            transforms.Resize(image_size),
            RandomRotationFull(),
            RandomIsotropicScale(0.5, 2.5, image_size),
            transforms.ToTensor(),
        ])
        self.dataset = torchvision.datasets.MNIST(
            root=root, train=train, download=download,
            transform=base_transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class MQRTMnist(Dataset):
    """
    MQRT-MNIST: training set confined to rotations in [0, pi/4],
    test set spans all rotations (Section 5.2).
    """

    def __init__(self, root: str = './data', train: bool = True,
                 image_size: int = 32, download: bool = True):
        max_angle = 45.0 if train else 360.0
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Lambda(lambda img: TF.rotate(
                img, random.uniform(0, max_angle),
                interpolation=TF.InterpolationMode.BILINEAR, fill=0)),
            transforms.ToTensor(),
        ])
        self.dataset = torchvision.datasets.MNIST(
            root=root, train=train, download=download,
            transform=transform
        )
        # Keep only classes 3 and 4 as in the paper
        targets = self.dataset.targets
        mask = (targets == 3) | (targets == 4)
        self.dataset.data    = self.dataset.data[mask]
        self.dataset.targets = targets[mask]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_loaders(dataset: str = 'rotated_mnist',
                root: str = './data',
                image_size: int = 32,
                batch_size: int = 64,
                num_workers: int = 4,
                download: bool = True):
    """
    Returns (train_loader, test_loader).

    dataset options: 'rotated_mnist', 'sr_mnist', 'mqrt_mnist'
    """
    _map = {
        'rotated_mnist': RotatedMNIST,
        'sr_mnist':       SRMnist,
        'mqrt_mnist':     MQRTMnist,
    }
    assert dataset in _map, f"Unknown dataset {dataset}. Choose from {list(_map)}"
    cls = _map[dataset]

    train_ds = cls(root=root, train=True,
                   image_size=image_size, download=download)
    test_ds  = cls(root=root, train=False,
                   image_size=image_size, download=download)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, test_loader
