"""RealObject: real photos organized as one folder per object class.

Layout::

    <data_root>/RealObject/
        <class_a>/xxx.jpg
        <class_b>/yyy.png
"""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from counthallu.datasets.common import IMG_EXTENSIONS
from counthallu.utils import ResizeWithPadding


class RealObject(Dataset):
    SUBDIR = "RealObject"

    def __init__(self, img_size=256, mode="train", hub_dataset=None, data_root=None):
        if data_root is None:
            raise ValueError("RealObject requires a local `data_root`.")
        self.data_path = os.path.join(data_root, self.SUBDIR)
        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(f"Dataset directory not found: {self.data_path}")

        self.classes = sorted(
            d for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
        )
        if not self.classes:
            raise FileNotFoundError(f"No class subfolders found in {self.data_path}.")
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        self.samples = [
            (os.path.join(self.data_path, cls, f), self.class_to_idx[cls])
            for cls in self.classes
            for f in sorted(os.listdir(os.path.join(self.data_path, cls)))
            if f.lower().endswith(IMG_EXTENSIONS)
        ]

        # Aspect-ratio-preserving resize with padding, then normalize to [-1, 1].
        self.transform = transforms.Compose([
            ResizeWithPadding(img_size, fill=(0, 0, 0)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.transform(Image.open(path).convert("RGB"))
        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}
