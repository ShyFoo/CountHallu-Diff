"""Shared building blocks for the counting datasets.

Every dataset returns dicts so that training / evaluation code can stay
agnostic to the concrete dataset: ``{"image": Tensor}`` plus optional
``"label"`` (per-class instance counts) and ``"mask"`` (segmentation mask).
Images are normalized to [-1, 1] as expected by the diffusion pipelines.
"""

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def normalize_transform(img_size: int, random_flip: bool = False):
    """Resize -> (optional flip) -> tensor in [-1, 1]."""
    steps = [transforms.Resize((img_size, img_size))]
    if random_flip:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    steps += [
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
    return transforms.Compose(steps)


def list_images(folder: str):
    """Image filenames in ``folder``, sorted by their integer stem (0.png, 1.png, ...)."""
    names = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTENSIONS)]
    return sorted(names, key=lambda f: int(os.path.splitext(f)[0]))


class LabeledImageFolder(Dataset):
    """A folder of images with per-image counting labels in ``labels.csv``.

    Expected layout::

        <data_root>/<subdir>/
            images/           # 0.png, 1.png, ...
            labels.csv        # filename, <class_1>, <class_2>, ...

    Alternatively pass ``hub_dataset`` (a loaded Hugging Face split with
    "image" and "label" columns) to skip local files entirely.
    """

    SUBDIR = None  # override in subclasses

    def __init__(self, img_size=128, mode="train", hub_dataset=None, data_root=None,
                 random_flip=False):
        self.transform = normalize_transform(img_size, random_flip and mode == "train")
        self.hub_dataset = hub_dataset

        if hub_dataset is None:
            if data_root is None:
                raise ValueError("Provide `data_root` when not using a hub dataset.")
            data_path = os.path.join(data_root, self.SUBDIR)
            self.img_dir = os.path.join(data_path, "images")
            self.img_names = list_images(self.img_dir)

            labels = pd.read_csv(os.path.join(data_path, "labels.csv"))
            label_cols = labels.columns[1:].tolist()
            self.filename2label = {
                row["filename"]: torch.tensor(row[label_cols].tolist(), dtype=torch.long)
                for _, row in labels.iterrows()
            }

    def __len__(self):
        return len(self.hub_dataset) if self.hub_dataset is not None else len(self.img_names)

    def __getitem__(self, idx):
        if self.hub_dataset is not None:
            item = self.hub_dataset[idx]
            image = self.transform(item["image"].convert("RGB"))
            label = torch.as_tensor(item["label"], dtype=torch.long)
        else:
            name = self.img_names[idx]
            image = self.transform(Image.open(os.path.join(self.img_dir, name)).convert("RGB"))
            label = self.filename2label[name]
        return {"image": image, "label": label}
