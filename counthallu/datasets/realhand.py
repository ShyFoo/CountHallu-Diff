"""RealHand: photos of single human hands paired with segmentation masks.

Returns {"image": Tensor, "mask": Tensor}; the mask enables joint
image+mask diffusion (JDM). Image i and mask i must share the same stem.
"""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from counthallu.datasets.common import list_images, normalize_transform


class RealHand(Dataset):
    SUBDIR = "RealHand"

    def __init__(self, img_size=256, mode="train", hub_dataset=None, data_root=None):
        self.transform = normalize_transform(img_size)
        self.flip = mode == "train"
        self.hub_dataset = hub_dataset

        if hub_dataset is None:
            if data_root is None:
                raise ValueError("Provide `data_root` when not using a hub dataset.")
            data_path = os.path.join(data_root, self.SUBDIR)
            self.img_dir = os.path.join(data_path, "images")
            self.mask_dir = os.path.join(data_path, "masks")
            self.img_names = list_images(self.img_dir)
            self.mask_names = list_images(self.mask_dir)
            if len(self.img_names) != len(self.mask_names):
                raise ValueError(
                    f"{len(self.img_names)} images but {len(self.mask_names)} masks in {data_path}."
                )

    def __len__(self):
        return len(self.hub_dataset) if self.hub_dataset is not None else len(self.img_names)

    def __getitem__(self, idx):
        if self.hub_dataset is not None:
            item = self.hub_dataset[idx]
            image, mask = item["image"].convert("RGB"), item["mask"].convert("RGB")
        else:
            image = Image.open(os.path.join(self.img_dir, self.img_names[idx])).convert("RGB")
            mask = Image.open(os.path.join(self.mask_dir, self.mask_names[idx])).convert("RGB")

        # Flip must be applied to image and mask together, so it cannot live
        # inside the (per-tensor) torchvision transform.
        if self.flip and torch.rand(1) < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {"image": self.transform(image), "mask": self.transform(mask)}
