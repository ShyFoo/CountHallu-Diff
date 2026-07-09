"""LSUN Bedrooms, streamed from the Hugging Face Hub (`pcuenq/lsun-bedrooms`).

Used only for pretraining latent diffusion backbones; it carries no counting
labels.
"""

from datasets import load_dataset
from torch.utils.data import Dataset

from counthallu.datasets.common import normalize_transform


class LsunBedrooms(Dataset):
    HUB_ID = "pcuenq/lsun-bedrooms"

    def __init__(self, img_size=256, mode="train", hub_dataset=None, data_root=None):
        self.transform = normalize_transform(img_size, random_flip=mode == "train")
        if hub_dataset is not None:
            self.dataset = hub_dataset
        else:
            splits = load_dataset(self.HUB_ID, trust_remote_code=True)
            if mode not in splits:
                raise ValueError(f"Invalid split '{mode}'. Available: {list(splits)}")
            self.dataset = splits[mode]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"].convert("RGB")
        return {"image": self.transform(image)}
