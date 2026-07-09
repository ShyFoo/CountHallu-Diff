"""Dataset registry and the single dataloader entry point used by all scripts."""

import os

from torch.utils.data import DataLoader

from counthallu.datasets.realhand import RealHand
from counthallu.datasets.simobject import SimObject
from counthallu.datasets.toyshape import ToyShape

# All datasets share the constructor signature
# (img_size, mode, hub_dataset, data_root) and return dict samples.
DATASETS = {
    "toyshape": ToyShape,
    "simobject": SimObject,
    "realhand": RealHand,
}


def get_dataloader(dataset_name, data_root=None, hub_dataset=None, img_size=128,
                   batch_size=8, num_workers=0, shuffle=True, pin_memory=False,
                   drop_last=True, mode="train"):
    """Build ``(dataset, dataloader, real_image_dir)`` for a registered dataset.

    ``real_image_dir`` points at the on-disk real images (used as the FID
    reference set); it is None for hub-only datasets.
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported: {', '.join(DATASETS)}."
        )

    dataset_cls = DATASETS[dataset_name]
    dataset = dataset_cls(
        img_size=img_size, mode=mode, hub_dataset=hub_dataset, data_root=data_root
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    subdir = getattr(dataset_cls, "SUBDIR", None)
    real_image_dir = os.path.join(data_root, subdir, "images") if (data_root and subdir) else None
    return dataset, dataloader, real_image_dir
