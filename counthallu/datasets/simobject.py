"""SimObject: rendered images of everyday objects with per-class counting labels."""

from counthallu.datasets.common import LabeledImageFolder


class SimObject(LabeledImageFolder):
    SUBDIR = "SimObject"

    def __init__(self, img_size=256, mode="train", hub_dataset=None, data_root=None):
        super().__init__(img_size, mode, hub_dataset, data_root, random_flip=True)
