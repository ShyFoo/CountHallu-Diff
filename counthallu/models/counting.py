"""Counting and quality models used to score generated images.

Checkpoints are plain ``state_dict`` files saved from these classes; the
``self.model`` attribute name is part of the checkpoint format, keep it.
"""

import timm
import torch.nn as nn
import torchvision.models as models


class CountingRegressor(nn.Module):
    """ResNet-50 regressor: predicts the instance count of each object class.

    Used for both ToyShape and SimObject counter checkpoints.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class RealHandQualityClassifier(nn.Module):
    """MaxViT binary classifier: is a generated hand image clean enough to count?"""

    def __init__(self, num_classes=2):
        super().__init__()
        self.model = timm.create_model("maxvit_small_tf_224", pretrained=True,
                                       num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
