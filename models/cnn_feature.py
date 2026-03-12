import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CNNFeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, 1)

        return x