import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights

class PatchCore(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-2]
        )

    def forward(self, x):

        features = self.feature_extractor(x)

        B, C, H, W = features.shape

        patches = features.permute(0,2,3,1).reshape(-1, C)

        return patches