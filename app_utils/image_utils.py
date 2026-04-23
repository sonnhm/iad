"""
Shared image utilities -- centralized helpers for image processing.

Consolidates common functions (denormalize, numpy_to_pil, etc.)
to avoid code duplication across visualization and app modules.
"""

import numpy as np
import torch
from PIL import Image

from data_processing.mvtec import IMAGENET_MEAN, IMAGENET_STD


def denormalize(img_tensor):
    """
    Convert ImageNet-normalized tensor to [0,1] numpy array (H, W, C).

    Reverses the standard ImageNet normalization:
        original = (normalized * std) + mean

    Args:
        img_tensor: (C, H, W) tensor, ImageNet-normalized

    Returns:
        (H, W, C) numpy array with values clamped to [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()
