"""
PatchCore heatmap visualization.

Hiển thị anomaly score ở mức patch-level:
    - Mỗi patch có 1 distance score (tới memory bank)
    - Reshape thành grid → heatmap

Sử dụng PatchCore class từ gốc (không dùng sklearn).
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from app_utils.image_utils import denormalize
from data_processing.mvtec import IMAGENET_MEAN, IMAGENET_STD


def show_patchcore_heatmap(patchcore, img_tensor, save_path=None):
    """
    Tạo và hiển thị PatchCore anomaly heatmap.

    Args:
        patchcore: PatchCore instance (đã fit memory bank)
        img_tensor: (C, H, W) test image tensor
        save_path: (optional) lưu ảnh
    """
    # Predict
    scores, anomaly_maps = patchcore.predict(img_tensor)
    score = scores[0]
    amap = anomaly_maps[0]  # (H', W')

    # Normalize anomaly map
    amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

    # Resize heatmap về kích thước ảnh gốc
    img_np = denormalize(img_tensor)
    h, w = img_np.shape[:2]

    heatmap_resized = (
        np.array(Image.fromarray((amap_norm * 255).astype(np.uint8)).resize((w, h)))
        / 255.0
    )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title(f"PatchCore Anomaly Score: {score:.4f}")
    axes[1].axis("off")

    axes[2].imshow(img_np)
    axes[2].imshow(heatmap_resized, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
