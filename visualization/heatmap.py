"""
Heatmap visualization cho Autoencoder anomaly detection.

Hiển thị: ảnh gốc | ảnh reconstruct | heatmap vùng bất thường
"""

import matplotlib.pyplot as plt
import torch

from app_utils.image_utils import denormalize
from data_processing.mvtec import IMAGENET_MEAN, IMAGENET_STD


def show_heatmap(input_img, recon_img, save_path=None):
    """
    Hiển thị heatmap anomaly từ reconstruction error.

    Args:
        input_img: ảnh gốc (1, C, H, W) hoặc (C, H, W)
        recon_img: ảnh reconstruct (1, C, H, W) hoặc (C, H, W)
        save_path: (optional) lưu ảnh ra file
    """
    # Xử lý batch dimension
    if input_img.dim() == 4:
        input_img = input_img.squeeze(0)
    if recon_img.dim() == 4:
        recon_img = recon_img.squeeze(0)

    # Tính error heatmap
    error = (input_img - recon_img) ** 2
    heatmap = error.mean(dim=0).numpy()  # mean qua channels

    # Denormalize để hiển thị
    img_display = denormalize(input_img)
    recon_display = denormalize(recon_img)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_display)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(recon_display)
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    axes[2].imshow(heatmap, cmap="jet")
    axes[2].set_title("Anomaly Heatmap")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
