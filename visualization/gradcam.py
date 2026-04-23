"""
Grad-CAM visualization — Giải thích trực quan vùng mà CNN "chú ý" đến.

Grad-CAM (Gradient-weighted Class Activation Mapping):
    1. Forward pass: tính output features
    2. Backward pass: tính gradients
    3. Nhân gradients × activation → weighted activation map
    4. Lấy ReLU → heatmap (chỉ giữ vùng ảnh hưởng tích cực)

Xem chi tiết giải thích trong docs/GRADCAM_EXPLAINED.md
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from app_utils.image_utils import denormalize
from data_processing.mvtec import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    """
    Grad-CAM cho CNN Feature Extractor.

    Cách hoạt động:
        1. Hook vào target_layer để lấy activations và gradients
        2. Forward pass → tính target score
        3. Backward pass → lấy gradients
        4. Global average pooling gradients → weights (alpha_k)
        5. Weighted sum: Σ(alpha_k × activation_k) → heatmap
        6. ReLU → chỉ giữ vùng ảnh hưởng tích cực

    Args:
        model: CNN model (cần có .features attribute)
        target_layer: layer cần visualize (ví dụ: model.features[-3])
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.activations = None  # Lưu output của target layer
        self.gradients = None  # Lưu gradient của target layer

        # Register hooks để "bắt" activations và gradients
        # Hook = callback function chạy khi forward/backward qua layer
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Hook: lưu activation (output) của target layer khi forward."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook: lưu gradient khi backward qua target layer."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_img):
        """
        Tạo Grad-CAM heatmap.

        Args:
            input_img: (1, C, H, W) tensor (đã normalize)

        Returns:
            heatmap: (H, W) numpy array, giá trị [0, 1]
        """
        # Bật gradient cho input (cần cho backward)
        input_img.requires_grad_(True)

        # Step 1: Forward pass
        output = self.model(input_img)

        # Step 2: Chọn target → dùng mean activation làm scalar target
        #         (vì anomaly detection không có class cụ thể)
        target = output.mean()

        # Step 3: Backward pass → tính gradients
        self.model.zero_grad()
        target.backward()

        # Step 4: Global Average Pooling trên gradients
        #         → importance weight cho mỗi feature map (channel)
        #
        #         gradients shape: (1, C, H', W')
        #         weights shape: (1, C, 1, 1) → (C,)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Step 5: Weighted combination
        #         Σ (weight_k × activation_k) cho mỗi channel k
        #
        #         weights: (1, C, 1, 1) × activations: (1, C, H', W')
        #         → cam: (1, 1, H', W')
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # Step 6: ReLU → chỉ giữ gradient dương (vùng ảnh hưởng tích cực)
        cam = torch.relu(cam)

        # Normalize về [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def show_gradcam(model, target_layer, img_tensor, save_path=None):
    """
    Tạo và hiển thị Grad-CAM.

    Args:
        model: CNN model
        target_layer: layer cần visualize
        img_tensor: (C, H, W) image tensor (normalized)
        save_path: (optional) lưu ảnh
    """
    gradcam = GradCAM(model, target_layer)

    # Generate heatmap
    img_batch = img_tensor.unsqueeze(0)
    heatmap = gradcam.generate(img_batch)

    # Resize heatmap về kích thước ảnh gốc
    img_np = denormalize(img_tensor)
    h, w = img_np.shape[:2]

    heatmap_resized = (
        np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h)))
        / 255.0
    )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
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
