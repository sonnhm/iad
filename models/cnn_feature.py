"""
CNN Feature Extractor — trích feature vector (Vector đặc trưng) từ ảnh dùng mạng ResNet18 Pre-trained.

Dùng kết hợp với thuật toán Học thống kê One-Class SVM (OC-SVM) để phát hiện anomaly:
    1. Trích features từ ảnh bình thường → Dùng mảng số lấy được đi huấn luyện biên giới cho OC-SVM (Train)
    2. Test: Trích features từ ảnh kiểm tra → OC-SVM tính khoảng cách đến đường biên giới (Decision Score)
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class CNNFeatureExtractor(nn.Module):
    """
    Feature extractor (Bộ rút xương đặc trưng) sử dụng ResNet18 pretrained (Đã học từ Kho dữ liệu khổng lồ ImageNet).

    Cắt bỏ lớp Fully Connected (FC - Mạng kết nối đầy đủ) phân loại cuối cùng →
    Đầu ra (Output) lúc này thuần túy là chuỗi mảng số liệu vector 512-chiều (512-d).

    Input: (B, 3, H, W)   (Batch Size, 3-Kênh Màu, Chiều cao, Chiều rộng).
    Output: (B, 512)      (Batch Size, 512 Điểm ảnh).
    """

    def __init__(self):
        super().__init__()

        # Tải mạng cấu trúc tủy sống xương (Backbone) ResNet18 từ PyTorch có kèm trí khôn (Weights) mặc định
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Lấy tất cả layers trừ FC phân loại cuối
        # Giải phẫu children(): conv1, bn1, relu, maxpool, seq layer1-4, avgpool, fc
        # Ở đây ta dùng thủ thuật slicing [:-1] trong mảng Python → Cắt sát đuôi, loại phần tử (fc) cuối cùng.
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        """
        Luồng đi Chuyển tiếp (Feed-Forward Pass).

        Args:
            x: input images (B, 3, H, W)

        Returns:
            feature_vectors (B, 512): Ma trận số đại diện cho tính chất của hình ảnh.
        """
        # Cho ảnh chạy qua cỗ máy lấy đặc trưng (Features)
        x = self.features(
            x
        )  # Kích thước tại đây sẽ là (B, 512, 1, 1). Là 1 khối lập phương chiều cực sâu.

        # San phẳng (Flatten) khối lập phương thành một dải băng dài 512 đốt
        x = torch.flatten(x, 1)  # Kích thước cuối cùng: (B, 512)

        return x
