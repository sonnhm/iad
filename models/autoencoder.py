"""
Convolutional Autoencoder (CAE) for Unsupervised Anomaly Detection
(Bộ Mã hóa Tự động Tích chập dùng cho Phát hiện Bất thường Không giám sát)

This module provides a robust Convolutional Autoencoder architecture deployed within the IAD pipeline.
(Mô-đun này cung cấp kiến trúc CAE mạnh mẽ được triển khai trong luồng IAD.)

The fundamental logic relies on the assumption that a model trained exclusively on 'nominal'
(Logic cơ bản phụ thuộc vào giả định rằng: một mô hình được huấn luyện ĐỘC QUYỀN trên tập dữ liệu 'chuẩn' - defect-free)
will perfectly reconstruct similar nominal inputs, but will fail significantly
(sẽ tái tạo hoàn hảo các đặc trưng chuẩn đó, nhưng sẽ GẶP THẤT BẠI nghiêm trọng)
when attempting to reconstruct anomalous regions (e.g., structural defects like scratches or dents).
(khi cố gắng vẽ lại các vùng bất thường như vết xước hay vết móp - do chưa từng nhìn thấy lúc học).

Methodology (Phương pháp luận):
- Encoder (Bộ Mã hóa): Compresses the high-dimensional input image (256x256x3) through successive convolution layers
  and ReLU activations into a lower-dimensional latent bottleneck representation (Ép nhỏ ảnh đầu vào qua các lớp
  tích chập liên tiếp xuống một không gian Nút thắt cổ chai tiềm ẩn - Latent Space).
- Decoder (Bộ Giải mã): Upsamples the latent vector back to the original input dimensions using transposed convolutions.
  (Kích phóng Vector tiềm ẩn quay ngược lại kích thước ảnh ban đầu bằng Tích chập Đảo ngược - ConvTranspose2d).
- Detection (Phát hiện): The Mean Squared Error (MSE) between the input pixel intensity and the reconstructed output
  serves as both the training loss and the pixel-level anomaly score map during inference.
  (Lấy Hiệu số Bình phương Trung bình - MSE - giữa điểm ảnh gốc và điểm ảnh sinh ra làm Anomaly Score).
"""

import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Architecture details (Chi tiết Kiến trúc):
        - Symmetric Encoder-Decoder blueprint (Thiết kế đối xứng hoàn hảo 1:1).
        - Input (Đầu vào): Tensor (B, 3, 256, 256) normalized via MVTec statistics (Đã chuẩn hóa).
        - Latent Space (Không gian Tiềm ẩn): Compressed feature map bottleneck (B, 64, 8, 8).
        - Output (Đầu ra dạng ảnh): Reconstructed Tensor (B, 3, 256, 256).

    The network uses Batch Normalization universally to stabilize training dynamics over
    diverse industrial artifact textures.
    (Mạng sử dụng Chuẩn hóa Hàng loạt trên toàn bộ cấu trúc để ổn định gradient huấn luyện đối với
    nhiều loại bề mặt vật liệu công nghiệp khác nhau).
    """

    def __init__(self):
        super().__init__()

        # Mạng thu hẹp Spatial Dimension (Kích thước bề mặt ảnh) liên tiếp chia đôi
        # Lộ trình: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            # Bước Nén 1: (256x256) -> (128x128). Số kênh màng lọc (Channels): 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Bước Nén 2: (128x128) -> (64x64). Kênh: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Bước Nén 3: (64x64) -> (32x32). Kênh: 64 -> 64 (Giữ nguyên độ sâu để chống loãng thông tin)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Bước Nén 4: (32x32) -> (16x16)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Bước Nén 5 (Đáy): (16x16) -> (8x8) -> Đây là Latent Space (Không gian Ẩn nén cao độ)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Mạng bung Kích thước Spatial Dimension (Kích thước bề mặt ảnh) lên gấp 2 liên tục
        # Lộ trình: 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.decoder = nn.Sequential(
            # Bước Bung 1: Tích chập Đảo (ConvTranspose2d) ép (8x8) -> (16x16)
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Bước Bung 2: (16x16) -> (32x32)
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Bước Bung 3: (32x32) -> (64x64)
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Bước Bung 4: Giảm Kênh (64 -> 32), (64x64) -> (128x128)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Bước Bung Cuối (Tái Tạo): Đẩy 32 kênh về 3 kênh màu RGB (3), (128x128) -> (256x256)
            # Không sử dụng BatchNorm hay Activation Function ở lớp này (Để cho dải màu được bung lụa hết mức)
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        """
        Luồng đi Chuyển tiếp (Feed-Forward Pass).

        Args:
            x: input images (Tensor Mảng gốc) mang định dạng (B, 3, 256, 256)

        Returns:
            out (reconstructed images): Tensor Đã qua vẽ lại mang định dạng (B, 3, 256, 256)
        """
        # Bước 1: Nén ảnh x vào không gian ẩn qua lõi Z (Latent Bottleneck)
        z = self.encoder(x)

        # Bước 2: Vẽ lại ảnh qua ngõ Decoder
        out = self.decoder(z)

        return out
