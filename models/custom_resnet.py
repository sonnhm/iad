"""
Custom ResNet18 backbone (Mạng Xương Sống Tùy chỉnh ResNet18)
Tự implement theo paper (luận văn học thuật) "Deep Residual Learning" (He et al., 2015).

Kiến trúc chuẩn của ResNet18 (Architecture Blueprint):
    - Stem (Đoạn cuống): conv1 (7x7, stride=2) → BN → ReLU → MaxPool
    - layer1: 2 BasicBlock (64 filters - 64 bộ lọc)
    - layer2: 2 BasicBlock (128 filters, stride=2 - Giảm thu nhỏ kích thước hình)
    - layer3: 2 BasicBlock (256 filters, stride=2)
    - layer4: 2 BasicBlock (512 filters, stride=2)
    - Head (Đoạn đầu): AdaptiveAvgPool (Hồ Gộp Trung bình) → FC(512, num_classes)

Cơ chế BasicBlock (Khối Mạng Cơ bản - Lõi của ResNet):
    Input (Đầu vào) → conv3x3 → BN → ReLU → conv3x3 → BN → (+ Shortcut/Skip Connection) → ReLU
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Basic Residual Block (Khối Thặng dư Cơ bản) cho ResNet18/34.

    Gồm 2 lớp conv3x3 (Lọc Tích chập kích thước 3x3) đi kèm đường nối tắt (skip connection / shortcut).
    Việc cộng đường nối tắt (shortcut) giúp triệt tiêu hiện tượng Vanishing Gradient (Mất mát Đạo hàm)
    ở các mạng Nơ-ron sâu.
    Nếu kích thước ảnh bị cắt đôi (do stride != 1 hoặc in/out channels bị chênh lệch),
    ta phải dùng thêm 1 mạng Conv 1x1 để nén shortcut cho vừa khớp bản đồ (Adjust shortcut).
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Main path (Đường chính): conv3x3 → BN → ReLU → conv3x3 → BN
        # Bước 1: Màng lọc tích chập (Conv2d)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,  # Không cần Bias vì ngay sau đó có BN (BatchNorm) triệt tiêu Bias rồi
        )
        # Chuẩn hóa lô (Batch Normalization)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Bước 2: Tích chập lớp thứ 2
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Trình kích hoạt phi tuyến (Activation)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection (Đường bay tắt - Residual)
        # Ban đầu khởi tạo đường tắt rỗng (Đi thẳng)
        self.shortcut = nn.Sequential()
        # Khi stride khác 1 (Tức là bị ép nhỏ kích thước màn hình),
        # hoặc số lượng Kênh (Channels) đầu vào khác đầu ra → Cần Mạng Downsample (Hạ chuẩn)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # Conv 1x1 ở đây chỉ làm nhiệm vụ "Bóp" hoặc "Mở" số kênh (Kích thước) mà không làm méo hình
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        """Luồng chảy dữ liệu (Forward Pass)"""
        # Chạy đường bộ (Main path)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Phép Cộng Thặng Dư thần thánh của ResNet (Residual connection)
        # Lấy kết quả bị nhào nặn (out) CỘNG với ảnh nguyên bản (shortcut(x))
        out += self.shortcut(x)

        # Bọc ReLU ở bước cuối cùng
        out = self.relu(out)

        return out


class CustomResNet18(nn.Module):
    """
    Custom ResNet18 — tự implement từ đầu (Từ lõi PyTorch), không gọi mô hình tạo sẵn
    của Thư viện torchvision (Đảm bảo việc can thiệp y khoa tới tận cùng các lớp Layer).

    Args:
        num_classes: Số lượng lớp vật thể nhãn (Mặc định 1000 cho dataset ImageNet).
                     - Tại dự án IAD, ta đặt "None" nếu chỉ muốn dùng nó làm "Máy Rút Tính Trạng" (Feature Extractor)
                       cho Knowledge Distillation và PatchCore.
    """

    def __init__(self, num_classes=1000):
        super().__init__()

        # Khởi tạo hẻm chứa kênh (channels) ban đầu
        self.in_channels = (
            64  # Số kênh nền móng (Dùng update tự động trong vòng lặp _make_layer)
        )

        # Stem (Phần Thân Cuống Bắt Đầu): conv7x7 → BN → ReLU → MaxPool
        # Ở đây dùng màng lọc rộng 7x7 để bắt các chi tiết thô, lớn
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 Layer Groups (Bốn giai đoạn màng lọc cốt lõi): Cấu tạo theo mảng [2, 2, 2, 2] blocks (Khối)
        # Tại mỗi layer, độ sâu kênh (Channels) TĂNG GẤP ĐÔI, và kích thước dài/rộng GIẢM MỘT NỬA
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        # Classification head (Ngõ Xuất Phân Loại Cuối Cùng)
        # Hồ nén trung bình (Ép cục gạch đặc trưng về thành 1 chấm duy nhất)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # FC (Fully Connected Layer) - Mạng thần kinh phẳng quyết định phân loại vật thể
        self.fc = None
        if num_classes is not None:
            self.fc = nn.Linear(512, num_classes)

        # Khởi tạo trọng số (Weights) theo luận văn khoa học
        self._initialize_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        """
        Tạo một cụm tầng lọc (Layer group) bao gồm nhiều khối (BasicBlock).
        Đặc biệt: Khối đầu tiên trong chuỗi có thể có stride > 1 để ép nhỏ kích thước hình (Downsample).
        """
        # Tạo mảng bước nhảy (strides). Chỉ khối đầu tiên mới dùng stride (thường là 2), còn lại là 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride=s))
            # Cập nhật số Kênh đầu vào của chu trình tiếp theo
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Khởi tạo trọng số rỗng theo chuẩn toán học Kaiming initialization (Trị số Variance bớt lệch)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Init theo phân phối chuẩn Kaiming (Chuyên dụng cho hàm kích hoạt ReLU)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # Điền 1
                nn.init.constant_(m.bias, 0)  # Điền 0
            elif isinstance(m, nn.Linear):
                # Bias ngẫu nhiên biên độ hẹp 0.01 cho FC
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass — Trả về Logit dự đoán (nếu train FC) hoặc Trả về Mảng Vector đặc trưng (nếu Extract)."""
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Đi xuyên qua 4 tầng sâu (Residual layers)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling (Ép bẹp ma trận 3D)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # San phẳng 1D

        # Classification head (Nếu tồn tại cỗ máy dự đoán nhãn thì đẩy vào đó)
        if self.fc is not None:
            x = self.fc(x)

        return x

    def get_feature_layers(self):
        """Trả về Dictionary dạng từ điển các Tầng Sâu ngầm (Intermediate layers) để móc trích xuất đặc trưng
        (Dành riêng cho PatchCore móc nối Layer2 và Layer3)."""
        return {
            "layer1": self.layer1,
            "layer2": self.layer2,
            "layer3": self.layer3,
            "layer4": self.layer4,
        }


def custom_resnet18(num_classes=1000):
    """
    Factory function (Hàm chế tạo dây chuyền) tạo Object Custom ResNet18 nhanh.
    """
    return CustomResNet18(num_classes=num_classes)
