"""
MVTec AD Dataset Loader (Mô-đun tải Dữ liệu MVTec AD).

Hỗ trợ 3 phân chia tĩnh (splits): train, valid, test.
- train (Tập huấn luyện): Gồm 80% ảnh đạt chuẩn (good) từ thư mục train/good. (Dùng để học trạng thái bình thường).
- valid (Tập kiểm định): Gồm 20% ảnh đạt chuẩn (good) còn lại. (Dùng để kiểm thử ngưỡng tin cậy).
- test (Tập thử nghiệm): Gồm TẤT CẢ các ảnh từ thư mục test, bao gồm cả ảnh chuẩn (good) và các loại lỗi (defect types).

Hỗ trợ tải Ground Truth Masks (Nhãn vùng lỗi ở cấp độ điểm ảnh - Pixel-level) cho công việc đánh giá:
- ground_truth/{defect_type}/{name}_mask.png → Binary mask (Mặt nạ nhị phân chỉ có trắng đen).
- Đối với ảnh 'good' → Tự động sinh ra mask toàn màu đen (all zeros) vì không có lỗi nào tồn tại.
"""

import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet normalization (Chuẩn hóa màu sắc theo hệ ImageNet)
# Mọi mô hình dùng Backbone (ResNet) pretrained trên tập ImageNet đều ĐẠT YÊU CẦU PHẢI dùng chuẩn cấu trúc này
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MVTecDataset(Dataset):
    """
    Dataset Adapter (Trạm Tiền xử lý Dữ liệu) cho chuỗi MVTec Anomaly Detection.

    Args:
        root (str): Đường dẫn tới thư mục gốc MVTec (nơi chứa các hộp folder như 'bottle', 'cable').
        category (str): Tên danh mục cấu hình (ví dụ: 'bottle').
        split (str): Chế độ tách dataset: "train" | "valid" | "test".
        image_size (int): Yêu cầu nắn/bóp kích thước ảnh (Scale Dimension) về cỡ này (Thường là 256x256).
        seed (int): Mầm gieo ngẫu nhiên (Random seed) để đảm bảo lần chạy nào cũng ra kết quả chia 80/20 giống hệt.
        load_masks (bool): Nếu True → Cho phép Load hệ thống Ground truth masks (Chỉ áp dụng cho tập Test).
        mask_size (int): Kích thước thu nén của mask (mặc định 32, vì Model PatchCore trích Feature Map có chiều 32x32).
    """

    VALID_SPLITS = ("train", "valid", "test")

    def __init__(
        self,
        root,
        category,
        split="train",
        image_size=256,
        seed=42,
        load_masks=False,
        mask_size=32,
    ):
        # Xác thực đầu vào chặt chẽ. Tránh gõ nhầm 'valid' thành 'val'
        assert (
            split in self.VALID_SPLITS
        ), f"split phải là một trong {self.VALID_SPLITS}, nhận được '{split}'"

        self.root = root
        self.category = category
        self.split = split
        self.image_size = image_size

        # CHỈ TRÍCH XUẤT MASKS khi ở trong chế độ Test
        self.load_masks = load_masks and (split == "test")
        self.mask_size = mask_size

        self.image_paths = []
        self.labels = []
        self.mask_paths = []  # Đường dẫn tới thư mục kho chứa ảnh Ground truth masks
        self.defect_types = []  # Chuỗi ghi lại Loại Defect cho mỗi bức ảnh

        # Lọc nhánh tiến trình
        if split in ("train", "valid"):
            self._load_train_valid(seed)
        else:
            self._load_test()

        # Transform (Chuỗi chuyển hóa hình học): Resize -> ToTensor (Ma trận) -> Normalize (Khoanh chuẩn)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _load_train_valid(self, seed):
        """
        Thuật toán xé nhỏ ảnh tốt thành Train (80%) và Validation (20%).
        """
        # Đường dẫn gốc trỏ về thư mục Good
        good_dir = os.path.join(self.root, self.category, "train", "good")

        # Quét lấy toàn bộ ảnh PNG trong thư mục Good (lọc bỏ các file linh tinh)
        all_images = sorted(
            [
                os.path.join(good_dir, f)
                for f in os.listdir(good_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ]
        )

        # Xáo trộn thứ tự ảnh ngẫu nhiên nhưng CỐ ĐỊNH qua Seed → Đảm bảo Reproducible Experiment
        rng = random.Random(seed)
        rng.shuffle(all_images)

        # Cắt kéo 80/20 theo độ dài mảng
        split_idx = int(len(all_images) * 0.8)

        if self.split == "train":
            selected = all_images[:split_idx]
        else:  # Cho valid
            selected = all_images[split_idx:]

        self.image_paths = selected

        # Mọi ảnh trong Train/Valid đều mang nhãn 0 (Normal / Good)
        self.labels = [0] * len(selected)

    def _load_test(self):
        """
        Quét và tải toàn bộ ảnh kiểm thử (Good + các loại Defect gãy/móp) + Cặp ảnh Ground Truth Mask của nó.
        """
        test_dir = os.path.join(self.root, self.category, "test")
        gt_dir = os.path.join(self.root, self.category, "ground_truth")

        for defect_type in sorted(os.listdir(test_dir)):
            defect_dir = os.path.join(test_dir, defect_type)

            if not os.path.isdir(defect_dir):
                continue

            for f in sorted(os.listdir(defect_dir)):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.image_paths.append(os.path.join(defect_dir, f))

                    # Logic Phân Nhãn (Labeling): Thấy chữ 'good' thì gán 0 (Bình thường), còn lại gán 1 (Lỗi)
                    self.labels.append(0 if defect_type == "good" else 1)
                    self.defect_types.append(defect_type)

                    # Trích lục Ground truth mask path
                    if self.load_masks and defect_type != "good":
                        # Cấu trúc Tên file mask của MVTec luốn có hậu tố: {name}_mask.png
                        name = os.path.splitext(f)[0]
                        mask_path = os.path.join(
                            gt_dir, defect_type, f"{name}_mask.png"
                        )
                        self.mask_paths.append(mask_path)
                    else:
                        # Với Good images (Ảnh bình thường) → Không tồn tại mask vật lý (sẽ được tự động tạo All-zeros ở dưới)
                        self.mask_paths.append(None)

    def _load_mask(self, mask_path):
        """
        Kích hoạt tải Ground truth mask vào RAM và nén cỡ (Resize) về định dạng mask_size (vd: 32x32).

        Returns:
            mask: NumPy Array ma trận 2D (H, W) chỉ có các hạt pixel giá trị 0 hoặc 1.
        """
        # Nếu Không có đường dẫn Mask (Ảnh đó là Good Image)
        if mask_path is None or not os.path.exists(mask_path):
            # Tạo tấm màn hình All Zero (Toàn màu đen trống) với cỡ mask_size để nộp lại
            return np.zeros((self.mask_size, self.mask_size), dtype=np.float32)

        # Mở ảnh ảnh gốc và nén về 1 kênh màu Trắng Đen (L - Grayscale) bằng PIL
        mask = Image.open(mask_path).convert("L")

        # Ép khung Resize nhưng dùng bộ lọc Nearest (Giữ nguyên các khối ảnh vuông góc, không nội suy làm mờ màu trắng/đen)
        mask = mask.resize((self.mask_size, self.mask_size), Image.NEAREST)
        mask = np.array(mask, dtype=np.float32)

        # Nhị Phân (Binarization): Ép giá trị hạt pixel > 0 thành 1.0 (Có lỗi Anomaly), Ngược lại bằng 0 (Bình Thường - Normal)
        mask = (mask > 0).astype(np.float32)
        return mask

    def __len__(self):
        """Trả về số lượng tổng cộng mẫu trong thư mục"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Hàm Hook cơ bản của Dataset Iterator.
        Trả về Lô dữ liệu cụ thể ở index: (image_tensor, label) hoặc (image_tensor, label, mask).

        Khi cờ hiệu load_masks=True:
            mask: Mảng Numpy (mask_size, mask_size) phản ánh lỗi tại cấp độ Pixel-level Ground Truth (GT).
        """
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]

        if self.load_masks:
            mask = self._load_mask(self.mask_paths[idx])
            return img, label, mask

        return img, label

    def __repr__(self):
        # Chuỗi in thông báo Debug khi được gọi tới Object
        return (
            f"MVTecDataset(category={self.category}, split={self.split}, "
            f"size={len(self)}, image_size={self.image_size}, "
            f"load_masks={self.load_masks})"
        )
