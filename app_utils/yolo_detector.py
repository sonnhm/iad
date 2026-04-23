"""
Module YOLO Detector - Nhận diện phân loại sản phẩm.

Module này cung cấp lớp YOLODetector sử dụng kiến trúc siêu nhẹ (Ultralytics YOLOv8n)
để đóng vai trò làm hệ thống phân loại sơ cấp (Primary Classification).
Mục tiêu là tự động nhận diện loại sản phẩm đưa vào (ví dụ: 'bottle', 'cable')
trước khi chuyển giao luồng dữ liệu sang cho các mô hình chuyên biệt (PatchCore, Autoencoder)
để phát hiện lỗi (Anomaly Detection).
"""

import os

import torch
from ultralytics import YOLO


class YOLODetector:
    """
    Hệ thống phân loại sản phẩm tiền xử lý (Pre-processing Classifier).

    YOLODetector tải trọng số (weights) đã được huấn luyện riêng biệt trên 15 danh mục
    (categories) của dataset MVTec AD nhằm xác thực tên sản phẩm một cách tự động.
    """

    def __init__(self, model_path=None, device="cpu"):
        """
        Khởi tạo hệ thống YOLO.

        Args:
            model_path (str, optional): Đường dẫn tới mô hình lưu trữ. Mặc định trỏ
                                        về 'checkpoints/yolo/mvtec/weights/best.pt'.
            device (str, optional): Tính toán sử dụng 'cuda' hoặc 'cpu'.
        """
        if model_path is None:
            # Liên kết động tới thư mục checkpoint an toàn
            model_path = os.path.join(
                "checkpoints", "yolo", "mvtec", "weights", "best.pt"
            )

        self.model = None
        self.device = device

        # Kiểm tra sự tồn tại của trọng số (weights) trước khi khởi tạo mô hình
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(
                f"[YOLO] Tải thành công mô hình nhận diện sản phẩm trên {self.device}: {model_path}"
            )
        else:
            print(
                f"[YOLO-WARNING] Không tìm thấy tệp trọng số tại {model_path}. "
                "Inference chỉ dựa trên PatchCore tĩnh."
            )

    def detect(self, image):
        """
        Trích xuất phân loại (Classification) lớn nhất từ khung bao (Bounding Box) đầu tiên.

        Args:
            image (PIL.Image / np.ndarray / str): Hình ảnh đầu vào chưa qua xử lý.

        Returns:
            tuple:
                - category_name (str): Tên phân loại nhận diện được (vd: 'bottle'). Trả về None nếu không phát hiện.
                - confidence (float): Độ tin cậy của mô hình YOLO đối với phân loại trên.
                - bbox (list): Tọa độ khung viền chứa vật thể theo dạng [x1, y1, x2, y2].
        """
        # Nếu mô hình không tồn tại do thiếu tệp trọng số, trả về rỗng (fallback)
        if self.model is None:
            return None, 0.0, None

        # Triển khai thuật toán Ultralytics (tắt log console bằng verbose=False để giữ Terminal sạch sẽ)
        results = self.model(image, verbose=False)

        # Trả về rỗng nếu YOLO không tìm thấy bất kỳ vật thể hợp lệ nào trên hình ảnh
        if not results or len(results[0].boxes) == 0:
            return None, 0.0, None

        # Box thứ 0 luôn được Ultralytics xếp hạng là Box có thuộc tính tin cậy cao nhất (Top-1 prediction)
        primary_box = results[0].boxes[0]
        category_index = int(primary_box.cls[0].item())
        category_name = self.model.names[category_index]

        # Làm tròn phần trăm độ tin cậy
        confidence = primary_box.conf[0].item()

        # Trích xuất tọa độ không gian Bounding Box
        bbox = primary_box.xyxy[0].tolist()

        return category_name, confidence, bbox
