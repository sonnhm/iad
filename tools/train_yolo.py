import os

from ultralytics import YOLO


def train_yolo():
    # PATH to last checkpoint trong thư mục an toàn tuyệt đối
    last_ckpt = r"C:\AIP\iad\checkpoints\yolo\mvtec\weights\last.pt"

    if os.path.exists(last_ckpt):
        print(
            f"--- Đã tìm thấy checkpoint tại {last_ckpt}. Đang chuẩn bị RESUME training... ---"
        )
        model = YOLO(last_ckpt)
        resume = True
    else:
        print(
            "--- Không tìm thấy checkpoint cũ. Đang bắt đầu training MỚI từ yolov8n.pt... ---"
        )
        model = YOLO("yolov8n.pt")
        resume = False

    # Train / Resume the model
    # dataset.yaml is in C:\AIP\iad\datasets\yolo_dataset\dataset.yaml
    results = model.train(
        data=r"C:\AIP\iad\datasets\yolo_dataset\dataset.yaml",
        epochs=50,
        imgsz=640,
        device=0,  # use GPU
        project=r"C:\AIP\iad\checkpoints\yolo",
        name="mvtec",
        resume=resume,
    )

    # Export the model to ONNX for fast inference if needed
    path = model.export(format="onnx")
    print(f"Model exported to: {path}")


if __name__ == "__main__":
    train_yolo()
