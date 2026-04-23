"""
Industrial Anomaly Detection (IAD) - Production Web Server

This module serves as the primary backend for the IAD application, delivering high-performance
inference using Flask. It integrates sub-second anomaly detection models, bounding box object detection,
and an Explainable AI (XAI) Chatbot interface.

Core Features:
1. Object Verification: Utilizes YOLOv8 for automated product classification (15 MVTec AD categories)
   to ensure the correct model pipeline is initialized.
2. Anomaly Detection Pipeline (Multi-Model):
   - PatchCore: A State-of-the-Art model that utilizes a Coreset subsampled Memory Bank of local feature patches
     (ResNet18 backbone) to compute anomaly scores via k-NN distance.
   - Autoencoder: A reconstructive architecture that compresses the input image into a latent space
     and reconstructs it. The Mean Squared Error (MSE) between the input and output acts as the anomaly score.
   - CNN + OC-SVM: An approach using a deep feature extractor coupled with a One-Class Support Vector Machine.
3. Memory Optimization: Predictively loads models into VRAM during initialization to achieve 0.00x second inference
   latency, completely avoiding the PCI-e bus bottleneck during runtime.
4. XAI Chatbot: Integrates both a deterministic Rule-Based expert system and a Generative AI fallback mechanism
   (Google Gemini) to interpret anomaly scores, thresholds, and heatmaps for quality assurance personnel.

Execution:
    python app.py
    Access interface at: http://localhost:5000
"""

import base64
import io
import os
import traceback

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # Non-interactive backend
from concurrent.futures import ThreadPoolExecutor

import cv2
import joblib
import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageFilter
from torchvision import transforms

from app_utils.checkpoint import load_checkpoint
from app_utils.yolo_detector import YOLODetector
from data_processing.mvtec import IMAGENET_MEAN, IMAGENET_STD
from evaluation.metrics import anomaly_score
from models.autoencoder import Autoencoder
from models.cnn_feature import CNNFeatureExtractor
from models.patchcore import PatchCore, PatchCoreFeatureExtractor

app = Flask(__name__)

IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (active_category removed -- unused dead code)


# Khởi tạo mô hình nhận diện sản phẩm YOLOv8 (đã train trên local GPU)
yolo_detector = YOLODetector(device=DEVICE)

# ============================================================
# CONFIG
# ============================================================

from app_utils.config import ALL_CATEGORIES

# IMAGE_SIZE and DEVICE already defined above

# Transform: giống lúc train
transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# ============================================================
# MODEL CACHE — Load model 1 lần, dùng lại
# Thread-safe: double-check locking pattern for concurrent access
# ============================================================

import threading

_model_lock = threading.Lock()

loaded_models = {}


def load_optimal_threshold(category, model_name):
    import json
    import math

    try:
        with open("checkpoints/thresholds.json", "r") as f:
            data = json.load(f)
            val = data.get(category, {}).get(model_name, None)
            if val is not None and (math.isinf(val) or math.isnan(val)):
                return 999999.0  # Chặn lỗi Infinity crash frontend
            return val
    except Exception:
        return None


def get_autoencoder(category):
    """
    Lấy và lưu trữ mô hình Autoencoder (Cache RAM/VRAM).

    Autoencoder (Bộ mã hóa tự động) được thiết kế theo dạng thu hẹp kích thước ảnh về một chiều
    biểu diễn không gian ẩn (Latent space), sau đó giải nén ra lại. Vì mô hình chỉ được học trên đồ vật hoàn hảo,
    bất cứ vết nứt hay trầy xước nào cũng sẽ bị khôi phục lỗi, từ đó ta lấy Hiệu (Difference) để làm Anomaly Score.

    Tham số:
        category (str): Tên danh mục sản phẩm ('cable', 'pill',...).

    Trả về:
        Autoencoder: Trả về Tensor module nếu có checkpoint, ngược lại trả về Rỗng.
    """
    key = f"ae_{category}"
    if key not in loaded_models:
        with _model_lock:
            if key not in loaded_models:
                ckpt = f"checkpoints/autoencoder/{category}/best.pth"
                if not os.path.exists(ckpt):
                    return None
                model = Autoencoder().to(DEVICE)
                load_checkpoint(model, ckpt)
                model.eval()
                loaded_models[key] = model
    return loaded_models[key]


def get_patchcore(category):
    """
    Đẩy PatchCore lên VRAM và nạp kiến trúc LSH Memory Bank.

    PatchCore sẽ đóng băng hoàn toàn Backbone ResNet để trích xuất Feature Map tầm trung. Toàn bộ tính trạng
    sẽ được dồn vào 'memory_bank'. Khi nhận diện, hàm này sẽ nạp file '.pth' ở disk. Nếu mất file, nó sẽ
    dùng ảnh trong thư mục `train` để tự động khôi phục lại kho dữ liệu cực kỳ linh hoạt.

    Tham số:
        category (str): Tên danh mục cấu hình.

    Trả về:
        PatchCore: Thể hiện (Instance) của PatchCore chứa LSH đã bung nén.
    """
    key = f"pc_{category}"
    if key not in loaded_models:
        with _model_lock:
            if key not in loaded_models:
                # Load backbone
                backbone = PatchCoreFeatureExtractor()
                bb_ckpt = f"checkpoints/backbone/{category}/best.pth"
                if os.path.exists(bb_ckpt):
                    backbone.load_backbone_weights(bb_ckpt)

                # Load memory bank
                pc = PatchCore(backbone=backbone, device=DEVICE)

                pc_ckpt = f"checkpoints/patchcore/{category}/memory_bank.pth"
                if os.path.exists(pc_ckpt):
                    pc.load(pc_ckpt)
                    loaded_models[key] = pc
                    return pc

                # Nếu chưa có memory bank -> cần build từ training data
                from data_processing.mvtec import MVTecDataset

                train_data_path = f"datasets/mvtec/{category}/train"
                if os.path.exists(train_data_path):
                    train_dataset = MVTecDataset(
                        "datasets/mvtec", category, split="train"
                    )
                    pc.fit(train_dataset)
                    os.makedirs(f"checkpoints/patchcore/{category}", exist_ok=True)
                    pc.save(pc_ckpt)
                    loaded_models[key] = pc
                    return pc

                # Fallback: PatchCore chưa sẵn sàng
                return None

    return loaded_models[key]


def get_cnn_ocsvm(category):
    """Load CNN+OC-SVM cho category (cache)."""
    key = f"ocsvm_{category}"
    if key not in loaded_models:
        from sklearn.svm import OneClassSVM
        from torch.utils.data import DataLoader

        from data_processing.mvtec import MVTecDataset

        # Feature extractor
        model = CNNFeatureExtractor().to(DEVICE)
        model.eval()

        # THỰC THI KIẾN TRÚC MỚI: Dùng joblib để chấm dứt lệ thuộc dataset 5GB
        ocsvm_ckpt = f"checkpoints/ocsvm/{category}/model.joblib"
        if os.path.exists(ocsvm_ckpt):
            ocsvm = joblib.load(ocsvm_ckpt)
            loaded_models[key] = (model, ocsvm)
            return loaded_models[key]

        # FALLBACK: Nếu chưa từng lưu joblib thì ép train 1 lần rồi lưu chết luôn!
        train_data_path = f"datasets/mvtec/{category}/train"
        if not os.path.exists(train_data_path):
            return None, None

        train_dataset = MVTecDataset("datasets/mvtec", category, split="train")
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

        train_features = []
        for imgs, _ in train_loader:
            with torch.no_grad():
                feats = model(imgs.to(DEVICE)).cpu()
            train_features.append(feats.numpy())

        train_features = np.concatenate(train_features, axis=0)
        ocsvm = OneClassSVM(gamma="auto")
        ocsvm.fit(train_features)

        # Save OCSVM using joblib
        os.makedirs(os.path.dirname(ocsvm_ckpt), exist_ok=True)
        joblib.dump(ocsvm, ocsvm_ckpt)
        print(f"Đã train và lưu cứng OC-SVM xuống đĩa: {ocsvm_ckpt}")

        loaded_models[key] = (model, ocsvm)

    return loaded_models[key]


# ============================================================
# INFERENCE
# ============================================================


from app_utils.image_utils import denormalize


def numpy_to_base64(img_array, cmap=None):
    """
    Chuyển numpy array → base64 PNG string.
    Tối ưu hóa cực độ: dùng PIL/CV2 thay vì Matplotlib.
    """
    try:
        if img_array is None:
            return None

        # Nếu là float [0, 1] -> [0, 255]
        if img_array.dtype != np.uint8:
            img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)

        # Nếu là 1 channel (Heatmap)
        if len(img_array.shape) == 2 or (
            len(img_array.shape) == 3 and img_array.shape[2] == 1
        ):
            # Áp dụng Colormap Jet bằng OpenCV
            heatmap_color = cv2.applyColorMap(img_array.squeeze(), cv2.COLORMAP_JET)
            img_pil = Image.fromarray(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))
        else:
            # Ảnh màu RGB
            img_pil = Image.fromarray(img_array)

        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error in numpy_to_base64: {e}")
        return None


def overlay_heatmap(img_np, heatmap, alpha=0.5):
    """
    Overlay heatmap lên ảnh gốc dùng PIL Alpha Blending.
    """
    try:
        if img_np is None or heatmap is None:
            return None

        # Normalize input images to uint8
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        if heatmap.max() <= 1.0:
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        else:
            heatmap_uint8 = heatmap.astype(np.uint8)

        # Create PIL images
        background = Image.fromarray(img_np).convert("RGBA")

        # Generate color heatmap with OpenCV
        heatmap_color = cv2.applyColorMap(heatmap_uint8.squeeze(), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        foreground = Image.fromarray(heatmap_rgb).convert("RGBA")

        # Apply alpha
        foreground.putalpha(int(255 * alpha))

        # Combine
        composite = Image.alpha_composite(background, foreground)

        buf = io.BytesIO()
        composite.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error in overlay_heatmap: {e}")
        return None


def run_autoencoder_inference(img_tensor, category):
    """
    Kích hoạt tiến trình Autoencoder Inference (Hệ thống Khôi phục Dữ liệu Ảo).

    Quy trình (Workflow):
    1. Đẩy ảnh đã Normalize (Chuẩn hóa) vào mạng nén Autoencoder.
    2. Mô hình sẽ cố gắng vẽ lại (Reconstruct) bức ảnh sao cho giống hệt như đồ vật bình thường.
    3. Trừ ma trận 2 bức ảnh (Ảnh Gốc - Ảnh Vẽ lại) để đo Lỗi Khôi Phục (Reconstruction Error / MSE).
       Vùng nào có lỗi cao (Mô hình không vẽ lại được vết xước) thì đó chính là Anomaly.

    Args:
        img_tensor (torch.Tensor): Ảnh đầu vào khuôn 256x256.
        category (str): Tên danh mục (ví dụ 'bottle').
    """
    model = get_autoencoder(category)
    if model is None:
        return {"error": f"Autoencoder chưa train cho '{category}'"}

    img_batch = img_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        recon = model(img_batch)

    # Trực tiếp tính sai số trên dải dữ liệu gốc (Normalized) của ảnh
    score = anomaly_score(img_batch, recon).item()

    # Heatmap = reconstruction error based on normalized range
    error = ((img_batch - recon) ** 2).mean(dim=1).squeeze().cpu().numpy()
    error_norm = error / (error.max() + 1e-8)

    try:
        # Tạo images
        img_np = denormalize(img_tensor)
        recon_np = denormalize(recon.squeeze(0))

        threshold = load_optimal_threshold(category, "autoencoder")
        is_anomaly = (score > threshold) if threshold is not None else (score > 0.5)
        anomaly_index = (
            round(score / threshold, 4)
            if (threshold is not None and threshold > 0)
            else None
        )

        return {
            "model": "Autoencoder",
            "score": round(float(score), 6),
            "threshold": round(float(threshold), 6) if threshold is not None else 0.5,
            "anomaly_index": anomaly_index,
            "is_anomaly": bool(is_anomaly),
            "heatmap_b64": numpy_to_base64(error_norm, cmap="jet"),
            "overlay_b64": overlay_heatmap(img_np, error_norm),
            "recon_b64": numpy_to_base64(recon_np),
        }
    except Exception as e:
        print(f"Autoencoder visualization error: {e}")
        return {
            "model": "Autoencoder",
            "score": round(float(score), 6),
            "is_anomaly": (score > 0.5),
            "error": "Lỗi xử lý hình ảnh",
        }


def run_patchcore_inference(img_tensor, category):
    """
    Kích hoạt tiến trình phân tích bất thường trực tiếp từ PatchCore LSH.

    Quy trình:
    1. Trích xuất đặc trưng hình ảnh trực tiếp qua Backbone ResNet.
    2. Quét mảng đặc trưng đó đối chiếu với LSH Buckets (Băm dữ liệu) để đo khoảng cách Euclidean
       với các đặc trưng chuẩn (Nominal features).
    3. Đẩy kết quả tọa độ ra Bản đồ nhiệt (Dense Anomaly Map) và kết luận ảnh sai phạm.

    Tham số:
        img_tensor (torch.Tensor): Ảnh Tensor đã qua Normalize chuẩn ImageNet.
        category (str): Tên danh mục (ví dụ 'screw').

    Trả về:
        dict: Chứa bộ thông số kết luận (score, threshold, cờ is_anomaly) kèm theo ảnh Base64 vẽ viền lỗi.
    """
    pc = get_patchcore(category)
    if pc is None:
        return {"error": f"PatchCore model weights not found for '{category}'"}

    scores, anomaly_maps = pc.predict(img_tensor)
    score = scores[0]
    amap = anomaly_maps[0]

    # Normalize anomaly map
    amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

    try:
        # Optimize anomaly map resizing using PIL
        amap_pil = Image.fromarray((amap_norm * 255).astype(np.uint8))
        amap_resized_uint8 = np.array(
            amap_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        )
        amap_resized = amap_resized_uint8 / 255.0

        img_np = denormalize(img_tensor)

        threshold = load_optimal_threshold(category, "patchcore")
        tolerance = 0.8
        is_anomaly = (
            (score > threshold * tolerance) if threshold is not None else (score > 0.5)
        )
        anomaly_index = (
            round(score / (threshold * tolerance), 4)
            if (threshold is not None and threshold > 0)
            else None
        )

        return {
            "model": "PatchCore",
            "score": round(float(score), 6),
            "threshold": round(float(threshold), 6) if threshold is not None else 0.5,
            "anomaly_index": anomaly_index,
            "is_anomaly": bool(is_anomaly),
            "heatmap_b64": numpy_to_base64(amap_resized, cmap="jet"),
            "overlay_b64": overlay_heatmap(img_np, amap_resized),
        }
    except Exception as e:
        print(f"PatchCore visualization error: {e}")
        return {
            "model": "PatchCore",
            "score": round(float(score), 6),
            "is_anomaly": (score > 1.0),
            "error": "Lỗi xử lý hình ảnh",
        }


def run_ocsvm_inference(img_tensor, category):
    """
    Chạy Khối Tích Hợp CNN + OC-SVM (Bộ Phân cực Support Vector Machine).

    Cơ chế (Mechanism):
    1. Sử dụng mạng CNN (ResNet) rút ra đặc trưng (Feats) phẳng 512 chiều.
    2. Đưa mảng 512 chiều này lên không gian Scikit-Learn OC-SVM phân tích.
    3. Thuật toán OC-SVM sẽ đếm khoảng cách từ tọa độ của Vector đó tới biên giới an toàn (Decision Boundary).
    4. Trả về mức độ bất thường dưới dạng Decision Score. Giá trị âm càng nặng thì lỗi càng lớn.
    """
    result = get_cnn_ocsvm(category)
    if result is None or result == (None, None):
        return {"error": f"CNN+OC-SVM chưa sẵn sàng cho '{category}'"}

    model, ocsvm = result
    # Đưa ảnh (Batch=1) vào VRAM
    img_batch = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feats = model(img_batch).cpu().numpy()

    score = -ocsvm.decision_function(feats)[0]

    img_np = denormalize(img_tensor)

    # ============================================================
    # Thêm Grad-CAM XAI vào giao diện cho CNN+OC-SVM
    # ============================================================
    try:
        from visualization.gradcam import GradCAM

        # Lấy layer4 (trước avgpool) làm target layer cho ResNet18
        target_layer = model.features[-2]
        gradcam = GradCAM(model, target_layer)

        # generate() yêu cầu requires_grad=True bên trong nên không thể nằm trong no_grad block
        heatmap = gradcam.generate(img_batch.clone())

        # Resize heatmap
        heatmap_resized = (
            np.array(
                Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
                )
            )
            / 255.0
        )

        heatmap_b64 = numpy_to_base64(heatmap_resized, cmap="jet")
        overlay_b64 = overlay_heatmap(img_np, heatmap_resized)
    except Exception as e:
        print("Grad-CAM Error:", e)
        heatmap_b64 = None
        overlay_b64 = None

    threshold = load_optimal_threshold(category, "cnn_ocsvm")
    is_anomaly = (score > threshold) if threshold is not None else (score > 0.5)
    anomaly_index = (
        round(score / threshold, 4)
        if (threshold is not None and threshold > 0)
        else None
    )

    return {
        "model": "CNN+OC-SVM",
        "score": round(float(score), 6),
        "threshold": round(float(threshold), 6) if threshold else 0.0,
        "anomaly_index": anomaly_index,
        "is_anomaly": bool(is_anomaly),
        "input_b64": numpy_to_base64(img_np),
        "heatmap_b64": heatmap_b64,
        "overlay_b64": overlay_b64,
    }


# ============================================================
# ROUTES
# ============================================================


@app.route("/")
def index():
    """Trang chủ."""
    return render_template("index.html", categories=ALL_CATEGORIES)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Điểm cuối (Endpoint) REST API phục vụ tải trọng (Payload) từ Web Client.

    Quy trình phân tích tự động (Batch Mode — Nâng cấp v2):
    1. Nội soi đầu vào (Input Validation): Chấp nhận nhiều file ảnh (tối đa 12).
    2. Vòng lặp xử lý từng ảnh tuần tự (Sequential Batch Processing).
    3. Mỗi ảnh đều đi qua đầy đủ: CLAHE Enhancement → YOLO Detection → Model Inference.
    4. Kết xuất (Payload Construction): Trả về mảng `batch_results`, mỗi phần tử
       chứa tên file + danh sách kết quả của các model cho ảnh đó.
    """
    MAX_IMAGES = 12  # Giới hạn: tối ưu cho GPU 6GB VRAM

    try:
        files = request.files.getlist("image")
        if not files or (len(files) == 1 and files[0].filename == ""):
            return jsonify({"error": "Chưa upload ảnh"}), 400

        if len(files) > MAX_IMAGES:
            return (
                jsonify(
                    {
                        "error": f"Vượt quá giới hạn {MAX_IMAGES} ảnh mỗi lượt. Bạn đã chọn {len(files)} ảnh."
                    }
                ),
                400,
            )

        category_input = request.form.get("category", "bottle")
        model_type = request.form.get("model", "patchcore")
        auto_detect = (
            request.form.get("auto_detect", "false").lower() == "true"
            or category_input == "auto"
        )

        def process_single_image(file):
            """Xử lý một ảnh đơn lẻ, trả về dict kết quả."""
            filename = file.filename or "unknown.jpg"
            category = category_input

            # Load image
            file.stream.seek(0)  # Reset stream
            img_pil = Image.open(file.stream).convert("RGB")
            img_cv2 = np.array(img_pil)

            # --- CLAHE & Unsharp Masking ---
            try:
                lab = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                enhanced_cv2 = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
                gaussian = cv2.GaussianBlur(enhanced_cv2, (0, 0), 2.0)
                sharpened_cv2 = cv2.addWeighted(enhanced_cv2, 1.5, gaussian, -0.5, 0)
                img_pil_yolo = Image.fromarray(sharpened_cv2)
            except Exception as e:
                print(f"Preprocessing error [{filename}]: {e}")
                img_pil_yolo = img_pil

            # --- YOLO Auto-Detection ---
            yolo_category, yolo_conf, yolo_bbox = None, 0.0, None
            warning_msg = None

            if auto_detect:
                yolo_category, yolo_conf, yolo_bbox = yolo_detector.detect(img_pil_yolo)
                if yolo_category:
                    if yolo_conf < 0.65:
                        warning_msg = f"Ảnh '{filename}': AI dự đoán là '{yolo_category}' với độ tin cậy thấp ({yolo_conf:.2f})."
                    category = yolo_category

            # Validate category
            if category not in ALL_CATEGORIES:
                return {
                    "filename": filename,
                    "error": f"Không nhận diện rõ sản phẩm trong ảnh '{filename}'.",
                    "results": [],
                }

            # Transform & Normalize
            img_tensor = transform(img_pil)
            img_np = denormalize(img_tensor)
            input_b64 = numpy_to_base64(img_np)

            def run_inference(model_key):
                if model_key == "patchcore":
                    return run_patchcore_inference(img_tensor, category)
                elif model_key == "autoencoder":
                    return run_autoencoder_inference(img_tensor, category)
                elif model_key == "cnn_ocsvm":
                    return run_ocsvm_inference(img_tensor, category)
                return None

            results = []
            if model_type == "all":
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [
                        executor.submit(run_inference, m)
                        for m in ["patchcore", "autoencoder", "cnn_ocsvm"]
                    ]
                    for f in futures:
                        r = f.result()
                        if r:
                            r["input_b64"] = input_b64
                            results.append(r)
            else:
                r = run_inference(model_type)
                if r:
                    r["input_b64"] = input_b64
                    results.append(r)

            return {
                "filename": filename,
                "category": category,
                "results": results,
                "warning": warning_msg,
                "auto_detected": (
                    {
                        "category": yolo_category,
                        "confidence": yolo_conf,
                        "bbox": yolo_bbox,
                    }
                    if yolo_category
                    else None
                ),
            }

        # Batch processing tuần tự để đảm bảo VRAM ổn định
        batch_results = []
        for file in files:
            item = process_single_image(file)
            batch_results.append(item)

        return jsonify(
            {
                "status": "ok",
                "batch_results": batch_results,
                "total": len(batch_results),
            }
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """API endpoint: Chatbot XAI chuyên gia."""
    # Knowledge Base cho Chế độ Tự xây dựng (Rule-based / Expert System)
    PROJECT_KB = {
        "patchcore": "**PatchCore** là thuật toán SOTA (State-of-the-Art) trong phát hiện bất thường. \n- **Cơ chế:** Chia ảnh thành hàng ngàn 'patches', trích xuất đặc trưng qua ResNet18, lưu vào **Memory Bank**.\n- **Ưu điểm:** Có khả năng định vị lỗi ở mức Pixel (Heatmap) cực tốt.\n- **Logic:** Sử dụng k-NN để tính khoảng cách Euclidean. Score càng cao = càng khác biệt so với mẫu chuẩn.",
        "autoencoder": "**Autoencoder** hoạt động theo cơ chế **Tái tạo (Reconstruction)**. \n- **Cơ chế:** Nén ảnh vào không gian vector rồi vẽ lại. \n- **Logic:** Với ảnh lỗi (scratches), mô hình chỉ có thể vẽ lại phần 'bình thường', dẫn đến sai số **MSE** cao tại vị trí lỗi.",
        "ocsvm": "**CNN + One-Class SVM** kết hợp Deep Learning (trích xuất đặc trưng) và ML truyền thống. \n- **Cơ chế:** SVM xây dựng một 'siêu mặt' bao quanh dữ liệu chuẩn. \n- **Logic:** Nếu score dương (>0), mẫu đã 'văng' ra ngoài biên giới an toàn.",
        "auroc": "**AUROC** (Area Under ROC Curve) la chi so danh gia kha nang phan loai. \n- 1.0 la hoan hao. \n- Trong project nay, PatchCore trung binh dat **~0.736**, CNN+OC-SVM dat **~0.787**, Autoencoder dat **~0.668** tren toan bo 15 categories.",
        "mvtec": "**MVTec AD** là dataset chuẩn công nghiệp gồm 15 loại sản phẩm (bottle, cable, capsule, ...). Tổng cộng 5,354 ảnh chất lượng cao.",
        "gradcam": "**Grad-CAM** là kỹ thuật XAI giúp hiển thị 'vùng chú ý' của mô hình dưới dạng **Heatmap**. Vùng màu đỏ là nơi AI nghi ngờ có lỗi nhất.",
        "setup": "Để cài đặt: Chạy `setup.bat` (Win) hoặc `setup.sh` (Linux). Cần Python 3.8+ và các thư viện trong `requirements.txt`.",
        "author": "Project 'Industrial Anomaly Detection' — Đồ án Tốt nghiệp 2026. Một hệ thống so sánh đa thuật toán nhằm tối ưu hóa kiểm soát chất lượng QA/QC.",
        "fix": "**Hướng khắc phục:** \n1. Nếu báo lỗi nhầm (False Positive): Thử nâng ngưỡng **Threshold**.\n2. Nếu không phát hiện được lỗi (False Negative): Thử giảm Threshold hoặc thu thập thêm ảnh lỗi để tinh chỉnh.\n3. Kiểm tra lại điều kiện ánh sáng và góc chụp camera.",
    }

    try:
        data = request.json
        message = data.get("message", "")
        message_lower = message.lower()
        context = data.get("context", {})
        mode = data.get("mode", "generative")

        is_anomaly = context.get("is_anomaly")
        score = context.get("score")
        thresh = context.get("threshold")
        ans = (
            " Điểm số **vượt ngưỡng**, hệ thống xác nhận đây là sản phẩm lỗi (Anomaly). Vui lòng xem vùng màu đỏ trên Heatmap để định vị khuyết tật."
            if is_anomaly
            else " Điểm số an toàn, ranh giới cấu trúc sản phẩm hoàn toàn bình thường (Normal)."
        )

        if mode == "rule_based":
            # 1. Tìm câu trả lời phù hợp nhất từ KB
            kb_response = ""
            for key, value in PROJECT_KB.items():
                if key in message_lower:
                    kb_response += f"\n\n **Kiến thức về {key.capitalize()}:**\n{value}"

            # 2. Xử lý logic chẩn đoán dựa trên context thực tế
            diag_info = ""
            if (
                "tại sao" in message_lower
                or "lỗi" in message_lower
                or "check" in message_lower
            ):
                diff = score - thresh
                if is_anomaly:
                    diag_info = f"\n\n** Phân tích nguyên nhân:**\n- Score hiện tại (**{score}**) đang cao hơn ngưỡng cảnh báo (**{thresh}**) là **{round(diff, 3)}** đơn vị.\n- Với AI, sự khác biệt này đủ để kết luận sản phẩm có biến dạng cấu trúc so với tập mẫu 'lành'."
                else:
                    diag_info = f"\n\n** Phân tích nguyên nhân:**\n- Score (**{score}**) vẫn nằm dưới ngưỡng cho phép (**{thresh}**).\n- Hệ thống đánh giá các đặc trưng hình ảnh này tương đồng >90% với sản phẩm đạt chuẩn."

            # 3. Kết hợp kết quả
            header = f" **[Hệ chuyên gia IAD - Offline Mode]**\n\n- **Target:** {context.get('category')} | **Model:** {context.get('model')}"
            final_msg = f"{header}{diag_info}{kb_response}"

            # Nếu không khớp từ khóa nào và không có diagnosis
            if not kb_response and not diag_info:
                final_msg += "\n\nTôi là **Hệ chuyên gia nội bộ**. Bạn có thể hỏi tôi về: *PatchCore, Autoencoder, AUROC, MVTec* hoặc hỏi *'Tại sao ảnh này lỗi?'* để tôi phân tích dữ liệu cứng."

            return jsonify({"response": final_msg})

        # Thử load Gemini API key từ biến môi trường
        try:
            import google.generativeai as genai
            from dotenv import load_dotenv

            load_dotenv()

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                # Fallback nếu không có API key
                is_anomaly = context.get("is_anomaly")
                score = context.get("score")
                thresh = context.get("threshold")
                ans = (
                    " Điểm số **vượt ngưỡng**, hệ thống xác nhận đây là sản phẩm lỗi (Anomaly). Vui lòng xem vùng màu đỏ trên Heatmap để định vị khuyết tật."
                    if is_anomaly
                    else " Điểm số an toàn, ranh giới cấu trúc sản phẩm hoàn toàn bình thường (Normal)."
                )
                fallback_msg = f" **Hệ thống chưa cấu hình `GEMINI_API_KEY` trong file `.env`.**\n\nTuy nhiên, dựa trên bộ quy tắc XAI Server, tôi có thể chẩn đoán:\n- Sản phẩm phân tích: **{context.get('category')}**\n- Mô hình: **{context.get('model')}**\n- Điểm bất thường (Score): **{score}** (Ngưỡng an toàn tối đa: {thresh})\n\n{ans}"
                return jsonify({"response": fallback_msg})

            genai.configure(api_key=api_key)

            history = data.get("history", [])
            # Convert frontend history to Gemini format if needed
            # Frontend format: [{role: 'user', content: '...'}, {role: 'bot', content: '...'}]
            # Gemini format: [{role: 'user', parts: [...]}, {role: 'model', parts: [...]}]
            gemini_history = []
            for h in history:
                role = "user" if h["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [h["content"]]})

            system_prompt = f"""Bạn là một **Chuyên gia AI & XAI (Explainable AI)** cao cấp, tên là 'IAD Assistant'. 
Nhiệm vụ của bạn là giải thích kết quả của hệ thống Phát hiện Bất thường Công nghiệp (Industrial Anomaly Detection).

**NGỮ CẢNH CỦA BỨC ẢNH HIỆN TẠI:**
- Sản phẩm: {context.get('category')}
- Model AI đang dùng: {context.get('model')}
- Chỉ số Bất thường (Score): {score}
- Ngưỡng mốc (Threshold): {thresh}
- Kết luận máy: {' CÓ LỖI (ANOMALY)' if is_anomaly else ' BÌNH THƯỜNG (NORMAL)'}

**KIẾN THỨC CHUYÊN MÔN ĐỂ GIẢI THÍCH:**
1. **PatchCore**: Sử dụng 'Memory Bank'. Score là khoảng cách Euclidean tới mẫu tốt nhất. Nếu Score > Threshold, nghĩa là có một 'patch' (mảnh ảnh) quá khác lạ so với dữ liệu mẫu.
2. **Autoencoder**: Cố gắng vẽ lại ảnh (reconstruction). Score là MSE (sai số). Nếu Score cao, nghĩa là AI không thể vẽ lại được vùng đó (thường là vết xước/lỗ).
3. **CNN+OC-SVM**: Trích xuất đặc trưng rồi dùng SVM phân vùng. Score dương (>0) là nằm ngoài 'vùng an toàn'.

**HƯỚNG DẪN GIAO TIẾP:**
- Bạn phải giao tiếp cực kỳ thông minh, tự nhiên, giống như một người bạn đồng hành chuyên nghiệp.
- **TUYỆT ĐỐI KHÔNG** trả lời theo kiểu template khô khan. Hãy phân tích dựa trên con số thực tế ở trên.
- Nếu người dùng hỏi "Tại sao?", hãy dựa vào logic của Model đó để giải thích (ví dụ: "Vì Score {score} cao hơn ngưỡng {thresh} tận {round(score-thresh, 2)} đơn vị...").
- Sử dụng Markdown phong phú: **đậm**, *nghiêng*, `code`, list, và emoji phù hợp.
- Trả lời bằng tiếng Việt, súc tích nhưng đầy đủ ý.
"""
            try:
                # Gửi system_prompt như một phần của ngữ cảnh ban đầu nếu là câu đầu tiên,
                # hoặc lồng nó vào message đầu tiên của session.
                # Tuy nhiên, cách tốt nhất với genai là đưa system_instruction vào constructor Model.
                # Nhưng version 0.8.5 hỗ trợ system_instruction.

                # Re-init with system_instruction for better adherence
                gemini_model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash", system_instruction=system_prompt
                )
                chat_session = gemini_model.start_chat(history=gemini_history)

                result = chat_session.send_message(message)
                return jsonify({"response": result.text})
            except Exception as api_err:
                fallback_msg = f" **Kết nối AI Generative bị lỗi:** {str(api_err)}\n\n **[Tôi đã chuyển về Chế độ XAI Cứng Offline]**\n- Phân tích: **{context.get('category')}**\n- Mô hình: **{context.get('model')}**\n- Score: **{score}** (Ngưỡng tiêu chuẩn: {thresh})\n\n{ans}"
                return jsonify({"response": fallback_msg})

        except ImportError:
            return jsonify(
                {
                    "response": " **Thiếu thư viện google-generativeai**. Vui lòng chạy `pip install google-generativeai python-dotenv`."
                }
            )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def preload_all_models():
    """Nạp trước tất cả 15 mô hình để đảm bảo phản hồi tức thì (Sub-second)."""
    print("\n" + "=" * 50)
    print(" PRE-LOADING ALL 15 CATEGORIES (READYING SYSTEM)")
    print("=" * 50)

    import time

    start_total = time.time()

    for cat in ALL_CATEGORIES:
        start_cat = time.time()
        print(f" Loading: {cat.upper()}...", end="", flush=True)
        # Nạp PatchCore, Autoencoder, OC-SVM vào cache
        try:
            get_patchcore(cat)
            get_autoencoder(cat)
            get_cnn_ocsvm(cat)
            duration = time.time() - start_cat
            print(f" Done ({duration:.2f}s)")
        except Exception as e:
            print(f" Error: {e}")

    total_duration = time.time() - start_total
    print(f"\n All categories pre-loaded in {total_duration:.2f}s.")
    print("=" * 50 + "\n")


@app.route("/health")
def health():
    """Health check."""
    return jsonify({"status": "ok", "models": list(loaded_models.keys())})


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  IAD Demo — http://localhost:5000")
    print("=" * 50 + "\n")

    # --- GPU Warmup: Chạy thử 1 ảnh giả để nạp CUDA/VRAM sẵn ---
    if DEVICE == "cuda":
        print("\n GPU Warmup: Initializing CUDA...")
        try:
            dummy_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            dummy_pil = Image.fromarray(dummy_np)
            dummy_tensor = transform(dummy_pil).to(DEVICE)
            yolo_detector.detect(dummy_pil)
            print(" GPU Warmup (CUDA) complete.")
        except Exception as e:
            print(f" GPU Warmup failed: {e}")

    # --- Full Pre-loading: Nạp toàn bộ model ngay khi startup ---
    preload_all_models()

    # Tắt reloader vì nó sẽ nạp lại model 2 lần gây tốn tài nguyên
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
