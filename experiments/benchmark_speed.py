import io
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import (
    DEVICE,
    IMAGE_SIZE,
    numpy_to_base64,
    run_autoencoder_inference,
    run_ocsvm_inference,
    run_patchcore_inference,
)
from app_utils.yolo_detector import YOLODetector


def benchmark():
    print(f" Bắt đầu Benchmark hiệu năng trên thiết bị: {DEVICE}")

    # 1. Warmup
    print(" Đang làm nóng GPU...")
    yolo = YOLODetector(device=DEVICE)
    dummy_img = Image.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
    for _ in range(5):
        yolo.detect(dummy_img)

    # 2. Benchmark data
    category = "bottle"
    # Ensure models are loaded (first call loads them)
    print(f" Đang nạp mô hình cho category '{category}'...")
    img_tensor = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE)
    run_patchcore_inference(img_tensor, category)
    run_autoencoder_inference(img_tensor, category)
    run_ocsvm_inference(img_tensor, category)

    results = []

    models = [
        ("PatchCore", run_patchcore_inference),
        ("Autoencoder", run_autoencoder_inference),
        ("CNN+OC-SVM", run_ocsvm_inference),
    ]

    num_runs = 20

    for name, inference_func in models:
        print(f" Đang chạy benchmark: {name} ({num_runs} vòng)...")

        times_detection = []
        times_inference = []
        times_viz = []
        times_total = []

        for _ in range(num_runs):
            start_total = time.time()

            # Step 1: Detection
            start_det = time.time()
            yolo_cat, conf, bbox = yolo.detect(dummy_img)
            times_detection.append(time.time() - start_det)

            # Step 2: Model Inference
            start_inf = time.time()
            res = inference_func(img_tensor, category)
            times_inference.append(time.time() - start_inf)

            # Step 3: Visualization (Base64 encoding)
            start_viz = time.time()
            # Simulate the work done in app.py after inference
            if "anomaly_map" in res:
                numpy_to_base64(res["anomaly_map"])
            times_viz.append(time.time() - start_viz)

            times_total.append(time.time() - start_total)

        avg_det = np.mean(times_detection) * 1000
        avg_inf = np.mean(times_inference) * 1000
        avg_viz = np.mean(times_viz) * 1000
        avg_total = np.mean(times_total) * 1000

        results.append(
            {
                "Model": name,
                "Detection (ms)": f"{avg_det:.2f}",
                "Inference (ms)": f"{avg_inf:.2f}",
                "Visualization (ms)": f"{avg_viz:.2f}",
                "Total E2E (ms)": f"{avg_total:.2f}",
                "FPS": f"{1000/avg_total:.1f}",
            }
        )

    # Output Table
    print("\n" + "=" * 80)
    print(
        f"{'Model':<15} | {'Detect (ms)':<12} | {'Infer (ms)':<12} | {'Viz (ms)':<10} | {'Total (ms)':<12} | {'FPS':<6}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['Model']:<15} | {r['Detection (ms)']:<12} | {r['Inference (ms)']:<12} | {r['Visualization (ms)']:<10} | {r['Total E2E (ms)']:<12} | {r['FPS']:<6}"
        )
    print("=" * 80)
    print(" Lưu ý: Kết quả trên đã bao gồm tối ưu hóa Phase 6 (PIL/CV2 & GPU).")


if __name__ == "__main__":
    benchmark()
