"""
Quality Assurance & Evaluation Metrics Suite (Bộ Công Cụ Đo Lường Hiệu Năng & Đảm Bảo Chất Lượng)

This module provides a comprehensive suite of statistical metrics specifically formulated
(Mô-đun này cung cấp một bộ số liệu thống kê toàn diện được xây dựng dành riêng)
for Industrial Anomaly Detection evaluation. It computes robust thresholds and validates
(cho Đánh giá Phát hiện Lỗi Bất Thường Công nghiệp. Nó tính toán các Ngưỡng Tối Ưu và xác thực)
model performance at both the image and pixel level, critical for preventing financial
(hiệu suất mô hình ở cả cấp độ Toàn Ảnh và cấp độ Điểm Ảnh, rất quan trọng để ngăn ngừa thiệt hại tài chính)
losses associated with manufacturing defects (False Negatives - Bỏ sót Lỗi) and unnecessary downtime (False Positives - Báo Động Giả).

IMAGE-LEVEL METRICS (Các Chỉ Số Thống Kê Cấp Độ Ảnh):
- anomaly_score: Computes the Mean Squared Error (MSE) reconstruction loss for Autoencoder deployments.
  (Tính hiệu số rủi ro thất thoát tái tạo hình ảnh).
- compute_auroc: Area Under ROC Curve (Diện tích Dưới Đường Cong Hiệu Suất ROC) - Đánh giá năng lực xếp hạng toàn cầu không phụ thuộc ngưỡng.
- compute_average_precision: Area Under Precision-Recall Curve (AUPRC) - Cực kỳ mạnh mẽ chống lại sự thiên lệch nhãn (Class Imbalance).
- compute_f1_precision_recall: F1, Precision (Độ chính xác), and Recall (Độ phủ) tại Ngưỡng Thống Kê Tối Ưu.
- compute_confusion_matrix: Extracts True Positives (TP - Lỗi bắt đúng), False Positives (FP - Báo nhầm), True Negatives (TN - Bình thường an toàn), and False Negatives (FN - Lọt vành đai).
- compute_specificity: Measures True Negative Rate (Correct rejection of nominal samples - Độ nhạy từ chối ảnh chuẩn).
- find_optimal_threshold: Maximizes Youden's J statistic (TPR - FPR) to locate the peak separation threshold (Tìm Ngưỡng Cắt lý tưởng).

PIXEL-LEVEL METRICS (Chỉ Số Đánh Giá Ở Cấp Độ Điểm Ảnh):
- compute_pixel_auroc: Evaluates localization accuracy by comparing scalar maps against ground-truth segmentation masks.
  (Đánh giá độ chính xác vùng định vị Lỗi so với Bản đồ chuẩn).
- compute_pro_score: Per-Region Overlap (PRO) - The standard industrial metric from the MVTec AD paper.
  (Tỷ lệ Diện Tích Vùng Lỗi Bao Phủ). Bất chấp kích cỡ lớn hay nhỏ, mọi vùng Lỗi (Component) đều được định giá cân bằng.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ============================================================
# 1. ANOMALY SCORE — Tính điểm bất thường từ reconstruction error
# ============================================================


def anomaly_score(input_img, recon_img):
    """
    Tính anomaly score bằng MSE giữa ảnh gốc và ảnh reconstruct.

    Ý tưởng: Autoencoder được train trên ảnh bình thường → reconstruct tốt ảnh normal.
    Với ảnh bất thường → reconstruct kém → MSE cao → anomaly score cao.

    Tính toán:
        error = (input - recon)^2       → pixel-wise MSE
        score = mean(error, dim=C,H,W)  → 1 scalar cho mỗi ảnh

    Args:
        input_img: ảnh gốc (B, C, H, W)
        recon_img: ảnh reconstruct từ Autoencoder (B, C, H, W)

    Returns:
        scores: anomaly score cho mỗi ảnh trong batch (B,)
                Giá trị càng cao → khả năng bất thường càng lớn
    """
    # reduction="none" → giữ nguyên shape, không mean trước
    error = F.mse_loss(recon_img, input_img, reduction="none")

    # Mean qua channels (C), height (H), width (W)
    # → mỗi ảnh có 1 scalar score
    scores = error.mean(dim=[1, 2, 3])

    return scores


# ============================================================
# 2. AUROC — Area Under ROC Curve
# ============================================================


def compute_auroc(labels, scores):
    """
    Tính Area Under ROC Curve (AUROC).

    ROC Curve vẽ True Positive Rate vs False Positive Rate ở các ngưỡng khác nhau.
    AUROC = diện tích dưới đường cong ROC.

    Ý nghĩa:
        - 1.0 = phân biệt hoàn hảo (tất cả anomaly có score > tất cả normal)
        - 0.5 = random guess (model không phân biệt được gì)
        - < 0.5 = tệ hơn random (model đảo ngược, coi normal là anomaly)

    Khi nào dùng:
        - Đánh giá tổng quan khả năng xếp hạng của model
        - Không phụ thuộc vào ngưỡng cụ thể

    Hạn chế:
        - Không phản ánh tốt khi dữ liệu rất mất cân bằng
          (ví dụ: 99% normal, 1% anomaly → AUROC có thể cao nhưng thực tế detect kém)

    Args:
        labels: ground truth (0 = normal, 1 = anomaly)
        scores: anomaly scores (càng cao = càng khả năng anomaly)

    Returns:
        auroc: float (0.0 → 1.0)
    """
    labels, scores = _to_numpy(labels, scores)
    return roc_auc_score(labels, scores)


# ============================================================
# 3. AVERAGE PRECISION (AP) — Tốt hơn AUROC cho dữ liệu mất cân bằng
# ============================================================


def compute_average_precision(labels, scores):
    """
    Tính Average Precision (AP), còn gọi là AUPRC
    (Area Under Precision-Recall Curve).

    Precision-Recall Curve vẽ Precision vs Recall ở các ngưỡng khác nhau.
    AP = diện tích dưới đường cong PR.

    Ý nghĩa:
        - 1.0 = phát hiện hoàn hảo (precision = 1, recall = 1 ở mọi ngưỡng)
        - Giá trị baseline = tỷ lệ anomaly trong dataset
          (ví dụ: 10% anomaly → random AP ≈ 0.1)

    Khi nào dùng:
        - Khi dữ liệu MẤT CÂN BẰNG (ít anomaly, nhiều normal)
          → AP phản ánh chính xác hơn AUROC
        - Trong anomaly detection thực tế, luôn dùng AP bên cạnh AUROC

    Tại sao tốt hơn AUROC cho dữ liệu mất cân bằng:
        - AUROC dùng FPR = FP/(FP+TN). Khi TN rất lớn, FPR nhỏ dù FP lớn
          → AUROC "lạc quan" giả
        - AP dùng Precision = TP/(TP+FP), không bị ảnh hưởng bởi TN

    Args:
        labels: ground truth (0 = normal, 1 = anomaly)
        scores: anomaly scores

    Returns:
        ap: float (0.0 → 1.0)
    """
    labels, scores = _to_numpy(labels, scores)
    return average_precision_score(labels, scores)


# ============================================================
# 4. OPTIMAL THRESHOLD — Tìm ngưỡng tối ưu
# ============================================================


def find_optimal_threshold(labels, scores):
    """
    Tìm ngưỡng (threshold) tối ưu để phân loại normal vs anomaly.

    Phương pháp: dùng Youden's J statistic trên ROC curve.
    J = TPR - FPR → maximize J = tìm điểm xa đường chéo nhất.

    Ý nghĩa:
        - Ngưỡng tối ưu cân bằng giữa phát hiện đúng anomaly (TPR cao)
          và tránh báo nhầm normal (FPR thấp)
        - Score > threshold → predict anomaly
        - Score <= threshold → predict normal

    Args:
        labels: ground truth (0 = normal, 1 = anomaly)
        scores: anomaly scores

    Returns:
        optimal_threshold: float — ngưỡng tối ưu
    """
    labels, scores = _to_numpy(labels, scores)

    # ROC curve: trả về (FPR, TPR, thresholds) ở nhiều ngưỡng
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Youden's J = TPR - FPR
    # Tìm ngưỡng có J lớn nhất = cân bằng tốt nhất
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)

    return thresholds[best_idx]


# ============================================================
# 5. F1, PRECISION, RECALL — Metrics tại ngưỡng cụ thể
# ============================================================


def compute_f1_precision_recall(labels, scores, threshold=None):
    """
    Tính F1 Score, Precision, Recall tại 1 ngưỡng.

    Nếu không truyền threshold → tự tìm ngưỡng tối ưu.

    Giải thích các metrics:

    PRECISION = TP / (TP + FP)
        "Trong số ảnh model nói là anomaly, bao nhiêu % thực sự anomaly?"
        - Precision cao → ít false alarm (ít báo nhầm)
        - Quan trọng khi: chi phí kiểm tra false alarm cao
          (ví dụ: dừng dây chuyền sản xuất để kiểm tra → tốn tiền)

    RECALL (Sensitivity / True Positive Rate) = TP / (TP + FN)
        "Trong số tất cả ảnh anomaly thực tế, model phát hiện được bao nhiêu %?"
        - Recall cao → ít bỏ sót anomaly
        - Quan trọng khi: bỏ sót anomaly gây hậu quả nghiêm trọng
          (ví dụ: linh kiện lỗi đưa vào sản phẩm → nguy hiểm)

    F1 SCORE = 2 × (Precision × Recall) / (Precision + Recall)
        "Trung bình hài hòa giữa Precision và Recall"
        - F1 cao khi CẢ HAI Precision và Recall đều cao
        - F1 = 1.0 → hoàn hảo
        - F1 = 0.0 → hoàn toàn sai

    Args:
        labels: ground truth (0 = normal, 1 = anomaly)
        scores: anomaly scores
        threshold: ngưỡng phân loại (None = tự tìm tối ưu)

    Returns:
        dict: {"f1": float, "precision": float, "recall": float, "threshold": float}
    """
    labels, scores = _to_numpy(labels, scores)

    # Tìm ngưỡng tối ưu nếu chưa có
    if threshold is None:
        threshold = find_optimal_threshold(labels, scores)

    # Áp dụng ngưỡng: score > threshold → predict anomaly (1)
    predictions = (scores >= threshold).astype(int)

    return {
        "f1": f1_score(labels, predictions, zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "threshold": float(threshold),
    }


# ============================================================
# 6. SPECIFICITY — Tỷ lệ nhận đúng ảnh bình thường
# ============================================================


def compute_specificity(labels, scores, threshold=None):
    """
    Tính Specificity (True Negative Rate).

    SPECIFICITY = TN / (TN + FP)
        "Trong số tất cả ảnh bình thường, model nhận ra đúng bao nhiêu %?"
        - Specificity cao → model không báo nhầm normal thành anomaly
        - Bổ sung cho Recall: Recall đo anomaly, Specificity đo normal

    Ví dụ thực tế:
        - Recall = 0.95: model phát hiện 95% sản phẩm lỗi
        - Specificity = 0.90: model xác nhận đúng 90% sản phẩm tốt
        → 10% sản phẩm tốt bị kiểm tra lại (false alarm)

    Args:
        labels: ground truth (0 = normal, 1 = anomaly)
        scores: anomaly scores
        threshold: ngưỡng (None = tự tìm tối ưu)

    Returns:
        specificity: float (0.0 → 1.0)
    """
    labels, scores = _to_numpy(labels, scores)

    if threshold is None:
        threshold = find_optimal_threshold(labels, scores)

    predictions = (scores >= threshold).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return specificity


# ============================================================
# 7. CONFUSION MATRIX — Ma trận nhầm lẫn
# ============================================================


def compute_confusion_matrix(labels, scores, threshold=None):
    """
    Tính Confusion Matrix (Ma trận nhầm lẫn).

    Ma trận 2x2 cho biết model phân loại đúng/sai bao nhiêu:

                        Predicted Normal    Predicted Anomaly
    Actual Normal       TN (True Neg)       FP (False Pos)
    Actual Anomaly      FN (False Neg)      TP (True Pos)

    Giải thích:
        - TN (True Negative): ảnh normal, model nói normal
        - FP (False Positive): ảnh normal, model nói anomaly  (false alarm)
        - FN (False Negative): ảnh anomaly, model nói normal  (bỏ sót!)
        - TP (True Positive): ảnh anomaly, model nói anomaly

    Trong sản xuất:
        - FP cao → dừng dây chuyền nhiều lần vô ích → tốn tiền
        - FN cao → sản phẩm lỗi lọt qua → nguy hiểm!

    Args:
        labels: ground truth (0 = normal, 1 = anomaly)
        scores: anomaly scores
        threshold: ngưỡng (None = tự tìm tối ưu)

    Returns:
        dict: {"tn": int, "fp": int, "fn": int, "tp": int}
    """
    labels, scores = _to_numpy(labels, scores)

    if threshold is None:
        threshold = find_optimal_threshold(labels, scores)

    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    return {
        "tn": int(tn),  # True Negative — normal, đoán đúng normal
        "fp": int(fp),  # False Positive — normal, đoán sai anomaly (false alarm)
        "fn": int(fn),  # False Negative — anomaly, đoán sai normal (BỎ SÓT!)
        "tp": int(tp),  # True Positive — anomaly, đoán đúng anomaly
    }


# ============================================================
# 8. PIXEL-LEVEL AUROC — Đánh giá anomaly map vs ground truth mask
# ============================================================


def compute_pixel_auroc(masks, anomaly_maps):
    """
    Tính Pixel-level AUROC — so sánh anomaly map với ground truth mask.

    Ý tưởng:
        - Flatten tất cả pixels từ tất cả ảnh thành 1 list
        - Ground truth: mask pixel = 1 → anomaly pixel, 0 → normal pixel
        - Prediction: anomaly map score cho mỗi pixel
        - Tính AUROC giống image-level nhưng trên toàn bộ pixels

    Ý nghĩa:
        - 1.0 = model định vị hoàn hảo vùng lỗi ở cấp pixel
        - 0.5 = random (model không biết vùng nào lỗi)
        - Metric này chỉ có ý nghĩa cho models tạo anomaly map (PatchCore)

    Args:
        masks: list of numpy arrays (H, W) — ground truth masks (0/1)
        anomaly_maps: list of numpy arrays (H, W) — predicted anomaly maps

    Returns:
        pixel_auroc: float (0.0 → 1.0)
    """
    all_mask_pixels = []
    all_score_pixels = []

    for mask, amap in zip(masks, anomaly_maps):
        mask = np.asarray(mask, dtype=np.float32)
        amap = np.asarray(amap, dtype=np.float32)

        # Resize anomaly map về cùng kích thước mask nếu khác
        if mask.shape != amap.shape:
            from PIL import Image

            amap_img = Image.fromarray(amap)
            amap_img = amap_img.resize((mask.shape[1], mask.shape[0]), Image.BILINEAR)
            amap = np.array(amap_img, dtype=np.float32)

        all_mask_pixels.append(mask.flatten())
        all_score_pixels.append(amap.flatten())

    all_mask_pixels = np.concatenate(all_mask_pixels)
    all_score_pixels = np.concatenate(all_score_pixels)

    # Kiểm tra có cả 2 class không (cần ít nhất 1 pixel anomaly và 1 pixel normal)
    if len(np.unique(all_mask_pixels)) < 2:
        return 0.0

    return roc_auc_score(all_mask_pixels, all_score_pixels)


# ============================================================
# 9. PRO SCORE — Per-Region Overlap
# ============================================================


def compute_pro_score(masks, anomaly_maps, num_thresholds=200):
    """
    Tính Per-Region Overlap (PRO) Score — metric chuẩn MVTec paper.

    PRO đo chất lượng localization bằng cách:
        1. Với mỗi ngưỡng, tính overlap cho TỪNG connected component (vùng lỗi riêng biệt)
        2. Trung bình overlap qua tất cả regions
        3. Vẽ PRO curve vs FPR → tính diện tích dưới đường cong

    Tại sao PRO tốt hơn Pixel-AUROC:
        - Pixel-AUROC bị bias bởi vùng lỗi lớn (nhiều pixels → weight lớn)
        - PRO trung bình qua từng REGION → công bằng cho lỗi nhỏ và lớn

    Args:
        masks: list of numpy arrays (H, W) — ground truth masks (0/1)
        anomaly_maps: list of numpy arrays (H, W) — predicted anomaly maps
        num_thresholds: số ngưỡng để tính PRO curve

    Returns:
        pro_score: float (0.0 → 1.0)
    """
    from scipy import ndimage

    all_pro_values = []
    all_fpr_values = []

    # Chuẩn bị data
    processed_masks = []
    processed_amaps = []

    for mask, amap in zip(masks, anomaly_maps):
        mask = np.asarray(mask, dtype=np.float32)
        amap = np.asarray(amap, dtype=np.float32)

        if mask.shape != amap.shape:
            from PIL import Image

            amap_img = Image.fromarray(amap)
            amap_img = amap_img.resize((mask.shape[1], mask.shape[0]), Image.BILINEAR)
            amap = np.array(amap_img, dtype=np.float32)

        processed_masks.append(mask)
        processed_amaps.append(amap)

    # Tìm min/max score để tạo thresholds
    all_scores = np.concatenate([a.flatten() for a in processed_amaps])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)

    for threshold in thresholds:
        region_overlaps = []
        total_fp = 0
        total_tn = 0

        for mask, amap in zip(processed_masks, processed_amaps):
            # Binary prediction tại threshold
            pred = (amap >= threshold).astype(np.float32)

            # FPR: False Positive Rate trên pixel bình thường
            normal_pixels = mask == 0
            if normal_pixels.sum() > 0:
                total_fp += (pred[normal_pixels] == 1).sum()
                total_tn += (pred[normal_pixels] == 0).sum()

            # Tìm connected components trong ground truth
            labeled_mask, num_regions = ndimage.label(mask)

            for region_id in range(1, num_regions + 1):
                region = labeled_mask == region_id
                region_size = region.sum()

                if region_size == 0:
                    continue

                # Overlap = tỷ lệ pixels trong region được predict đúng
                overlap = (pred[region] == 1).sum() / region_size
                region_overlaps.append(overlap)

        # Trung bình PRO qua tất cả regions
        if region_overlaps:
            mean_pro = np.mean(region_overlaps)
        else:
            mean_pro = 0.0

        # FPR
        fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0

        all_pro_values.append(mean_pro)
        all_fpr_values.append(fpr)

    # Sort by FPR ascending
    sorted_indices = np.argsort(all_fpr_values)
    fpr_sorted = np.array(all_fpr_values)[sorted_indices]
    pro_sorted = np.array(all_pro_values)[sorted_indices]

    # Giới hạn FPR ≤ 0.3 (theo MVTec paper)
    fpr_limit = 0.3
    valid = fpr_sorted <= fpr_limit

    if valid.sum() < 2:
        return 0.0

    fpr_valid = fpr_sorted[valid]
    pro_valid = pro_sorted[valid]

    # Tính diện tích dưới đường cong (AUC) và normalize
    pro_auc = np.trapz(pro_valid, fpr_valid)
    pro_score = pro_auc / fpr_limit  # Normalize về [0, 1]

    return float(np.clip(pro_score, 0.0, 1.0))


# ============================================================
# 10. EVALUATE ALL — Chạy tất cả metrics cùng lúc
# ============================================================


def evaluate_all(
    labels, scores, masks=None, anomaly_maps=None, adaptive_threshold=None
):
    """
    Chạy tất cả metrics và trả về dict tổng hợp.

    Đây là hàm chính nên dùng khi evaluate model.
    Trả về đầy đủ: AUROC, AP, F1, Precision, Recall, Specificity, Confusion Matrix.
    Nếu có masks + anomaly_maps: thêm Pixel-AUROC và PRO Score.

    Args:
        labels: ground truth (0 = normal, 1 = anomaly)
        scores: anomaly scores
        masks: (optional) list of ground truth masks cho pixel-level eval
        anomaly_maps: (optional) list of anomaly maps cho pixel-level eval
        adaptive_threshold: (optional) float — ngưỡng dynamic threshold học từ Inference

    Returns:
        dict chứa tất cả metrics:
        {
            "auroc": float,
            "average_precision": float,
            "f1": float,
            "precision": float,
            "recall": float,
            "specificity": float,
            "threshold": float,
            "adaptive_threshold": float (nếu có),
            "confusion_matrix": {"tn": int, "fp": int, "fn": int, "tp": int},
            "pixel_auroc": float (nếu có masks),
            "pro_score": float (nếu có masks)
        }
    """
    labels, scores = _to_numpy(labels, scores)

    # Tìm ngưỡng tối ưu 1 lần, dùng chung cho tất cả
    threshold = find_optimal_threshold(labels, scores)

    # Tính tất cả metrics
    auroc = compute_auroc(labels, scores)
    ap = compute_average_precision(labels, scores)
    f1_pr = compute_f1_precision_recall(labels, scores, threshold=threshold)
    spec = compute_specificity(labels, scores, threshold=threshold)
    cm = compute_confusion_matrix(labels, scores, threshold=threshold)

    result = {
        "auroc": auroc,
        "average_precision": ap,
        "f1": f1_pr["f1"],
        "precision": f1_pr["precision"],
        "recall": f1_pr["recall"],
        "specificity": spec,
        "threshold": threshold,
        "confusion_matrix": cm,
    }

    if adaptive_threshold is not None:
        result["adaptive_threshold"] = adaptive_threshold

    # Pixel-level metrics (nếu có ground truth masks)
    if masks is not None and anomaly_maps is not None:
        try:
            result["pixel_auroc"] = compute_pixel_auroc(masks, anomaly_maps)
        except Exception as e:
            print(f"  [WARNING] Pixel AUROC failed: {e}")
            result["pixel_auroc"] = 0.0

        try:
            result["pro_score"] = compute_pro_score(masks, anomaly_maps)
        except Exception as e:
            print(f"  [WARNING] PRO Score failed: {e}")
            result["pro_score"] = 0.0

    return result


def print_metrics(metrics, model_name="Model"):
    """
    In kết quả metrics dạng đẹp, kèm chú thích tiếng Việt.

    Args:
        metrics: dict từ evaluate_all()
        model_name: tên model để hiển thị
    """
    print(f"\n{'='*70}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*70}")

    # --- Ranking metrics (không phụ thuộc ngưỡng) ---
    print(f"\n   Ranking Metrics (không phụ thuộc ngưỡng):")
    print(f"  {'─'*60}")

    auroc = metrics["auroc"]
    ap = metrics["average_precision"]

    print(f"  AUROC:              {auroc:.4f}    ", end="")
    if auroc >= 0.9:
        print("← Rất tốt ")
    elif auroc >= 0.8:
        print("← Khá tốt")
    elif auroc >= 0.7:
        print("← Trung bình")
    else:
        print("← Cần cải thiện ")
    print(
        f"    ↳ Khả năng xếp hạng anomaly cao hơn normal (1.0 = hoàn hảo, 0.5 = random)"
    )

    print(f"  Average Precision:  {ap:.4f}    ", end="")
    if ap >= 0.9:
        print("← Rất tốt ")
    elif ap >= 0.8:
        print("← Khá tốt")
    elif ap >= 0.7:
        print("← Trung bình")
    else:
        print("← Cần cải thiện ")
    print(
        f"    ↳ Độ chính xác trung bình khi detect anomaly (tốt hơn AUROC cho data mất cân bằng)"
    )

    # --- Pixel-level metrics (nếu có) ---
    if "pixel_auroc" in metrics:
        print(f"\n   Pixel-Level Metrics (anomaly localization):")
        print(f"  {'─'*60}")

        px_auroc = metrics["pixel_auroc"]
        print(f"  Pixel AUROC:        {px_auroc:.4f}    ", end="")
        if px_auroc >= 0.9:
            print("← Rất tốt ")
        elif px_auroc >= 0.8:
            print("← Khá tốt")
        elif px_auroc >= 0.7:
            print("← Trung bình")
        else:
            print("← Cần cải thiện ")
        print(f"    ↳ Khả năng định vị chính xác vùng lỗi ở cấp pixel")

        if "pro_score" in metrics:
            pro = metrics["pro_score"]
            print(f"  PRO Score:          {pro:.4f}    ", end="")
            if pro >= 0.8:
                print("← Rất tốt ")
            elif pro >= 0.6:
                print("← Khá tốt")
            elif pro >= 0.4:
                print("← Trung bình")
            else:
                print("← Cần cải thiện ")
            print(f"    ↳ Per-Region Overlap — công bằng cho cả lỗi nhỏ và lớn")

    # --- Threshold-based metrics ---
    if "adaptive_threshold" in metrics and metrics["adaptive_threshold"] is not None:
        print(f"\n   Threshold-based Metrics (tại ngưỡng linh hoạt/tối ưu):")
        print(f"  {'─'*60}")
        print(f"  Youden's J Threshold: {metrics['threshold']:.6f} (Optimal Lab)")
        print(
            f"  Adaptive Threshold:   {metrics['adaptive_threshold']:.6f} (Dynamic Inference)"
        )
        print(
            f"  Difference (Δ):        {abs(metrics['threshold'] - metrics['adaptive_threshold']):.6f}"
        )
        print(f"    ↳ Score ≥ ngưỡng → Anomaly | Score < ngưỡng → Normal")
    else:
        print(f"\n   Threshold-based Metrics (tại ngưỡng tối ưu):")
        print(f"  {'─'*60}")
        print(f"  Optimal Threshold:  {metrics['threshold']:.6f}")
        print(f"    ↳ Score ≥ ngưỡng → Anomaly | Score < ngưỡng → Normal")

    f1 = metrics["f1"]
    prec = metrics["precision"]
    rec = metrics["recall"]
    spec = metrics["specificity"]

    print(f"\n  F1 Score:           {f1:.4f}")
    print(f"    ↳ Trung bình hài hòa Precision & Recall (1.0 = hoàn hảo)")

    print(f"  Precision:          {prec:.4f}")
    print(f"    ↳ Trong số ảnh model nói anomaly, {prec*100:.1f}% thực sự là anomaly")

    print(f"  Recall:             {rec:.4f}")
    print(f"    ↳ Model phát hiện được {rec*100:.1f}% tổng số ảnh anomaly thực tế")

    print(f"  Specificity:        {spec:.4f}")
    print(f"    ↳ Model nhận đúng {spec*100:.1f}% ảnh bình thường (không báo nhầm)")

    # --- Confusion Matrix ---
    cm = metrics["confusion_matrix"]
    total = cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]

    print(f"\n   Confusion Matrix (Ma trận nhầm lẫn):")
    print(f"  {'─'*60}")
    print(f"                      Pred Normal   Pred Anomaly")
    print(f"  Actual Normal       {cm['tn']:>8d}       {cm['fp']:>8d}")
    print(f"  Actual Anomaly      {cm['fn']:>8d}       {cm['tp']:>8d}")
    print(f"  {'─'*60}")
    print(f"  TN={cm['tn']} (normal→đúng)  FP={cm['fp']} (normal→báo nhầm)")
    print(f"  FN={cm['fn']} (anomaly→bỏ sót!)  TP={cm['tp']} (anomaly→phát hiện)")
    print(f"  Tổng: {total} ảnh test")
    print(f"{'='*70}")


# ============================================================
# HELPER — Chuyển đổi input
# ============================================================


def _to_numpy(labels, scores):
    """
    Chuyển labels và scores sang numpy array.
    Chấp nhận: list, torch.Tensor, numpy array.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)

    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    elif isinstance(scores, list):
        scores = np.array(scores)

    return labels, scores
