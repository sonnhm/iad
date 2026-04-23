"""
Test suite — kiểm tra model shapes, PatchCore from scratch, backbone KD, dataset, metrics.

Usage:
    python tests/test_all.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def test_autoencoder():
    """Test Autoencoder input/output shape."""
    from models.autoencoder import Autoencoder

    model = Autoencoder()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)

    assert y.shape == x.shape, f"Shape mismatch: {x.shape} != {y.shape}"
    print("[PASS] Autoencoder: input (2,3,256,256) → output", y.shape)


def test_custom_resnet18():
    """Test Custom ResNet18 output shape."""
    from models.custom_resnet import custom_resnet18

    model = custom_resnet18(num_classes=1000)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)

    assert y.shape == (2, 1000), f"Expected (2,1000), got {y.shape}"
    print("[PASS] Custom ResNet18: output", y.shape)

    # Feature mode
    model_feat = custom_resnet18(num_classes=None)
    y_feat = model_feat(x)
    assert y_feat.shape == (2, 512), f"Expected (2,512), got {y_feat.shape}"
    print("[PASS] Custom ResNet18 (feature mode): output", y_feat.shape)


def test_patchcore_feature_extractor():
    """Test PatchCoreFeatureExtractor output shape."""
    from models.patchcore import PatchCoreFeatureExtractor

    model = PatchCoreFeatureExtractor()
    model.eval()
    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        features = model(x)

    # features: (1, 384, H', W')
    assert features.dim() == 4, f"Expected 4D, got {features.dim()}D"
    assert features.shape[1] == 384, f"Expected 384 channels, got {features.shape[1]}"
    print(
        f"[PASS] PatchCoreFeatureExtractor: input (1,3,256,256) → features {features.shape}"
    )


def test_patchcore_full():
    """Test PatchCore full pipeline: fit → predict."""
    from models.patchcore import PatchCore, PatchCoreFeatureExtractor

    backbone = PatchCoreFeatureExtractor()

    # Tạo fake dataset
    class FakeDataset:
        def __init__(self, n=5):
            self.n = n

        def __len__(self):
            return self.n

        def __iter__(self):
            for _ in range(self.n):
                yield torch.randn(3, 256, 256), 0

        def __getitem__(self, idx):
            return torch.randn(3, 256, 256), 0

    patchcore = PatchCore(
        backbone=backbone,
        device="cpu",
        coreset_ratio=0.5,  # ratio cao để test nhanh
        k_neighbors=1,
    )

    # Fit
    fake_train = FakeDataset(n=3)
    patchcore.fit(fake_train)
    assert patchcore.memory_bank is not None, "Memory bank should not be None after fit"
    assert patchcore.memory_bank.dim() == 2, "Memory bank should be 2D"
    assert (
        patchcore.memory_bank.shape[1] == 384
    ), f"Expected D=384, got {patchcore.memory_bank.shape[1]}"
    print(f"[PASS] PatchCore fit: memory bank {patchcore.memory_bank.shape}")

    # Predict
    test_img = torch.randn(3, 256, 256)
    scores, anomaly_maps = patchcore.predict(test_img)
    assert scores.shape == (1,), f"Expected (1,), got {scores.shape}"
    assert anomaly_maps.ndim == 3, f"Expected 3D anomaly map, got {anomaly_maps.ndim}D"
    print(f"[PASS] PatchCore predict: score={scores[0]:.4f}, map={anomaly_maps.shape}")


def test_euclidean_distance():
    """Test self-implemented Euclidean distance."""
    from models.patchcore import PatchCore

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    dist = PatchCore._euclidean_distance_batch(x, y)

    # dist[0,0] = ||[1,0] - [0,0]|| = 1.0
    # dist[0,1] = ||[1,0] - [1,1]|| = 1.0
    # dist[1,0] = ||[0,1] - [0,0]|| = 1.0
    # dist[1,1] = ||[0,1] - [1,1]|| = 1.0
    assert dist.shape == (2, 2), f"Expected (2,2), got {dist.shape}"
    assert torch.allclose(
        dist, torch.ones(2, 2), atol=1e-5
    ), f"Distance incorrect: {dist}"
    print(f"[PASS] Euclidean distance: {dist.tolist()}")


def test_coreset():
    """Test self-implemented coreset subsampling."""
    from models.patchcore import PatchCore, PatchCoreFeatureExtractor

    backbone = PatchCoreFeatureExtractor()
    patchcore = PatchCore(backbone=backbone, coreset_ratio=0.5)

    # 10 random features
    features = torch.randn(10, 384)
    coreset = patchcore._coreset_subsampling(features)

    expected_size = max(1, int(10 * 0.5))
    assert coreset.shape == (
        expected_size,
        384,
    ), f"Expected ({expected_size},384), got {coreset.shape}"
    print(f"[PASS] Coreset subsampling: {features.shape} → {coreset.shape}")


def test_backbone_kd_shapes():
    """Test Knowledge Distillation: teacher/student output shapes match."""
    from torchvision.models import ResNet18_Weights, resnet18

    from models.custom_resnet import CustomResNet18

    teacher = resnet18(weights=ResNet18_Weights.DEFAULT)
    student = CustomResNet18(num_classes=None)

    teacher.eval()
    student.eval()

    x = torch.randn(1, 3, 256, 256)

    # Teacher features
    with torch.no_grad():
        t_x = teacher.relu(teacher.bn1(teacher.conv1(x)))
        t_x = teacher.maxpool(t_x)
        t1 = teacher.layer1(t_x)
        t2 = teacher.layer2(t1)
        t3 = teacher.layer3(t2)
        t4 = teacher.layer4(t3)

    # Student features
    with torch.no_grad():
        s_x = student.relu(student.bn1(student.conv1(x)))
        s_x = student.maxpool(s_x)
        s1 = student.layer1(s_x)
        s2 = student.layer2(s1)
        s3 = student.layer3(s2)
        s4 = student.layer4(s3)

    # Shapes must match for KD loss
    assert t1.shape == s1.shape, f"Layer1 mismatch: {t1.shape} vs {s1.shape}"
    assert t2.shape == s2.shape, f"Layer2 mismatch: {t2.shape} vs {s2.shape}"
    assert t3.shape == s3.shape, f"Layer3 mismatch: {t3.shape} vs {s3.shape}"
    assert t4.shape == s4.shape, f"Layer4 mismatch: {t4.shape} vs {s4.shape}"

    print(
        f"[PASS] KD shapes match: L1={t1.shape}, L2={t2.shape}, L3={t3.shape}, L4={t4.shape}"
    )


def test_cnn_feature():
    """Test CNN Feature Extractor output shape."""
    from models.cnn_feature import CNNFeatureExtractor

    model = CNNFeatureExtractor()
    model.eval()
    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        feat = model(x)

    assert feat.shape == (2, 512), f"Expected (2,512), got {feat.shape}"
    print("[PASS] CNNFeatureExtractor: output", feat.shape)


def test_dataset_split():
    """Test MVTec dataset train/valid/test split."""
    from data_processing.mvtec import MVTecDataset

    data_root = "datasets/mvtec"
    category = "bottle"

    if not os.path.exists(os.path.join(data_root, category)):
        print("[SKIP] Dataset not found, skipping dataset test")
        return

    train = MVTecDataset(data_root, category, split="train")
    valid = MVTecDataset(data_root, category, split="valid")
    test = MVTecDataset(data_root, category, split="test")

    assert len(train) + len(valid) > 0, "Dataset rỗng!"
    assert len(train) > len(valid), "Train nên lớn hơn valid (80/20)"

    img, label = test[0]
    assert img.shape[0] == 3
    assert label in (0, 1)

    print(f"[PASS] Dataset split: {len(train)}/{len(valid)}/{len(test)}")


def test_config():
    """Test YAML config loading."""
    from app_utils.config import load_config

    config = load_config("configs/autoencoder.yaml")
    assert "data" in config
    assert "training" in config
    print("[PASS] Config loaded OK")

    # Test patchcore config
    pc_config = load_config("configs/patchcore.yaml")
    assert "patchcore" in pc_config
    assert "coreset_ratio" in pc_config["patchcore"]
    print("[PASS] PatchCore config loaded OK")


def test_metrics():
    """Test evaluation metrics."""
    from evaluation.metrics import (
        anomaly_score,
        compute_auroc,
        compute_average_precision,
        compute_confusion_matrix,
        compute_f1_precision_recall,
        compute_specificity,
        evaluate_all,
    )

    # Test anomaly score
    img = torch.randn(2, 3, 64, 64)
    recon = img + torch.randn_like(img) * 0.1
    scores = anomaly_score(img, recon)
    assert scores.shape == (2,)

    # Classification metrics
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    scores_list = [0.1, 0.2, 0.3, 0.15, 0.7, 0.8, 0.9, 0.85]

    auroc = compute_auroc(labels, scores_list)
    assert 0.0 <= auroc <= 1.0

    ap = compute_average_precision(labels, scores_list)
    assert 0.0 <= ap <= 1.0

    f1_pr = compute_f1_precision_recall(labels, scores_list)
    assert 0.0 <= f1_pr["f1"] <= 1.0

    spec = compute_specificity(labels, scores_list)
    assert 0.0 <= spec <= 1.0

    cm = compute_confusion_matrix(labels, scores_list)
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(labels)

    all_metrics = evaluate_all(labels, scores_list)
    assert len(all_metrics) >= 7

    print(f"[PASS] Metrics: AUROC={auroc:.4f}, AP={ap:.4f}, F1={f1_pr['f1']:.4f}")


def test_pixel_metrics():
    """Test pixel-level evaluation metrics (Pixel AUROC, PRO Score)."""
    import numpy as np

    from evaluation.metrics import compute_pixel_auroc, compute_pro_score

    # Tạo fake ground truth masks và anomaly maps
    # Mask 1: có vùng lỗi ở góc trên trái
    mask1 = np.zeros((32, 32), dtype=np.float32)
    mask1[0:8, 0:8] = 1.0  # vùng lỗi 8x8

    # Mask 2: có vùng lỗi ở giữa
    mask2 = np.zeros((32, 32), dtype=np.float32)
    mask2[12:20, 12:20] = 1.0

    # Mask 3: ảnh good (all zeros)
    mask3 = np.zeros((32, 32), dtype=np.float32)

    # Anomaly maps tương ứng (model tốt → anomaly map cao ở vùng lỗi)
    amap1 = np.random.rand(32, 32).astype(np.float32) * 0.3
    amap1[0:8, 0:8] += 0.5  # score cao ở vùng lỗi

    amap2 = np.random.rand(32, 32).astype(np.float32) * 0.3
    amap2[12:20, 12:20] += 0.5

    amap3 = np.random.rand(32, 32).astype(np.float32) * 0.2  # low scores cho good

    masks = [mask1, mask2, mask3]
    amaps = [amap1, amap2, amap3]

    # Test Pixel AUROC
    pixel_auroc = compute_pixel_auroc(masks, amaps)
    assert 0.0 <= pixel_auroc <= 1.0, f"Pixel AUROC out of range: {pixel_auroc}"
    print(f"[PASS] Pixel AUROC: {pixel_auroc:.4f}")

    # Test PRO Score
    pro_score = compute_pro_score(masks, amaps)
    assert 0.0 <= pro_score <= 1.0, f"PRO Score out of range: {pro_score}"
    print(f"[PASS] PRO Score: {pro_score:.4f}")

    # Test evaluate_all với pixel-level metrics
    from evaluation.metrics import evaluate_all

    labels = [1, 1, 0]  # mask1, mask2 = anomaly, mask3 = good
    scores = [0.8, 0.7, 0.1]  # anomaly scores

    all_metrics = evaluate_all(labels, scores, masks=masks, anomaly_maps=amaps)
    assert "pixel_auroc" in all_metrics, "pixel_auroc missing from evaluate_all"
    assert "pro_score" in all_metrics, "pro_score missing from evaluate_all"
    print(
        f"[PASS] evaluate_all with pixel-level: pixel_auroc={all_metrics['pixel_auroc']:.4f}, "
        f"pro_score={all_metrics['pro_score']:.4f}"
    )


def test_experiment_tracker():
    """Test ExperimentTracker logging."""
    import json
    import tempfile

    from experiments.experiment_tracker import ExperimentTracker

    # Sử dụng temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(log_dir=tmpdir)

        # Log 1 experiment
        log_path = tracker.log_experiment(
            experiment_name="test_exp",
            model="patchcore",
            category="bottle",
            config={"coreset_ratio": 0.1, "k_neighbors": 1},
            metrics={"auroc": 0.95, "f1": 0.90, "average_precision": 0.92},
            duration_seconds=60.5,
            notes="Test experiment",
        )

        # Verify file created
        assert os.path.exists(log_path), f"Log file not created: {log_path}"

        # Verify JSON content
        with open(log_path, "r") as f:
            data = json.load(f)

        assert data["model"] == "patchcore"
        assert data["category"] == "bottle"
        assert data["config"]["coreset_ratio"] == 0.1
        assert data["status"] == "completed"

        # Verify load_all
        all_exps = tracker.load_all()
        assert len(all_exps) == 1

        # Verify summarize doesn't crash
        tracker.summarize()

        # Verify best_config
        best = tracker.best_config(model="patchcore")
        assert best is not None

    print("[PASS] ExperimentTracker: log, load, summarize, best_config")


if __name__ == "__main__":
    print("=" * 60)
    print("  IAD Project Tests")
    print("=" * 60)

    tests = [
        test_autoencoder,
        test_custom_resnet18,
        test_patchcore_feature_extractor,
        test_patchcore_full,
        test_euclidean_distance,
        test_coreset,
        test_backbone_kd_shapes,
        test_cnn_feature,
        test_dataset_split,
        test_config,
        test_metrics,
        test_pixel_metrics,
        test_experiment_tracker,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            print(f"\n--- {test_fn.__name__} ---")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("   All tests passed!")
    else:
        print(f"   {failed} test(s) failed")
    print(f"{'=' * 60}")
