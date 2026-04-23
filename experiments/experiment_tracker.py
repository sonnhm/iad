"""
Experiment Tracker — Hệ thống log structured cho mọi thí nghiệm.

Lấy cảm hứng từ karpathy/autoresearch: mỗi experiment → 1 JSON log.
Dễ dàng so sánh, tìm config tốt nhất, và review lịch sử.

Usage:
    from experiments.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker()
    tracker.log_experiment(
        experiment_name="patchcore_bottle_v1",
        model="patchcore",
        category="bottle",
        config={"coreset_ratio": 0.1, "k_neighbors": 1},
        metrics={"auroc": 0.95, "f1": 0.90},
        duration_seconds=120.5
    )
    tracker.summarize()
"""

import json
import os
from datetime import datetime


class ExperimentTracker:
    """
    Quản lý experiment logs trong folder experiments/logs/.

    Mỗi experiment lưu 1 file JSON chứa:
        - timestamp: thời điểm chạy
        - experiment_name: tên thí nghiệm
        - model: tên model (patchcore, autoencoder, cnn_ocsvm)
        - category: MVTec category
        - config: dict hyperparameters
        - metrics: dict evaluation metrics
        - duration_seconds: thời gian chạy
        - status: "completed" | "failed"
        - notes: ghi chú thêm
    """

    def __init__(self, log_dir="experiments/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log_experiment(
        self,
        experiment_name,
        model,
        category,
        config,
        metrics,
        duration_seconds=0,
        status="completed",
        notes="",
    ):
        """
        Ghi log 1 thí nghiệm.

        Args:
            experiment_name: tên thí nghiệm (unique identifier)
            model: "patchcore" | "autoencoder" | "cnn_ocsvm"
            category: MVTec category name
            config: dict hyperparameters (coreset_ratio, lr, epochs, ...)
            metrics: dict evaluation metrics (auroc, f1, ...)
            duration_seconds: thời gian chạy (giây)
            status: "completed" | "failed"
            notes: ghi chú thêm

        Returns:
            log_path: đường dẫn file JSON đã lưu
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        entry = {
            "timestamp": timestamp,
            "experiment_name": experiment_name,
            "model": model,
            "category": category,
            "config": config,
            "metrics": self._serialize_metrics(metrics),
            "duration_seconds": round(duration_seconds, 2),
            "status": status,
            "notes": notes,
        }

        # Tên file: {timestamp}_{model}_{category}.json
        filename = f"{timestamp}_{model}_{category}.json"
        log_path = os.path.join(self.log_dir, filename)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False, default=str)

        print(f"   Experiment logged: {log_path}")
        return log_path

    def load_all(self):
        """Load tất cả experiment logs."""
        experiments = []

        for filename in sorted(os.listdir(self.log_dir)):
            if filename.endswith(".json"):
                filepath = os.path.join(self.log_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    experiments.append(data)
                except Exception as e:
                    print(f"  [WARNING] Cannot load {filepath}: {e}")

        return experiments

    def summarize(self, model=None, category=None):
        """
        In bảng tổng hợp tất cả experiments.

        Args:
            model: lọc theo model (None = tất cả)
            category: lọc theo category (None = tất cả)
        """
        experiments = self.load_all()

        if model:
            experiments = [e for e in experiments if e["model"] == model]
        if category:
            experiments = [e for e in experiments if e["category"] == category]

        if not experiments:
            print("  No experiments found.")
            return

        print(f"\n{'='*100}")
        print(f"   EXPERIMENT SUMMARY ({len(experiments)} experiments)")
        print(f"{'='*100}")

        print(
            f"\n  {'Timestamp':<20} {'Model':<12} {'Category':<12} "
            f"{'AUROC':<8} {'F1':<8} {'Duration':<10} {'Status':<10}"
        )
        print(f"  {'-'*90}")

        for exp in experiments:
            m = exp.get("metrics", {})
            auroc = m.get("auroc", "N/A")
            f1 = m.get("f1", "N/A")
            auroc_str = f"{auroc:.4f}" if isinstance(auroc, (int, float)) else auroc
            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else f1
            dur = f"{exp.get('duration_seconds', 0):.1f}s"

            print(
                f"  {exp['timestamp']:<20} {exp['model']:<12} "
                f"{exp['category']:<12} {auroc_str:<8} {f1_str:<8} "
                f"{dur:<10} {exp['status']:<10}"
            )

        print(f"{'='*100}")

    def best_config(self, model=None, category=None, metric="auroc"):
        """
        Tìm config tốt nhất theo metric chọn.

        Args:
            model: lọc theo model
            category: lọc theo category
            metric: metric để so sánh (mặc định "auroc")

        Returns:
            best_experiment: dict hoặc None
        """
        experiments = self.load_all()

        if model:
            experiments = [e for e in experiments if e["model"] == model]
        if category:
            experiments = [e for e in experiments if e["category"] == category]

        # Chỉ lấy completed experiments
        experiments = [e for e in experiments if e["status"] == "completed"]

        if not experiments:
            return None

        # Sort by metric (descending)
        def get_metric(exp):
            return exp.get("metrics", {}).get(metric, 0)

        best = max(experiments, key=get_metric)

        print(
            f"\n   Best config for {model or 'all'}/{category or 'all'} "
            f"(by {metric}):"
        )
        print(f"     {metric}: {get_metric(best):.4f}")
        print(f"     Config: {json.dumps(best['config'], indent=2)}")
        print(f"     Timestamp: {best['timestamp']}")

        return best

    @staticmethod
    def _serialize_metrics(metrics):
        """Convert metrics dict to a JSON-serializable format."""
        if metrics is None:
            return {}

        serialized = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                serialized[key] = {
                    k: float(v) if isinstance(v, (int, float)) else str(v)
                    for k, v in value.items()
                }
            elif isinstance(value, (int, float)):
                serialized[key] = float(value)
            else:
                serialized[key] = str(value)
        return serialized

    def get_summary_table(self, metric="auroc"):
        """
        Generate a per-category best-metric summary table across all models.

        Provides an MLflow-style leaderboard view without external dependencies.
        Reads all completed experiment logs and reports the best run per
        (model, category) combination.

        Args:
            metric: metric to rank by, default "auroc"

        Returns:
            summary: dict mapping (model, category) -> best metric value
        """
        experiments = self.load_all()
        experiments = [e for e in experiments if e["status"] == "completed"]

        # Group by (model, category) -> keep best metric
        summary = {}
        for exp in experiments:
            key = (exp["model"], exp["category"])
            val = exp.get("metrics", {}).get(metric, 0)
            if key not in summary or val > summary[key]:
                summary[key] = val

        if not summary:
            print("  [Tracker] No completed experiments found in logs/.")
            return summary

        # Print leaderboard
        models = sorted(set(k[0] for k in summary))
        categories = sorted(set(k[1] for k in summary))

        header = f"  {'Category':<15}" + "".join(f"{m:<14}" for m in models)
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT LEADERBOARD -- Best {metric.upper()} per Category")
        print(f"{'='*70}")
        print(header)
        print(f"  {'-'*65}")

        for cat in categories:
            row = f"  {cat:<15}"
            for mod in models:
                val = summary.get((mod, cat), None)
                row += f"{val:.4f}       " if val is not None else f"{'N/A':<14}"
            print(row)

        print(f"{'='*70}")
        return summary


if __name__ == "__main__":
    # Demo: show experiment history and leaderboard
    tracker = ExperimentTracker()
    tracker.summarize()
    tracker.get_summary_table()
