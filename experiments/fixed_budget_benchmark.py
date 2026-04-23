"""
Fixed Budget Benchmark — So sánh models với thời gian training cố định.

Lấy cảm hứng từ karpathy/autoresearch: mỗi experiment chạy trong
đúng N phút, bất kể model architecture hay hyperparameters.
→ So sánh công bằng: cùng budget, model nào tốt nhất?

Usage:
    python experiments/fixed_budget_benchmark.py --category bottle --budget-minutes 5
    python experiments/fixed_budget_benchmark.py --category bottle --budget-minutes 10 --skip-training
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data_processing.mvtec import MVTecDataset
from evaluation.metrics import evaluate_all, print_metrics
from experiments.experiment_tracker import ExperimentTracker
from models.autoencoder import Autoencoder
from training.trainer import Trainer

DATA_ROOT = "datasets/mvtec"


class FixedBudgetBenchmark:
    """
    Benchmark models với thời gian training cố định.

    Mỗi model train trong đúng budget_minutes phút → evaluate → log.
    Autoencoder và backbone training sẽ bị giới hạn bởi time budget.
    PatchCore và CNN+OC-SVM không cần training (chỉ fit/inference).

    Args:
        category: MVTec category name
        budget_minutes: thời gian training tối đa cho mỗi model (phút)
    """

    def __init__(self, category, budget_minutes=5):
        self.category = category
        self.budget_minutes = budget_minutes
        self.budget_seconds = budget_minutes * 60
        self.tracker = ExperimentTracker()

    def run_all(self, skip_training=False):
        """
        Chạy benchmark với fixed budget cho tất cả models.

        Args:
            skip_training: True = chỉ evaluate (đã train trước)
        """
        print(f"\n{'='*70}")
        print(f"    FIXED BUDGET BENCHMARK")
        print(f"  Category: {self.category}")
        print(f"  Budget: {self.budget_minutes} minutes per model")
        print(f"{'='*70}")

        results = {}

        # 1. Autoencoder (có training với budget)
        print(f"\n{'─'*40}")
        print(f"  [1/3] Autoencoder (budget: {self.budget_minutes} min)")
        print(f"{'─'*40}")
        ae_result = self._run_autoencoder(skip_training)
        results["autoencoder"] = ae_result

        # 2. CNN+OC-SVM (fit thôi, không training)
        print(f"\n{'─'*40}")
        print(f"  [2/3] CNN+OC-SVM")
        print(f"{'─'*40}")
        oc_result = self._run_ocsvm()
        results["cnn_ocsvm"] = oc_result

        # 3. PatchCore (fit memory bank, không training)
        print(f"\n{'─'*40}")
        print(f"  [3/3] PatchCore")
        print(f"{'─'*40}")
        pc_result = self._run_patchcore()
        results["patchcore"] = pc_result

        # Summary
        self._print_comparison(results)

        return results

    def _run_autoencoder(self, skip_training=False):
        """Autoencoder với fixed time budget."""
        start_time = time.time()

        if not skip_training:
            # Train với time limit
            train_dataset = MVTecDataset(DATA_ROOT, self.category, split="train")
            valid_dataset = MVTecDataset(DATA_ROOT, self.category, split="valid")

            model = Autoencoder()
            ckpt_dir = f"checkpoints/autoencoder/{self.category}"

            trainer = Trainer(
                model=model,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                batch_size=32,
                epochs=9999,  # sẽ bị giới hạn bởi time
                lr=1e-3,
                checkpoint_dir=ckpt_dir,
            )

            # Custom training loop với time limit
            completed_epochs = 0
            deadline = start_time + self.budget_seconds

            print(f"  Training with {self.budget_minutes}-minute budget...")
            for epoch in range(1, 9999):
                if time.time() >= deadline:
                    print(f"   Budget reached after {completed_epochs} epochs")
                    break

                try:
                    trainer._train_epoch(epoch)
                    completed_epochs = epoch
                except Exception:
                    # Fallback: nếu _train_epoch không khả dụng
                    break

            if completed_epochs == 0:
                # Fallback: train bình thường nếu custom loop không hoạt động
                print(f"  Fallback: standard training...")
                max_epochs = max(1, int(self.budget_minutes))
                trainer.epochs = max_epochs
                trainer.train()

        # Evaluate
        from experiments.run_autoencoder import run as run_ae

        metrics = run_ae(self.category)

        duration = time.time() - start_time

        if metrics:
            self.tracker.log_experiment(
                experiment_name=f"budget_{self.category}_autoencoder",
                model="autoencoder",
                category=self.category,
                config={"budget_minutes": self.budget_minutes},
                metrics=metrics,
                duration_seconds=duration,
                notes=f"Fixed budget: {self.budget_minutes} min",
            )

        return {"metrics": metrics, "duration": duration}

    def _run_patchcore(self):
        """PatchCore evaluation."""
        start_time = time.time()

        from experiments.run_patchcore import run as run_pc

        metrics = run_pc(self.category)

        duration = time.time() - start_time

        self.tracker.log_experiment(
            experiment_name=f"budget_{self.category}_patchcore",
            model="patchcore",
            category=self.category,
            config={"coreset_ratio": 0.1, "k_neighbors": 1},
            metrics=metrics,
            duration_seconds=duration,
            notes=f"Fixed budget benchmark",
        )

        return {"metrics": metrics, "duration": duration}

    def _run_ocsvm(self):
        """CNN+OC-SVM evaluation."""
        start_time = time.time()

        from experiments.run_ocsvm import run as run_oc

        metrics = run_oc(self.category)

        duration = time.time() - start_time

        self.tracker.log_experiment(
            experiment_name=f"budget_{self.category}_cnn_ocsvm",
            model="cnn_ocsvm",
            category=self.category,
            config={"gamma": "auto"},
            metrics=metrics,
            duration_seconds=duration,
            notes=f"Fixed budget benchmark",
        )

        return {"metrics": metrics, "duration": duration}

    def _print_comparison(self, results):
        """In bảng so sánh."""
        print(f"\n{'='*70}")
        print(f"   FIXED BUDGET RESULTS — {self.category}")
        print(f"  Budget: {self.budget_minutes} minutes")
        print(f"{'='*70}")

        print(f"\n  {'Model':<15} {'AUROC':<8} {'F1':<8} {'AP':<8} {'Duration':<12}")
        print(f"  {'-'*55}")

        for model_name, result in results.items():
            m = result.get("metrics")
            dur = result.get("duration", 0)

            if m:
                print(
                    f"  {model_name:<15} {m['auroc']:.4f}   "
                    f"{m['f1']:.4f}   {m['average_precision']:.4f}   "
                    f"{dur:.1f}s"
                )
            else:
                print(f"  {model_name:<15} N/A      N/A      N/A      " f"{dur:.1f}s")

        print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed Budget Benchmark")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument(
        "--budget-minutes",
        type=float,
        default=5,
        help="Training budget in minutes per model",
    )
    parser.add_argument(
        "--skip-training", action="store_true", help="Skip training, only evaluate"
    )
    args = parser.parse_args()

    benchmark = FixedBudgetBenchmark(args.category, args.budget_minutes)
    benchmark.run_all(skip_training=args.skip_training)
