import json
import os
import subprocess

# Ngưỡng AUROC để kích hoạt tối ưu hóa (Dưới 0.80 là cần tinh chỉnh)
THRESHOLD = 0.80


def get_low_performers(results_dir="results"):
    low_performers = []
    if not os.path.exists(results_dir):
        return []

    for f in os.listdir(results_dir):
        if f.endswith(".json"):
            cat = f.replace(".json", "")
            with open(os.path.join(results_dir, f)) as j:
                data = json.load(j)
                auroc = data["patchcore"]["auroc"]
                if auroc < THRESHOLD:
                    low_performers.append((cat, auroc))

    return sorted(low_performers, key=lambda x: x[1])


def run_optimization():
    targets = get_low_performers()

    if not targets:
        print("[OK] Khong co category nao duoi nguong 0.80 AUROC.")
        return

    print(f"[OPTIMIZE] Tim thay {len(targets)} categories can toi uu hoa:")
    for cat, score in targets:
        print(f"   - {cat:15s} | Hiện tại: {score:.4f}")

    print("\n" + "=" * 50)
    for cat, score in targets:
        # "Thả ga" -> Để số iterations cực lớn (100) để kích hoạt Early Stopping nội tại của file optimize
        iters = 100

        print(
            f"\n[RUNNING] Dang toi uu hoa (Early Stopping) [{cat.upper()}] (Score: {score:.4f})..."
        )
        cmd = [
            "python",
            "experiments/auto_optimize.py",
            "--category",
            cat,
            "--iterations",
            str(iters),
        ]

        try:
            subprocess.run(cmd, check=True)
            # Sau khi xong mỗi category, cập nhật lại kết quả vào results.json
            update_results = ["python", "-c", f"import json, os; \
                with open('results/{cat}.json') as r: d = json.load(r); \
                from models.patchcore import PatchCore; \
                # Do result dẽ được tự cập nhật qua ExperimentTracker sau này, \
                # ở đây chỉ in ra để user biết đã hoàn thành tối ưu."]
            print(f"[DONE] Hoan thanh toi uu hoa thuc nghiem cho {cat}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Loi khi toi uu hoa {cat}: {e}")


if __name__ == "__main__":
    run_optimization()
