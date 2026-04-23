import json
import os
import subprocess


def harvest_and_sync():
    print("[SYNC] Dang thu hoach ket qua toi uu tu cac Agent...")
    results_path = "results"
    checkpoints_path = "checkpoints/patchcore"

    # Duyệt qua các folder checkpoints để lấy config tốt nhất
    for cat in os.listdir(checkpoints_path):
        config_file = os.path.join(checkpoints_path, cat, "best_agent_config.json")
        if os.path.exists(config_file):
            with open(config_file) as f:
                best = json.load(f)

            # Cập nhật kết quả vào results.json tương ứng
            res_json = os.path.join(results_path, f"{cat}.json")
            if os.path.exists(res_json):
                with open(res_json, "r") as f:
                    data = json.load(f)

                # Cập nhật AUROC và Config từ kết quả Agent
                # Lưu ý: Agent lưu multi_objective_score, chúng ta cần raw_auroc từ history nếu cần
                # Nhưng đơn giản nhất là ghi nhận config để tái hiện.
                data["patchcore"]["config"] = best["optimal_config"]
                # data["patchcore"]["auroc"] = best.get("raw_auroc", data["patchcore"]["auroc"])

                with open(res_json, "w") as f:
                    json.dump(data, f, indent=4)
                print(f"   - Đã cập nhật kết quả tối ưu cho {cat.upper()}")

    # Chạy lại sync để kết xuất ra kết quả CSV cuối cùng
    print("\n[EXPORT] Dang ket xuat Bang so lieu Luan van (results.csv)...")
    try:
        # Gọi script sync có sẵn
        import subprocess

        subprocess.run(["python", "tools/sync_results.py"], check=True)
        print("[DONE] results.csv da diem danh day du so lieu SOTA!")
    except Exception as e:
        print(f"[ERROR] Loi khi dong bo CSV: {e}")


if __name__ == "__main__":
    harvest_and_sync()
