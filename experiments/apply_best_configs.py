"""
Script: Áp dụng hàng loạt Cấu Hình Tối Ưu (Best Configs).
Chức năng: Quét toàn bộ 15 file `best_agent_config.json` do AI sinh ra,
sau đó tự động gọi hàm `run_patchcore` để rèn lại (Rebuild) 15 file `memory_bank.pth` siêu cấp.
Rất hữu ích khi bạn đổi máy tính hoặc rỗng thư mục Checkpoints.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_patchcore import run as run_pc


def rebuild_all_best_models():
    print("==================================================================")
    print(" [REBUILD] KHOI DONG CHUOI PHUC HOI PATCHCORE TU AI CHECKPOINTS")
    print("==================================================================\n")

    # Tìm tất cả các file best_agent_config.json
    checkpoint_dir = os.path.join("checkpoints", "patchcore")
    json_files = glob.glob(os.path.join(checkpoint_dir, "*", "best_agent_config.json"))

    if not json_files:
        print(" [LỖI] Không tìm thấy file JSON cấu hình nào.")
        print(
            " Hãy đảm bảo bạn đã chạy `auto_optimize.py` hoặc `run_all_auto.py` trước."
        )
        return

    print(f" Tiên lượng: Tìm thấy {len(json_files)} file cấu hình Vàng.\n")

    for i, json_path in enumerate(json_files, 1):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            category = data["category"]
            config = data["optimal_config"]
            score = data.get("multi_objective_score", "N/A")

            print(f"[{i}/{len(json_files)}] ĐANG TÁI TẠO: {category.upper()}")
            print(f"   => Thông số nạp: {config} | Kỷ lục đã xác lập: {score}")

            # Xóa bank cũ để ép tạo mới
            mem_bank_path = os.path.join(checkpoint_dir, category, "memory_bank.pth")
            if os.path.exists(mem_bank_path):
                os.remove(mem_bank_path)

            # Đốt lò (Rebuild)
            run_pc(category, **config)
            print(
                f"   [THÀNH CÔNG] Đã in xong memory_bank.pth cho {category.upper()}!\n"
            )

        except Exception as e:
            print(f"   [THẤT BẠI] Lỗi khi xử lý {json_path}: {e}\n")

    print("==================================================================")
    print(" TOÀN BỘ MÔ HÌNH ĐÃ ĐƯỢC PHỤC HỒI VÀ SẴN SÀNG CHO APP.PY!")
    print("==================================================================")


if __name__ == "__main__":
    rebuild_all_best_models()
