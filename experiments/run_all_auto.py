"""
Chạy Agent Auto-Optimizer trên toàn bộ 15 danh mục của MVTec AD.
Mục tiêu là gom toàn bộ công đoạn tối ưu (research) vào một cú click duy nhất.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Khai báo đường dẫn root để Python nhận diện các module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.auto_optimize import AnonymousResearchAgent

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def run_all(iterations_per_category=15, categories_to_run=MVTEC_CATEGORIES):
    print("============================================================")
    print(f" KHỞI ĐỘNG HỆ THỐNG AGENT OPTIMIZE CHO {len(categories_to_run)} DANH MỤC")
    print(f" Số vòng lặp tự tối ưu cho mỗi danh mục: {iterations_per_category}")
    print(" Cảnh báo: Việc này có thể mất nhiều giờ, tuỳ sức mạnh GPU.")
    print("============================================================\n")

    for i, category in enumerate(categories_to_run, 1):
        print(f"\n============================================================")
        print(f"[{i}/{len(categories_to_run)}] BẮT ĐẦU NGHIÊN CỨU: {category.upper()}")
        print(f"============================================================")

        # Khởi tạo Autonomous Agent
        agent = AnonymousResearchAgent(category)

        # Bắt đầu vòng lặp suy luận tối ưu
        try:
            agent.run(iterations=iterations_per_category)
        except Exception as e:
            print(f"[LỖI NGHIÊM TRỌNG Ở DANH MỤC {category.upper()}]: {e}")
            print(f"Bỏ qua danh mục {category} để tiếp tục các danh mục khác...")

        print(f"[{i}/{len(categories_to_run)}] HOÀN TẤT: {category.upper()}\n")

    print("\n============================================================")
    print(
        f" TOÀN BỘ {len(categories_to_run)} DANH MỤC ĐÃ ĐƯỢC LLM AGENT TỐI ƯU HOÀN TẤT."
    )
    print(" Hãy kiểm tra kết quả tại 'experiments/research_log.txt'")
    print("============================================================")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Chạy Auto LLM Optimize cho toàn bộ MVTec AD"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=15,
        help="Số vòng lặp suy luận cho mỗi category",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=MVTEC_CATEGORIES,
        help="Danh sách danh mục cần chạy (Cách nhau bởi dấu cách)",
    )
    args = parser.parse_args()

    # Nạp API Key nếu có
    if not os.environ.get("GEMINI_API_KEY"):
        print(
            "[THÔNG BÁO] Không tìm thấy GEMINI_API_KEY. Hệ thống sẽ thay thế bằng chế độ Mock LLM Random Configuration."
        )

    run_all(args.iterations, args.categories)
