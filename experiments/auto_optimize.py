"""
Advanced Autonomous Research Agent v3.0 (Fixed 5 Critical Bugs).
Triển khai kỹ thuật Agentic ML với:
1. Đồng bộ Dữ liệu: Raw AUROC và Combined Score (để đánh giá tốc độ).
2. Xóa lỗi Greedy Regex (lấy block `{...}` cực chuẩn).
3. Vòng lặp chống lặp `while duplicate` ngăn xoáy lặp.
4. Memory Limit (chỉ nạp 30 bản ghi gần nhất + Top 5 tốt nhất).
5. Global Diversity: Bỏ qua LLM nếu Stale = 2, ép Random sang cực trị hoàn toàn mới.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.experiment_tracker import ExperimentTracker

# =====================================================================
# 1. CORE UTILITIES (LLM API & ROBUST JSON EXTRACTOR)
# =====================================================================


def setup_llm():
    """Khởi tạo Google Gemini API."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "[WARNING] Không tìm thấy GEMINI_API_KEY. Để chạy Agent thực sự, bạn cần cấp API Key."
        )
        print("[WARNING] Tạm thời dùng chế độ MOCK RANDOMS (không dùng LLM).")
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def robust_json_parser(text: str) -> dict:
    """Bắt chính xác block JSON, loại bỏ Markdown và các chuỗi text rác xung quanh."""
    match = text.find("{")
    last_match = text.rfind("}")
    if match != -1 and last_match != -1 and last_match > match:
        json_str = text[match : last_match + 1]
        return json.loads(json_str)
    # Fallback parse toàn bộ
    return json.loads(text)


def ask_llm_api(prompt: str, model=None, max_retries: int = 3) -> dict:
    """Calls the LLM with robust retry logic."""
    if model is None:
        return {
            "config": {
                "coreset_ratio": round(random.uniform(0.01, 0.49), 3),
                "k_neighbors": random.randint(1, 9),
            },
            "reasoning": "Mock Fallback vì không có API Key.",
        }

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt, generation_config=genai.types.GenerationConfig(temperature=0.8)
            )
            result = robust_json_parser(response.text)

            if "config" not in result:
                raise ValueError("JSON parse thành công nhưng thiếu key 'config'.")

            return result

        except Exception as e:
            print(
                f"   [API LỖI/JSON SAI LỆCH] Lần thử {attempt+1}/{max_retries} thất bại: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))  # Exponential backoff
            else:
                print("   [CRITICAL] LLM bế tắc, trả về giá trị Random ngẫu nhiên.")
                return {
                    "config": {
                        "coreset_ratio": round(random.uniform(0.01, 0.49), 3),
                        "k_neighbors": random.randint(1, 10),
                    },
                    "reasoning": "Fallback Error 500 do API chết cứng.",
                }


# =====================================================================
# 2. COGNITIVE UPGRADES (SUMMARIZATION, MEMORY LIMIT & HYPOTHESIS)
# =====================================================================


def summarize_history(history: List[Dict]) -> str:
    """Tổng hợp lịch sử thành text (Top models, Average by K) có giới hạn cửa sổ Token."""
    if not history:
        return "Chưa có dữ liệu lịch sử nào."

    # Top 5 Configs dựa trên Điểm Đa Mục Tiêu (score)
    sorted_hist = sorted(history, key=lambda x: x["score"], reverse=True)
    top_exps = sorted_hist[:5]

    summary = "=== TOP 5 CẤU HÌNH TỐT NHẤT ===\n"
    for i, exp in enumerate(top_exps, 1):
        summary += f"Hạng {i}: {exp['config']} | Độ chính xác (AUROC gốc): {exp['raw_auroc']:.4f} | Điểm đa mục tiêu (Penalty Tốc độ): {exp['score']:.4f}\n"

    # Nhóm trung bình điểm AUROC (Bỏ qua thời gian) để LLM đánh giá chuẩn xác Core Metric
    k_raw_scores = defaultdict(list)
    for exp in history:
        k = exp["config"].get("k_neighbors")
        if k:
            k_raw_scores[k].append(exp["raw_auroc"])

    summary += "\n=== TRUNG BÌNH ĐỘ CHÍNH XÁC (AUROC) THEO K_NEIGHBORS ===\n"
    summary += "(Giúp bạn phân biệt mốc K nào có khả năng Detect Lỗi cao nhất)\n"
    for k, raw_scores in k_raw_scores.items():
        avg = sum(raw_scores) / len(raw_scores)
        summary += f"k_neighbors={k} -> Trung bình AUROC {avg:.4f} (Trải qua {len(raw_scores)} lần test)\n"

    # Lấy Raw History nhưng GIỚI HẠN MEMORY SIZE (Mới nhất 30 lần)
    recent_hist = history[-30:]
    summary += f"\n=== {len(recent_hist)} THỰC NGHIỆM GẦN NHẤT (MEMORY CACHE) ===\n"
    for exp in recent_hist:
        summary += f"- {exp['config']} -> AUROC Tinh khiết: {exp['raw_auroc']:.4f} (Thời gian inference: {exp['duration']:.2f}s) | Score Chốt Hạ: {exp['score']:.4f}\n"

    return summary


def is_duplicate(cfg: dict, history: List[Dict]) -> bool:
    """Kiểm tra có tồn tại trong lịch sử không (Dung sai 1e-3)."""
    for exp in history:
        old_cfg = exp["config"]
        if old_cfg.get("k_neighbors") == cfg["k_neighbors"]:
            if (
                abs(float(old_cfg.get("coreset_ratio", 0)) - cfg["coreset_ratio"])
                < 1e-3
            ):
                return True
    return False


def get_random_divergent_config(best_config: Dict) -> dict:
    """Sinh tạo một điểm ngẫu nhiên hoàn toàn (Global Diversity Outlier)."""
    best_ratio = float(best_config.get("coreset_ratio", 0.1)) if best_config else 0.1
    best_k = int(best_config.get("k_neighbors", 3)) if best_config else 3

    # Bay ngược trục số: Nếu ratio nhỏ thì sinh lớn, và ngược lại.
    if best_ratio < 0.25:
        new_ratio = round(random.uniform(0.25, 0.50), 3)
    else:
        new_ratio = round(random.uniform(0.005, 0.249), 3)

    new_k = random.randint(1, 10)
    # Ép không được giống K cũ để đa dạng hẳn
    while new_k == best_k:
        new_k = random.randint(1, 10)

    return {"coreset_ratio": new_ratio, "k_neighbors": new_k}


def propose_next_config(
    history: List[Dict], best_config: Dict, force_exploration: bool, model
) -> dict:
    """Xử lý đề xuất Config - Đính kèm vòng lặp vĩnh viễn chặn dội trùng lặp."""

    # [GLOBAL DIVERSITY] XỬ LÝ BYPASS LLM
    if force_exploration and history:
        print(
            " [GLOBAL DIVERSITY BYPASS] Agent đang bỏ qua LLM để dời tọa độ ngẫu nhiên mạnh về cực còn lại!"
        )
        div_config = get_random_divergent_config(best_config)
        # Chặn trùng lặp luôn cả bypass
        while is_duplicate(div_config, history):
            div_config = {
                "coreset_ratio": round(random.uniform(0.005, 0.5), 3),
                "k_neighbors": random.randint(1, 10),
            }

        return {
            "config": div_config,
            "reasoning": "SYSTEM FORCED: Bypass LLM Model để kích hoạt Global Diversity Jump. Agent vọt sang một trục hoàn toàn khác.",
        }

    # Nếu không ép Exploration, cho phép LLM suy luận
    summary_text = summarize_history(history)
    best_text = f"Best Config hiện tại: {best_config}" if best_config else "Chưa có."

    prompt = f"""
    Bạn là Staff AI Engineer tối ưu Hypeparameters cho mạng CNN PatchCore (Anomaly Detection).
    - 'coreset_ratio': float ∈ [0.005, 0.50]
    - 'k_neighbors': int ∈ [1, 10]
    
    Mục tiêu Đa dụng (Multi-objective):
    - Đẩy 'AUROC Tinh khiết' lên cao nhất có thể (chạm 1.00 càng tốt).
    - Nhưng đồng thời để ý 'Thời gian inference' (Nếu nó quá 2 giây, thì Cấu hình đó vô dụng). 
    Điểm Score chốt hạ bị Penalty: Score = AUROC_Tinh_Khiết - (0.01 * Thời gian_Giây). Đừng tham lam 0.001 AUROC mà rớt 5s thời gian.
    
    TÓM TẮT BỘ NHỚ:
    {summary_text}
    {best_text}

    LƯU Ý NGHIÊM NGẶT CỦA HỆ THỐNG:
    Không bao giờ đưa ra Config GIỐNG HỆT như các config nào nằm trong "10 THỰC NGHIỆM GẦN NHẤT".
    Trích xuất ra chuỗi JSON thuần khiết duy nhất:
    {{
        "config": {{"coreset_ratio": float, "k_neighbors": int}},
        "reasoning": "Lý do của bạn..."
    }}
    """

    result = ask_llm_api(prompt, model)
    cfg = result.get("config", {"coreset_ratio": 0.1, "k_neighbors": 1})
    reason = result.get("reasoning", "LLM suy luận lỗi.")

    # Kẹp (Clamp) bounds
    try:
        cfg["coreset_ratio"] = max(
            0.005, min(0.5, float(cfg.get("coreset_ratio", 0.1)))
        )
        cfg["k_neighbors"] = max(1, min(10, int(cfg.get("k_neighbors", 1))))
    except:
        cfg = {"coreset_ratio": 0.1, "k_neighbors": 3}

    # [DEDUP] VÒNG LẶP CHỐNG TRÙNG LẶP TUYỆT ĐỐI
    attempts = 0
    while is_duplicate(cfg, history):
        attempts += 1
        # Ép ra số ngẫu nhiên thuần túy nếu liên tục đụng độ (Quá 2 lần)
        cfg["coreset_ratio"] = round(random.uniform(0.005, 0.5), 3)
        cfg["k_neighbors"] = random.randint(1, 10)
        reason += " [MÁY CHỦ BÁO TRÙNG LẶP]: Regex Code đã từ chối quyết định của vòng trước và ép quay xổ số lấy Config mới!"
        if attempts > 10:
            break  # Không cho dội sập máy

    return {"config": cfg, "reasoning": reason}


# =====================================================================
# 3. RESEARCH AGENT (EARLY STOPPING & MULTI-OBJECTIVE LOOP)
# =====================================================================


class AnonymousResearchAgent:
    def __init__(self, category: str):
        self.category = category
        self.model_name = "patchcore"
        self.score_formula = "AUROC - 0.01 * Time(s)"
        self.tracker = ExperimentTracker()
        self.llm_model = setup_llm()
        self.log_file = os.path.join(os.path.dirname(__file__), "research_log.txt")
        self.checkpoint_dir = f"checkpoints/patchcore/{self.category}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.history = []
        self.best_score = -float("inf")
        self.best_config = None

        self.stale_count = 0

    def _calculate_score(self, auroc: float, duration: float) -> float:
        """Thưởng AUROC, Phạt Giây."""
        return auroc - (0.01 * duration)

    def _persist_best_config(self, config: dict, score: float):
        """Khóa cứng JSON xuống Ổ rải C."""
        path = os.path.join(self.checkpoint_dir, "best_agent_config.json")
        data = {
            "category": self.category,
            "multi_objective_score": score,
            "optimal_config": config,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        import shutil

        current_bank = os.path.join(self.checkpoint_dir, "memory_bank.pth")
        best_bank = os.path.join(self.checkpoint_dir, "best_memory_bank.pth")
        if os.path.exists(current_bank):
            shutil.copy2(current_bank, best_bank)

    def run(self, iterations: int = 20):
        print(f"\n{'='*70}\n [AUTONOMOUS RESEARCH AGENT INIT V3.0]\n{'='*70}")
        print(f" Mục tiêu: {self.category.upper()} | Function: {self.score_formula}")

        all_exps = self.tracker.load_all()
        for exp in all_exps:
            if exp["model"] == self.model_name and exp["category"] == self.category:
                raw_auroc = exp.get("metrics", {}).get("auroc", 0)
                dur = exp.get("duration", 0)
                cfg = exp.get("config", {})
                if raw_auroc > 0:
                    combined_score = self._calculate_score(raw_auroc, dur)
                    # Gắn cờ raw_auroc để hàm summarize xử lý
                    self.history.append(
                        {
                            "config": cfg,
                            "score": combined_score,
                            "raw_auroc": raw_auroc,
                            "duration": dur,
                        }
                    )

                    if combined_score > self.best_score:
                        self.best_score = combined_score
                        self.best_config = cfg

        for i in range(1, iterations + 1):
            print(f"\n[{i}/{iterations}] --- Stale Count: {self.stale_count} ---")

            # [EARLY STOP] 1. EARLY STOPPING
            if self.stale_count >= 5:
                print(
                    " [STOP] EARLY STOPPING KICH HOAT: 5 lan lien tiep GPU hao mon ma diem khong tang. Dam kich tran SOTA! Ngung!"
                )
                break

            # [EXPLORE] 2. FORCED EXPLORATION
            force_expl = self.stale_count >= 2

            proposal = propose_next_config(
                self.history, self.best_config, force_expl, self.llm_model
            )
            config = proposal["config"]
            print(f" [LLM REASONING]: {proposal['reasoning']}")
            print(f" [ACTION]: {config}")

            start_time = time.time()

            try:
                from experiments.run_patchcore import run as run_pc

                ckpt_path = os.path.join(self.checkpoint_dir, "memory_bank.pth")
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)

                metrics = run_pc(self.category, **config)
                duration = time.time() - start_time
                auroc = metrics.get("auroc", 0)
                score = self._calculate_score(auroc, duration)

                self.history.append(
                    {
                        "config": config,
                        "score": score,
                        "raw_auroc": auroc,
                        "duration": duration,
                    }
                )

                if score > self.best_score:
                    print(
                        f" [IMPROVED] Ky luc vo nat! Vot tu {self.best_score:.4f} -> {score:.4f} (Raw AUROC: {auroc:.4f}, Elapsed: {duration:.1f}s)"
                    )
                    self.best_score = score
                    self.best_config = config
                    self.stale_count = 0

                    self._persist_best_config(config, score)
                else:
                    print(
                        f" [DECLINED] Diem Multi-Obj {score:.4f} guc nga truoc ky luc {self.best_score:.4f}."
                    )
                    self.stale_count += 1

                self.tracker.log_experiment(
                    experiment_name=f"agentic_evo_v3_{self.category}_exp{len(self.history)}",
                    model=self.model_name,
                    category=self.category,
                    config=config,
                    metrics=metrics,
                    duration_seconds=duration,
                    notes=proposal["reasoning"][:200],
                )

                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"Iter {i} ({self.category}): {proposal['reasoning']}\nConfig: {config} -> Score: {score:.4f} [Raw: {auroc:.4f}]\n\n"
                    )

            except Exception as e:
                print(
                    f" [ERROR BẤT TỬ] Hàm huấn luyện Model bị lỗi: {e}. Vẫn kéo Script chạy vòng tới."
                )
                self.stale_count += 1

        import shutil

        best_bank = os.path.join(self.checkpoint_dir, "best_memory_bank.pth")
        current_bank = os.path.join(self.checkpoint_dir, "memory_bank.pth")
        if os.path.exists(best_bank):
            shutil.move(best_bank, current_bank)

        print(f"\n==================================================================")
        print(
            f" [DONE] NGHIEN CUU {self.category.upper()} CHINH THUC HOAN THIEN (V3.0)"
        )
        print(f" Best Multi-objective Score: {self.best_score:.4f}")
        print(f" Best JSON Config chốt sổ: {self.best_config}")
        print(f"==================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Autonomous AI ML-Engineer")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec category")
    parser.add_argument("--iterations", type=int, default=15, help="Số vòng tiến hóa")
    args = parser.parse_args()

    agent = AnonymousResearchAgent(args.category)
    agent.run(iterations=args.iterations)
