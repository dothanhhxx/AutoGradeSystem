"""
Script tự động tải và tiền xử lý các tập dữ liệu Public ASAG:
1. SemEval-2013 Task 7 (Beetle & SciEntsBank)
2. Mohler (2011)

Lưu trực tiếp vào:
- data/semeval2013/
- data/mohler/
"""

import os
import json
from pathlib import Path

def setup_directories():
    """Tạo các thư mục lưu trữ dữ liệu chuẩn."""
    base_dir = Path(__file__).resolve().parent.parent / "data"
    semeval_dir = base_dir / "semeval2013"
    mohler_dir = base_dir / "mohler"

    semeval_dir.mkdir(parents=True, exist_ok=True)
    mohler_dir.mkdir(parents=True, exist_ok=True)

    print(f"[OK] Thư mục SemEval: {semeval_dir}")
    print(f"[OK] Thư mục Mohler:  {mohler_dir}")
    return semeval_dir, mohler_dir

def download_semeval(target_dir: Path):
    """Tải và convert SemEval-2013 từ Hugging Face sang JSON chuẩn ASAGSample."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("\n[!] Thư viện 'datasets' chưa được cài. Đang cài đặt nhanh...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets", "pandas"])
        from datasets import load_dataset

    print("\n--- Đang tải SemEval-2013 (SciEntsBank & Beetle) ---")
    datasets_to_fetch = [
        ("SciEntsBank", "nkazi/SciEntsBank"),
        ("Beetle", "nkazi/Beetle")
    ]

    all_samples = []

    for name, hf_id in datasets_to_fetch:
        print(f"-> Đang tải {name} ({hf_id})...")
        try:
            ds = load_dataset(hf_id)
            split_name = "train" if "train" in ds else list(ds.keys())[0]
            data = ds[split_name]

            count = 0
            for idx, item in enumerate(data):
                # Map fields linh hoạt theo schema của huggingface
                q = item.get("question") or item.get("Question") or ""
                r = item.get("referenceAnswer") or item.get("reference") or item.get("Reference") or ""
                s = item.get("studentAnswer") or item.get("student") or item.get("StudentAnswer") or ""
                lbl = item.get("accuracy") or item.get("label") or item.get("GoldAnswer") or "unknown"

                sample = {
                    "id": f"{name.lower()}_{idx:05d}",
                    "dataset": name.lower(),
                    "question": str(q).strip(),
                    "reference_answer": str(r).strip(),
                    "student_answer": str(s).strip(),
                    "gold_label": str(lbl).strip().lower()
                }
                all_samples.append(sample)
                count += 1

            print(f"   [+] Đã tải thành công {count} mẫu từ {name}.")
        except Exception as e:
            print(f"   [-] Không thể tải tự động {name}: {e}")

    if all_samples:
        out_file = target_dir / "semeval2013_all.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        print(f"[THÀNH CÔNG] Đã lưu tổng cộng {len(all_samples)} mẫu vào: {out_file}")

def download_mohler(target_dir: Path):
    """Tải và convert Mohler (2011) sang JSON chuẩn."""
    try:
        from datasets import load_dataset
    except ImportError:
        from datasets import load_dataset

    print("\n--- Đang tải Mohler Dataset (2011) ---")
    try:
        ds = load_dataset("nkazi/MohlerASAG")
        split_name = "train" if "train" in ds else list(ds.keys())[0]
        data = ds[split_name]

        samples = []
        for idx, item in enumerate(data):
            q = item.get("Question") or item.get("question") or ""
            r = item.get("Desired_Answer") or item.get("reference") or item.get("Reference") or ""
            s = item.get("Student_Answer") or item.get("student") or item.get("StudentAnswer") or ""
            score = item.get("Score") or item.get("score") or item.get("Score_Avg") or 0.0

            try:
                score_val = float(score)
            except:
                score_val = 0.0

            # Map điểm sang nhãn 3-way
            if score_val >= 4.0:
                label = "correct"
            elif score_val >= 2.5:
                label = "partially_correct_incomplete"
            else:
                label = "contradictory"

            samples.append({
                "id": f"mohler_{idx:05d}",
                "dataset": "mohler",
                "question": str(q).strip(),
                "reference_answer": str(r).strip(),
                "student_answer": str(s).strip(),
                "gold_score": score_val,
                "gold_label": label
            })

        out_file = target_dir / "mohler_dataset.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"[THÀNH CÔNG] Đã lưu {len(samples)} mẫu Mohler vào: {out_file}")

    except Exception as e:
        print(f"   [-] Không thể tải tự động Mohler: {e}")

if __name__ == "__main__":
    semeval_dir, mohler_dir = setup_directories()
    download_semeval(semeval_dir)
    download_mohler(mohler_dir)
    print("\nHoàn tất! Dữ liệu đã sẵn sàng trong thư mục data/.")

