# Per-criterion robustness study (Option 1 + CDI)

Kiểm tra grader Hybrid 5+1 có bị "lừa" không, tiêu chí nào bị lừa, và dùng bất đồng giữa các
tiêu chí (gồm Cognitive Discrepancy Index, CDI = coverage − logic) để phát hiện.

## Dữ liệu (thật, CC-BY-4.0)

| Dataset | Nguồn | Split dùng |
|---|---|---|
| SciEntsBank | SemEval-2013 Task 7, HF `nkazi/SciEntsBank` | train / dev (20% câu hỏi của train) / test_ua / test_uq / test_ud |
| Beetle | SemEval-2013 Task 7, HF `nkazi/Beetle` | train / dev / test_ua / test_uq |
| Mohler | Mohler et al. 2011, HF `Viswa45/ASAG` (open-ended) | 60/20/20 theo câu hỏi |

Nhãn 3-way: correct / partial / incorrect (contradictory, irrelevant, non_domain → incorrect).
Mohler: trung bình 2 người chấm ≥ 4.0 → correct, ≥ 2.5 → partial.
`question_id` = mã câu hỏi chính thức (vd. `FF.Q2`), không dùng nội dung câu hỏi vì
"Explain your reasoning." xuất hiện ở nhiều câu hỏi khác nhau.

Tải lại dữ liệu thô:

```python
from huggingface_hub import snapshot_download
for rid in ["nkazi/SciEntsBank", "nkazi/Beetle", "Viswa45/ASAG"]:
    snapshot_download(rid, repo_type="dataset", local_dir=f"data/raw/{rid.split('/')[1]}",
                      allow_patterns=["*.parquet", "*.md"])
```

## Chạy toàn bộ

```bash
python -m research.run_robustness prepare          # data/processed + splits.json
python -m research.run_robustness extract          # 5 đặc trưng cho mọi câu trả lời sạch (GPU, ~3 phút)
python -m research.run_robustness fit              # trọng số NNLS (train) + ngưỡng (dev) -> clean_results.csv
python -m research.run_robustness generate --llm   # tấn công theo luật + A2/A4 bằng Phi-3.5-mini
python -m research.run_robustness validate auto    # loại mẫu sinh hỏng
python -m research.run_robustness baselines finetune   # DeBERTa-v3-base, 3 seed
python -m research.run_robustness baselines llm        # Qwen2.5-3B holistic + analytic
python -m research.run_robustness eval             # attack_results.csv, criterion_shift.csv
python -m research.run_robustness detect           # D0..D3, leave-one-family-out
python -m research.run_robustness adaptive --llm   # tấn công biết bộ phát hiện
python -m research.run_robustness validate export  # phiếu chấm cho 2 người
python -m research.run_robustness report           # REPORT.md + figures/
pytest tests/robustness -m "not slow"              # test nhanh; bỏ -m để chạy cả test so khớp grader
```

Kết quả: `experiments/results/robustness/`. Cache đặc trưng và dự đoán:
`experiments/cache/robustness/` (khóa = hash nội dung câu hỏi + đáp án + bài làm).

## Thiết kế

- **Hệ bị tấn công** = đúng các model của `app/grader.py` (SimCSE, KeyBERT, RoBERTa-CoLA,
  RoBERTa-formality, DeBERTa-v3 NLI), chạy theo batch; `tests/robustness/test_features.py`
  so khớp với hàm từng mẫu của grader. Điểm = `calculate_final_grade` (gồm các luật cap).
- **Threat model**: học sinh biết thuật ngữ của chủ đề (xấp xỉ bằng từ khóa của đáp án mẫu),
  không biết đáp án mẫu. Bảng tấn công: xem docstring của `attacks.py`. A6, A7 là phép thử
  white-box riêng cho grader này và báo cáo tách biệt.
- **Mốc so sánh**: mọi ASR phải so với dòng `CLEAN_incorrect` (tỉ lệ câu trả lời sai *thật*
  bị chấm trên mức "incorrect").
- **Bộ phát hiện** chỉ đánh dấu để giáo viên xem, không tự trừ điểm. Ngưỡng chọn sao cho 5%
  câu trả lời thật ở dev bị đánh dấu.

## Việc cần người làm

Kiểm định nhãn bằng người: `validate export` tạo `validation/annotator_1.csv`,
`annotator_2.csv` (chấm mù, 300 mẫu, có 10% câu trả lời thật làm câu kiểm tra) và
`ANNOTATION_GUIDE.md`. Hai người điền độc lập, sau đó chạy `validate agree`.
Kiểm tra tự động (`validate auto`) chỉ loại lỗi sinh, không thay được bước này.
