# 🎓 Beyond Single LLMs — HybridASAG Grader

> **"Beyond Single LLMs: A Hybrid Multi-Model Framework for Explainable Short Answer Grading"**

Hệ thống **Chấm điểm Câu trả lời Ngắn Tự động** (Automated Short Answer Grading - ASAG) chuẩn nghiên cứu (Research-grade). Thay vì giao toàn bộ việc chấm cho một LLM duy nhất (dẫn đến "hộp đen" và ảo giác), hệ thống tách nhiệm vụ cho **5 mô hình NLP chuyên biệt nhẹ** để chấm điểm minh bạch, và **1 LLM (Qwen2.5-3B)** chỉ đóng vai trò diễn đạt lại thành lời nhận xét tự nhiên.

---

## 🏗️ Kiến trúc Hệ thống (5 + 1 Hybrid)

```
[Câu hỏi + Đáp án chuẩn + Câu trả lời học sinh]
        │
        ▼
┌──────────────────────────────────────────────────────┐
│            Layer 1: Feature Extraction               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │ SimCSE   │ │ KeyBERT  │ │Formality │ │  CoLA   │ │
│  │Semantic  │ │Coverage  │ │ RoBERTa  │ │ Grammar │ │
│  └──────────┘ └──────────┘ └──────────┘ └─────────┘ │
│                 ┌──────────────────┐                  │
│                 │  DeBERTa v3 NLI  │                  │
│                 │  Logic/Coherence │                  │
│                 └──────────────────┘                  │
└──────────────────────────────────────────────────────┘
        │  5 điểm thành phần (0.0 – 1.0)
        ▼
┌──────────────────────────────────────────────────────┐
│           Layer 2: Aggregation & Synthesis           │
│  [Weighted Sum → Điểm tổng] + [Qwen2.5-3B → Nhận xét]│
└──────────────────────────────────────────────────────┘
        │
        ▼
  [Điểm 0–100] + [Radar Chart] + [Feedback chi tiết]
```

### Bảng Model

| # | Mô hình | Kích thước | Tiêu chí |
|---|---------|-----------|---------|
| 1 | `princeton-nlp/sup-simcse-roberta-large` | ~1.3 GB | Ngữ nghĩa (Semantic Similarity) |
| 2 | `all-MiniLM-L6-v2` (KeyBERT backend) | ~80 MB | Độ phủ Từ khóa (Keyword Coverage) |
| 3 | `cointegrated/roberta-base-formality` | ~500 MB | Văn phong học thuật (Formality) |
| 4 | `textattack/roberta-base-CoLA` | ~500 MB | Ngữ pháp (Grammar) |
| 5 | `cross-encoder/nli-deberta-v3-base` | ~700 MB | Tính Logic / Phản biện (NLI) |
| LLM | `Qwen/Qwen2.5-3B-Instruct` | ~6.5 GB | Tổng hợp nhận xét (Feedback) |

---

## ⚙️ Cài đặt & Khởi chạy

### 1. Yêu cầu Hệ thống
- **Python** 3.10+, **Node.js** 18+ & npm
- **RAM:** Tối thiểu 8 GB (chế độ `skip_llm`). Khuyến nghị 32 GB để chạy full LLM trên CPU.
- **Ổ cứng:** ~10 GB cho 5 model nhỏ + ~7 GB cho Qwen2.5-3B

### 2. Cài đặt Dependencies
```bash
git clone https://github.com/dothanhhxx/AutoGradeSystem.git
cd AutoGradeSystem

# Backend
pip install -r app/requirements.txt

# Frontend
cd frontend && npm install && cd ..
```

### 3. Đăng nhập Hugging Face (Bắt buộc để tải Qwen2.5-3B)
Nếu bỏ qua bước này, tốc độ tải model sẽ bị bóp băng thông (rate limit) rất nặng.
```bash
hf auth login
# Nhập Access Token từ huggingface.co/settings/tokens
```

### 4. Tùy chỉnh `app/config.py` (Khuyến nghị)

| Flag | Giá trị mặc định | Mô tả |
|------|---------|---------|
| `skip_llm` | `True` | Bỏ qua Qwen2.5-3B. Load siêu nhanh (~5s), tốn < 3 GB RAM. Hệ thống tự động dùng Rule-based Fallback để sinh nhận xét. |
| `use_4bit_quantization` | `False` | Nén Qwen 4-bit NF4 (cần GPU NVIDIA + `bitsandbytes`). Giảm VRAM từ 12 GB xuống 2.5 GB. |

### 5. Khởi chạy
Mở **2 Terminal riêng biệt**:

```bash
# Terminal 1 — Backend API
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
cd frontend && npm run dev
```

- **Backend:** `http://localhost:8000` · Swagger docs: `http://localhost:8000/docs`
- **Frontend:** `http://localhost:5173`

> ⚠️ **Lần đầu khởi động:** Nếu `skip_llm = False`, hệ thống cần tải ~7 GB. Có thể mất 5–30 phút tuỳ tốc độ mạng và đường truyền HuggingFace.

---

## 🗂️ Cấu trúc Project

```
AutoGradeSystem/
├── app/
│   ├── api.py              ← FastAPI endpoints (REST API)
│   ├── grader.py           ← Pipeline chấm điểm chính (HybridASAGGrader)
│   ├── models.py           ← Data classes (MetricScores, GradingResult, LLMFeedback)
│   ├── config.py           ← Cấu hình (model names, thresholds, weight presets)
│   └── requirements.txt
│
├── research/               ← Module nghiên cứu (dành cho Paper)
│   ├── ablation_v2.py      ← Ablation Study + kiểm định thống kê (Wilcoxon, Bonferroni)
│   ├── feature_cache.py    ← Cache đặc trưng tăng tốc thí nghiệm (pkl)
│   ├── weight_optimizer.py ← Tối ưu trọng số bằng ML (LinearRegression, RandomForest)
│   ├── llm_benchmark.py    ← So sánh LLM (JSON compliance, latency, RAM)
│   ├── run_experiments.py  ← Script chạy Ablation Study từ Terminal
│   ├── run_optimizer.py    ← Script chạy ML Weight Optimizer từ Terminal
│   └── run_llm_benchmark.py← Script chạy LLM Benchmark (simulated / real mode)
│
├── evaluation/
│   └── evaluation_framework.py ← DataLoader (SemEval), ASAGSample, metrics
│
├── experiments/            ← Thư mục kết quả thí nghiệm (tự động tạo)
│   ├── cache/              ← Feature cache (.pkl)
│   └── results/
│       ├── ablation/       ← CSV + LaTeX từ Ablation Study
│       ├── optimizer/      ← Trọng số tối ưu từ ML (ml_weights.json)
│       └── llm_benchmark/  ← CSV + LaTeX từ LLM Benchmark
│
└── frontend/
    └── src/
        ├── App.tsx         ← Giao diện chính (Grading + Research Lab)
        ├── types.ts        ← TypeScript interfaces
        └── mockData.ts     ← Dữ liệu mô phỏng (Simulated Mode)
```

---

## 🔌 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/health` | Kiểm tra trạng thái backend & model |
| `POST` | `/grade` | Chấm điểm 1 câu trả lời |
| `POST` | `/grade/batch` | Chấm điểm nhiều câu (batch) |
| `POST` | `/grade/recalculate` | Tính lại điểm với trọng số mới (không cần re-inference) |
| `GET` | `/weights/presets` | Lấy danh sách preset trọng số |
| `GET` | `/thresholds` | Lấy ngưỡng phân loại hiện tại |
| `POST` | `/research/ablation` | Kích hoạt Ablation Study từ giao diện Web |

---

## ⚖️ Weight Presets

| Preset | Semantic | Coverage | Formality | Grammar | Logic | Ghi chú |
|--------|----------|----------|-----------|---------|-------|---------|
| `balanced` | 20% | 20% | 20% | 20% | 20% | Mặc định |
| `content_focused` | 40% | 30% | 5% | 10% | 15% | Ưu tiên nội dung |
| `academic_writing` | 20% | 15% | 25% | 25% | 15% | Chấm bài luận |
| `logic_heavy` | 25% | 15% | 10% | 15% | 35% | Môn Toán/Logic |
| `quick_check` | 50% | 30% | 5% | 5% | 10% | Kiểm tra nhanh |
| `ml_optimized` | 20% | 30% | 0% | 10% | 40% | **Học máy tối ưu** — dựa trên Phase 3 |

---

## 🧪 Research Scripts (Dành cho Bài báo)

### Phase 1: Ablation Study
```bash
python research/run_experiments.py
# Kết quả: experiments/results/ablation/ablation_results.csv + ablation_latex.tex
```

### Phase 3: ML Weight Optimizer
```bash
python research/run_optimizer.py
# Kết quả: experiments/results/optimizer/ml_weights.json
```

### Phase 4: LLM Comparative Benchmark
```bash
# Chế độ mô phỏng — nhanh, không cần tải model thêm
python research/run_llm_benchmark.py --mode simulated

# Chế độ thực — cần tải Phi-3-Mini (~7 GB) và GPU/RAM lớn
python research/run_llm_benchmark.py --mode real

# Kết quả: experiments/results/llm_benchmark/llm_comparison.csv + llm_comparison_latex.tex
```

---

## 🐛 Sự cố thường gặp

| Lỗi | Nguyên nhân | Cách fix |
|-----|------------|---------|
| `POST /grade → 500` | `skip_llm=True` nhưng `generate_feedback()` cố gọi tokenizer `None` | **Đã fix** trong `grader.py` — cập nhật code mới nhất |
| Server kẹt ở `[6/6] Loading Qwen...` mãi không xong | Chưa đăng nhập HF, bị rate limit (~160 KB/s) | Chạy `hf auth login` rồi restart server |
| `UnicodeEncodeError: 'charmap' codec` trên Windows | Terminal Windows không hỗ trợ ký tự `−` (Unicode minus) | **Đã fix** trong `ablation_v2.py` — cập nhật code mới nhất |
| `OOM / Memory allocation failed` khi load Qwen | RAM không đủ (~12 GB cần) | Đặt `skip_llm = True` hoặc `use_4bit_quantization = True` |
