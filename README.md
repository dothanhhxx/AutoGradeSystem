# 🎓 Beyond Single LLMs: HybridASAG Grader

Hệ thống **chấm điểm câu trả lời ngắn tự động** (Automated Short Answer Grading - ASAG) chuẩn nghiên cứu (Research-grade) sử dụng kiến trúc lai đa mô hình (Hybrid Multi-Model) kết hợp 5 mô hình NLP chuyên biệt để chấm điểm (Rule-based Tags) và 1 LLM (Qwen2.5-3B) để sinh phản hồi tự nhiên.

Hệ thống được thiết kế hướng tới tính minh bạch (Explainable AI), giải quyết triệt để vấn đề "hộp đen" (Black-box) và ảo giác (Hallucination) của các mô hình ngôn ngữ lớn hiện tại.

---

## 🏗️ Kiến trúc & Mô hình (5+1 Layer Hybrid)

Hệ thống sử dụng 5 mô hình nhẹ chạy song song để chấm điểm 5 tiêu chí độc lập, và 1 LLM để tổng hợp kết quả:

| Lớp | Mô hình | Kích thước | Chức năng (Tiêu chí) |
|---|---|---|---|
| 1 | `princeton-nlp/sup-simcse-roberta-large` | ~1.3 GB | Ngữ nghĩa (Semantic Similarity) |
| 2 | `all-MiniLM-L6-v2` | ~80 MB | Độ phủ từ khóa (Keyword Coverage) |
| 3 | `cointegrated/roberta-base-formality` | ~500 MB | Văn phong học thuật (Formality) |
| 4 | `textattack/roberta-base-CoLA` | ~500 MB | Ngữ pháp tiếng Anh (Grammar) |
| 5 | `cross-encoder/nli-deberta-v3-base` | ~700 MB | Tính Logic / Phản biện (NLI) |
| **LLM** | `Qwen/Qwen2.5-3B-Instruct` | ~6.5 GB | Viết phản hồi (Feedback Synthesis) |

---

## ⚙️ Hướng dẫn Cài đặt & Chạy Local

### 1. Yêu cầu Hệ thống (Prerequisites)
- **Python** 3.10+
- **Node.js** 18+ & npm
- **RAM/VRAM:** Hệ thống yêu cầu ít nhất 16GB RAM. Nếu chạy full cấu hình (có LLM), khuyến nghị 32GB RAM hoặc GPU có VRAM >= 6GB.
- **Hugging Face Account:** Bắt buộc phải có Access Token để tải mô hình lớn (Qwen).

### 2. Cài đặt Dependencies
```bash
# Clone dự án
git clone https://github.com/dothanhhxx/AutoGradeSystem.git
cd AutoGradeSystem

# Cài Backend Dependencies
pip install -r app/requirements.txt

# Cài Frontend Dependencies
cd frontend
npm install
cd ..
```

### 3. Xác thực Hugging Face (Quan trọng)
Vì mô hình LLM có dung lượng lớn, Hugging Face sẽ bóp băng thông nếu bạn không đăng nhập.
Mở Terminal và gõ:
```bash
huggingface-cli login
```
*Dán Access Token của bạn vào (lấy tại huggingface.co/settings/tokens) và nhấn Enter.*

### 4. Tùy chỉnh Cấu hình (app/config.py)
Để tối ưu tốc độ hoặc tránh tràn RAM (OOM), bạn có thể chỉnh sửa `app/config.py`:
- `skip_llm = True`: Tắt hoàn toàn LLM (chỉ dùng 5 mô hình nhỏ). Hệ thống sẽ load siêu nhanh (~3s) và tốn < 2GB RAM. Thích hợp để test code.
- `use_4bit_quantization = True`: Bật nén 4-bit NF4 (Chỉ hỗ trợ nếu máy có GPU NVIDIA) để giảm RAM của LLM từ 12GB xuống 2.5GB.

---

## 🚀 Khởi chạy Hệ thống

Mở **2 Terminal riêng biệt**:

**Terminal 1 — Backend (FastAPI):**
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```
Backend chạy tại `http://localhost:8000` (API docs tự động tại `http://localhost:8000/docs`).

**Terminal 2 — Frontend (React + Vite):**
```bash
cd frontend
npm run dev
```
Giao diện chạy tại `http://localhost:5173`.

---

## 🧪 Tab Research Lab (Dành cho Nghiên cứu)

Giao diện Web được tích hợp sẵn Tab **Research Lab** phục vụ trực tiếp cho việc làm báo cáo/nghiên cứu khoa học:
- **System Configuration:** Hiển thị thông số Hyper-parameters đang chạy.
- **Ablation Study Runner:** Cho phép chạy thử nghiệm cắt tỉa (tắt từng thành phần để đánh giá độ quan trọng) trực tiếp trên Web. Được tăng tốc bằng `FeatureCache` giúp chạy hàng trăm mẫu chỉ trong vài giây.
- **LaTeX Generator:** Tự động tạo bảng kết quả chuẩn LaTeX (có P-value, Sig., F1, QWK) sẵn sàng để copy/paste vào Paper (Overleaf).

---

## 🗂️ Cấu trúc Project Mới

```text
AutoGradeSystem/
├── app/
│   ├── api.py          ← FastAPI Endpoints
│   ├── grader.py       ← Pipeline chấm điểm chính
│   ├── models.py       ← Cấu trúc dữ liệu & Tính điểm
│   └── config.py       ← Cấu hình hệ thống (Weights, LLM flags)
├── research/           ← [MỚI] Module phục vụ viết Paper
│   ├── ablation_v2.py  ← Thuật toán kiểm định thống kê (Wilcoxon)
│   ├── feature_cache.py← Hệ thống lưu đệm (Cache) tăng tốc thí nghiệm
│   └── run_experiments.py ← Script chạy Benchmark qua Terminal
└── frontend/
    └── src/
        ├── App.tsx     ← Giao diện chính (Grading + Research Lab)
        └── ...
```

---

## 🔌 API Endpoints Cơ Bản

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/grade` | Chấm điểm 1 câu trả lời đầy đủ (có LLM feedback) |
| `POST` | `/grade/recalculate`| Tính lại điểm khi đổi Weight (không cần gọi lại Model) |
| `POST` | `/research/ablation`| Chạy thử nghiệm Ablation Study |
| `GET`  | `/weights/presets`  | Lấy danh sách preset trọng số |

*(Cấu hình Weight Presets hiện hỗ trợ: balanced, content_focused, academic_writing, logic_heavy, quick_check).*
