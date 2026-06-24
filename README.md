# 🎓 HybridASAG Grader

Hệ thống **chấm điểm câu trả lời ngắn tự động** (Automated Short Answer Grading) sử dụng pipeline AI lai ghép gồm 5 mô hình NLP chuyên biệt và một LLM sinh phản hồi.

---

## ⚙️ Hướng dẫn Cài đặt & Chạy Local

### 1. Yêu cầu (Prerequisites)

- **Python** 3.10 trở lên
- **Node.js** 18 trở lên + npm
- **GPU (khuyến nghị)**: CUDA-compatible GPU hoặc Apple Silicon (MPS) để tăng tốc inference. CPU vẫn hoạt động nhưng chậm hơn đáng kể.
- Khoảng **10–15 GB dung lượng ổ đĩa** để tải các mô hình AI lần đầu

---

### 2. Cài đặt

```bash
# Clone dự án
git clone https://github.com/dothanhhxx/AutoGradeSystem.git
cd AutoGradeSystem

# Cài Backend Dependencies
pip install -r app/requirements.txt
```

> **Lưu ý GPU:** Nếu muốn dùng CUDA, hãy cài PyTorch phù hợp với phiên bản CUDA của bạn trước:
> ```bash
> # Ví dụ với CUDA 12.1
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```
> Sau đó mới chạy lệnh `pip install -r app/requirements.txt`.

```bash
# Cài Frontend Dependencies
cd frontend
npm install
cd ..
```

---

### 3. Tải mô hình AI (lần đầu chạy)

Các mô hình sau sẽ được tải **tự động** khi khởi động backend lần đầu tiên (cần kết nối Internet):

| Mô hình | Dung lượng | Chức năng |
|---------|-----------|-----------|
| `princeton-nlp/sup-simcse-roberta-large` | ~1.3 GB | Semantic similarity |
| `all-MiniLM-L6-v2` | ~80 MB | Keyword extraction |
| `cointegrated/roberta-base-formality` | ~500 MB | Formality detection |
| `textattack/roberta-base-CoLA` | ~500 MB | Grammar checking |
| `cross-encoder/nli-deberta-v3-base` | ~700 MB | Logic/NLI |
| `Qwen/Qwen2.5-3B-Instruct` | ~6 GB | Feedback generation |

---

### 4. Khởi chạy hệ thống

Mở **2 Terminal riêng biệt**:

**Terminal 1 — Backend (FastAPI):**
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```
Backend chạy tại `http://localhost:8000`  
API docs tự động tại `http://localhost:8000/docs`

> ⏳ Lần đầu khởi động sẽ mất **5–15 phút** để tải tất cả mô hình AI vào bộ nhớ.  
> Kiểm tra trạng thái tại: `http://localhost:8000/health`

**Terminal 2 — Frontend (React + Vite):**
```bash
cd frontend
npm run dev
```
Giao diện chạy tại `http://localhost:5173`

---

### 5. Chạy không cần Backend (Demo Mode)

Frontend hỗ trợ chế độ **Simulated Mode** — chạy hoàn toàn trên client, không cần khởi động backend.

- Mở `http://localhost:5173` (chỉ cần Terminal 2)
- Giao diện mặc định ở **Simulated Mode** (góc trên bên phải)
- Mọi tính năng chấm điểm vẫn hoạt động với dữ liệu mô phỏng

---

## 🗂️ Cấu trúc Project

```
AutoGradeSystem/
├── app/
│   ├── api.py          ← FastAPI endpoints
│   ├── grader.py       ← Core grading pipeline
│   ├── models.py       ← Pydantic models & grade calculation
│   ├── config.py       ← Model config & weight presets
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.tsx     ← Main React application
    │   ├── types.ts    ← TypeScript interfaces
    │   └── mockData.ts ← Simulation data
    └── package.json
```

---

## 🔌 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/health` | Kiểm tra trạng thái backend & GPU |
| `POST` | `/grade` | Chấm điểm 1 câu trả lời |
| `POST` | `/grade/batch` | Chấm điểm nhiều câu trả lời |
| `POST` | `/grade/recalculate` | Tính lại điểm với trọng số mới |
| `GET` | `/weights/presets` | Lấy danh sách preset trọng số |
| `GET` | `/thresholds` | Lấy ngưỡng phân loại hiện tại |

Xem đầy đủ tại `http://localhost:8000/docs` (Swagger UI).

---

## 🏷️ Weight Presets

| Preset | Semantic | Coverage | Formality | Grammar | Logic |
|--------|----------|----------|-----------|---------|-------|
| `balanced` | 20% | 20% | 20% | 20% | 20% |
| `content_focused` | 40% | 30% | 5% | 10% | 15% |
| `academic_writing` | 20% | 15% | 25% | 25% | 15% |
| `logic_heavy` | 25% | 15% | 10% | 15% | 35% |
| `quick_check` | 50% | 30% | 5% | 5% | 10% |
