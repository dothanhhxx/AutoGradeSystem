# Hướng dẫn chạy AutoGradeSystem (Windows / PowerShell)

Dự án gồm 2 phần chạy song song: **Backend** (FastAPI, cổng 8000) và **Frontend** (Vite + React, cổng 5173).

## 1. Chuẩn bị (làm một lần)

Chạy tại thư mục gốc `d:\Github\AutoGradeSystem`:

```powershell
# Tạo môi trường ảo (bỏ qua nếu đã có .venv)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Backend
pip install -r app/requirements.txt

# (Tuỳ chọn) Thư viện cho research/, evaluation/, scripts/
pip install -r research/requirements.txt

# Frontend
cd frontend
npm install
cd ..
```

Yêu cầu: Python 3.10+, Node.js 18+. Lần chạy đầu sẽ tải model từ HuggingFace (~3 GB cho 5 model nhỏ).

## 2. Chạy ứng dụng

Mở **2 terminal riêng**.

**Terminal 1 — Backend**
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```
Đợi đến khi thấy `Grader initialized successfully!` (khoảng 1 phút trên CPU).
- Kiểm tra: http://127.0.0.1:8000/health → `{"status":"healthy","models_loaded":true}`
- Tài liệu API (Swagger): http://127.0.0.1:8000/docs

**Terminal 2 — Frontend**
```powershell
cd frontend
npm run dev
```
Mở http://localhost:5173

## 3. Chế độ LLM

Mặc định `skip_llm = True` trong [app/config.py](../app/config.py): bỏ qua Qwen2.5-3B, khởi động nhanh, nhận xét sinh bằng rule-based.
Muốn nhận xét chi tiết từ LLM: đặt `skip_llm = False` (cần tải ~7 GB; nên có GPU, bật `use_4bit_quantization = True` nếu dùng NVIDIA).

## 4. Thử nhanh API bằng PowerShell

```powershell
$b = @{
  context   = "Photosynthesis is the process by which green plants use sunlight to synthesize foods."
  question  = "Explain photosynthesis."
  reference = "Photosynthesis is the process by which green plants use sunlight to synthesize foods."
  student   = "Plants use sunlight, CO2 and water to make glucose and oxygen."
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/grade -ContentType "application/json" -Body $b
```

## 5. Các lệnh khác

| Mục đích | Lệnh |
|----------|------|
| Ablation Study | `python -m research.run_experiments` |
| ML Weight Optimizer | `python -m research.run_optimizer` |
| LLM Benchmark | `python -m research.run_llm_benchmark` |
| Xuất Excel từ kết quả chấm mẫu | `python scripts/create_excel.py` |
| Kiểm tra type frontend | `cd frontend; npx tsc -b` |
| Build frontend | `cd frontend; npm run build` |

Chạy các lệnh `python -m research...` từ thư mục gốc để import `app`, `evaluation` hoạt động.

## 6. Lỗi thường gặp

| Lỗi | Cách xử lý |
|-----|-----------|
| `UnicodeEncodeError: 'charmap' ...` khi khởi động backend | Đã sửa trong `app/__init__.py` (ép stdout UTF-8). |
| Cổng 8000 / 5173 bị chiếm | Đổi `--port` hoặc tắt tiến trình đang dùng cổng. |
| Frontend báo không kết nối được backend | Kiểm tra `/health` của backend trước. |
| Tải model chậm / bị giới hạn | Chạy `hf auth login` và nhập token HuggingFace. |
| `ModuleNotFoundError: openpyxl` | `pip install -r research/requirements.txt` |
