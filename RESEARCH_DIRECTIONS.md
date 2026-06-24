# Định hướng phát triển nghiên cứu hệ thống AutoGradeSystem

Tài liệu này tổng hợp chi tiết hai hướng phát triển nghiên cứu vừa sức, mang tính cải tiến và lai tạo từ hệ thống chấm điểm câu trả lời ngắn (Hybrid ASAG) hiện tại. Hai hướng này sử dụng lại hoàn toàn mã nguồn cốt lõi nhưng tạo ra những đóng góp học thuật mới rõ rệt.

---

## Hướng 1: Nghiên cứu so sánh và cắt tỉa mô hình (Ablation & Comparative Study)

### 1. Ý tưởng cốt lõi
Hệ thống hiện tại là một sự kết hợp (lai tạo) của 5 mô hình xử lý ngôn ngữ tự nhiên (NLP) chuyên biệt chạy song song và 1 mô hình ngôn ngữ lớn (LLM) để sinh phản hồi. Hướng nghiên cứu này tập trung vào việc **thay thế các mảnh ghép** trong pipeline để tìm ra sự kết hợp tối ưu nhất giữa độ chính xác, tốc độ xử lý (inference time) và tài nguyên phần cứng (RAM/VRAM).

### 2. Phương pháp thực hiện
- **Nghiên cứu cắt tỉa (Ablation Study):** 
  - Lần lượt vô hiệu hóa 1 trong 5 mô hình ở Layer Feature Extraction (ví dụ: tắt mô hình kiểm tra Formality hoặc Logic).
  - So sánh điểm số cuối cùng mà hệ thống chấm với điểm số do giáo viên (con người) chấm bằng các độ đo độ lỗi (như MAE, RMSE) hoặc độ đo tương quan (Pearson, Spearman).
  - **Mục tiêu:** Chứng minh đóng góp cụ thể của từng mô hình vào độ chính xác tổng thể. Ví dụ: Nếu bỏ mô hình Logic mà độ chính xác giảm mạnh, điều đó khẳng định tầm quan trọng của thành phần này trong hệ thống lai tạo của bạn.

- **Nghiên cứu so sánh (Comparative Study):**
  - **Với Layer Semantic:** Thử thay thế `princeton-nlp/sup-simcse-roberta-large` bằng các kiến trúc embedding mới hơn như `all-mpnet-base-v2` hoặc `BGE-m3`. So sánh xem model nào nắm bắt ngữ nghĩa tốt hơn và nhẹ hơn.
  - **Với Layer LLM Feedback:** Thay thế `Qwen2.5-3B` bằng `Llama-3-8B-Instruct`, `Gemma-2-2B-It` hoặc thậm chí sử dụng kết quả API của ChatGPT/Claude làm Baseline (mức chuẩn) để so sánh chất lượng sinh phản hồi dựa trên cùng một đầu vào Tag.
  
### 3. Ý nghĩa và Kết quả kỳ vọng
- Đưa ra được báo cáo chi tiết về tradeoff (sự đánh đổi) giữa tài nguyên sử dụng và độ chính xác của hệ thống, điều rất hay gặp trong các bài báo khoa học.
- Tìm ra cấu hình pipeline tối ưu nhất (Best Configuration) cho bài toán ASAG.
- **Gợi ý tên đề tài:** *"Đánh giá hiệu năng và phân tích thành phần của các mô hình ngôn ngữ trong hệ thống tự động chấm điểm câu trả lời ngắn"* hoặc *"Nghiên cứu so sánh các kiến trúc mô hình trong hệ thống Hybrid ASAG"*.

---

## Hướng 3: Học trọng số tự động (Data-driven Dynamic Weighting)

### 1. Ý tưởng cốt lõi
Trong Diagram 4 (Scoring & Aggregation Layer) hiện tại, hệ thống đang tính điểm cuối cùng dựa trên các trọng số tĩnh (static weights) được người dùng thiết lập, ví dụ: Semantic 20%, Coverage 20%, Logic 20%, v.v. Việc đặt trọng số tĩnh mang tính chủ quan (Rule-based). Hướng nghiên cứu này sẽ cải tiến bằng cách áp dụng **Machine Learning** để hệ thống "tự học" (Data-driven) xem thành phần nào thực sự quan trọng hơn để đưa ra điểm số giống với con người nhất.

### 2. Phương pháp thực hiện
- **Xây dựng bộ dữ liệu (Dataset):**
  - Chuẩn bị một tập dữ liệu khoảng vài trăm câu trả lời ngắn đã được giáo viên chấm điểm thực tế (Ground Truth Score).
- **Trích xuất đặc trưng (Feature Extraction):**
  - Chạy tập dữ liệu qua 5 mô hình của hệ thống để thu được 5 điểm số thành phần cho mỗi câu trả lời (`semantic_score`, `coverage_score`, `formality_score`, `grammar_score`, `logic_score`). Coi 5 điểm số này là đầu vào (Features vector - X).
  - Điểm của giáo viên là đầu ra mục tiêu (Target - Y).
- **Huấn luyện mô hình hồi quy (Regression Model):**
  - Sử dụng các thuật toán Machine Learning truyền thống (vừa sức, dễ triển khai, tốn ít tài nguyên và dễ giải thích) như: Linear Regression, Random Forest Regressor, hoặc Support Vector Regression (SVR).
  - Thay vì dùng công thức cộng dồn tĩnh, mô hình Machine Learning sẽ được học để tự tìm ra trọng số phân bổ, hoặc phát hiện mối liên hệ phi tuyến tính giữa 5 thành phần để dự đoán ra điểm số.

### 3. Ý nghĩa và Kết quả kỳ vọng
- Nâng cấp hệ thống từ phương pháp tĩnh "Rule-based scoring" sang phương pháp linh hoạt "Machine Learning-based scoring", tăng tính khách quan và nâng cao tự động hóa.
- Độ chính xác của hệ thống (so với điểm con người) dự kiến sẽ tăng đáng kể nhờ việc thuật toán tự điều chỉnh để khớp với "khẩu vị" chấm bài thực tế.
- Khả năng mở rộng tốt: Khi chấm điểm môn học khác, chỉ cần Re-train mô hình Regression nhỏ này là hệ thống tự động có trọng số mới mà không cần sửa code.
- **Gợi ý tên đề tài:** *"Cải thiện độ chính xác của hệ thống chấm điểm câu trả lời ngắn lai ghép bằng phương pháp tối ưu hóa trọng số dựa trên Machine Learning"* hoặc *"Áp dụng học máy trong việc dự đoán điểm số tổng hợp từ đa mô hình đánh giá ngôn ngữ"*.
