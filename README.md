PHÁT HIỆN TIN GIẢ TRÊN MẠNG XÃ HỘI DỰA TRÊN CẢM XÚC KÉP





💡 Giới thiệu (Introduction)

Sự bùng nổ của mạng xã hội đã khiến tin giả (Fake News) lan truyền mất kiểm soát, gây ảnh hưởng tiêu cực đến an ninh thông tin và nhận thức cộng đồng. Các phương pháp phát hiện truyền thống dựa trên nội dung văn bản thường thất bại trước các tin giả được ngụy tạo tinh vi.







Dự án này đề xuất giải pháp tiếp cận Lai ghép (Hybrid Approach) dựa trên lý thuyết Cảm xúc kép (Dual Emotion). Hệ thống không chỉ phân tích nội dung mà còn khai thác sự mâu thuẫn tâm lý giữa Người đăng (Publisher) và Cộng đồng (Social) để vạch trần tin giả với độ chính xác cao hơn.





📂 Cấu trúc thư mục dự án

Project\_Root/

│

├── 📜 code\_NL.py                 # Mã nguồn chính (Pipeline toàn bộ quy trình)

├── 📜 requirements.txt           # Danh sách thư viện phụ thuộc

├── 📜 README.md                  # Tài liệu hướng dẫn

│

└── 📂 PHEME\_veracity             # Thư mục chứa Dataset (Cần tải về)

&nbsp;   ├── charliehebdo/             # Sự kiện Charlie Hebdo

&nbsp;   ├── ferguson/                 # Sự kiện Ferguson

&nbsp;   └── ... (7 sự kiện khác) 



⚙️ Quy trình thực hiện (Methodology)

Dự án tuân thủ quy trình khoa học dữ liệu chặt chẽ gồm 4 bước chính:



1️⃣ Thu thập \& Tiền xử lý (Data Processing)



Nguồn dữ liệu: PHEME Dataset (9 sự kiện khẩn cấp trên Twitter).
Tải và giải nén file từ: https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078
Sau khi tải về ta được file 6392078.zip -> Giải nén ra Thư mục 6392078 chứa thư mục PHEME_veracity.
Nhấn vào thư mục PHEME_veracity, ta thấy thư mục all-rnr-annotated-threads -> Hãy đổi tên thư mục này thành PHEME_veracity sau đó cắt/copy thư mục này vào cùng cấp với file code\_NL.py như mô tả cấu trúc dự án




Sàng lọc: Chỉ sử dụng dữ liệu có nhãn TRUE (0) và FALSE (1), loại bỏ tin chưa xác minh (Unverified).





Làm sạch: Chuyển chữ thường, loại bỏ URL, User Mentions (@user), Hashtags (#) và ký tự đặc biệt.



2️⃣ Trích xuất Đặc trưng (Feature Engineering)

Hệ thống xây dựng vector đặc trưng kết hợp từ 2 luồng thông tin:





Phân tích Cảm xúc (Emotion Analysis): Sử dụng mô hình DistilRoBERTa (j-hartmann/emotion-english) để trích xuất 7 trạng thái cảm xúc (Anger, Fear, Joy...).





Publisher Emotion: Cảm xúc chủ đạo của bài đăng gốc.





Social Emotion: Phân phối cảm xúc của các bình luận phản hồi.





Dual Emotion Gap: Tính toán khoảng cách/mâu thuẫn giữa hai luồng cảm xúc này.





Biểu diễn Ngữ nghĩa (Semantic Embedding): Sử dụng Sentence-BERT (all-MiniLM-L6-v2) để mã hóa nội dung văn bản thành vector ngữ nghĩa.



3️⃣ Huấn luyện \& Tối ưu hóa (Training)



Cân bằng dữ liệu: Sử dụng kỹ thuật SMOTE để sinh mẫu nhân tạo, giải quyết vấn đề mất cân bằng dữ liệu.





Mô hình hóa: Triển khai và so sánh 4 thuật toán:



Logistic Regression



Support Vector Machine (SVM - RBF Kernel)



Random Forest



XGBoost (Gradient Boosting)





Tối ưu tham số: Sử dụng GridSearchCV (5-fold Cross-validation) để tìm cấu hình tốt nhất.



🚀 Hướng dẫn Cài đặt \& Chạy

Bước 1: Cài đặt môi trường Yêu cầu Python 3.10 trở lên. Cài đặt các thư viện cần thiết:

pip install -r requirements.txt



Bước 2: Chuẩn bị dữ liệu Tải bộ dữ liệu PHEME và giải nén vào thư mục PHEME\_veracity.



Bước 3: Chạy chương trình

python code\_NL.py







