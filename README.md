# Final-Project-PCA

Code gồm các phần chính trong thư mục Source: Reader, Vectorizer, PCA, Evaluate và Test_Data 
1. Phần Reader có file code reader_agent.py có chức năng đọc file bất kì (một số định dạng khả thi cho trước, như ở Test_Data)
2. Phần Vectorizer gồm file chính là vectorizer_module.py, file này có chức năng là số hoá dữ liệu từ đầu ra của phần Reader và sau đó chuẩn hoá nó về một khoảng scale (để cùng ngưỡng đánh giá sau này) và dạng dữ liệu là (samples, features) phù hợp với đầu vào PCA. Ngoài ra còn có file phụ tf_idf_module.py và vector_for_text_Way1.py để hỗ trợ cho phần số hoá chính.
3. Phần PCA là phần chính về hoạt động của PCA từ đầu đến cuối và có cả chuyển từ dữ liệu đã giảm chiều xây dựng lại về dữ liệu ban đầu.
4. Phần Test_Data gồm các file data với dữ liệu cơ bản đối với đầu vào Reader. Bới vì các dạng phức tạp quá mức chứa nhiều định dạng thì hệ thống đọc và số hoá thủ công sẽ gặp nhiều vấn đề khó xử lý. Nên dữ liệu cơ bản như pdf thì có thể chứ text bình thường và ảnh, không có phần thêm đặc biệt, ...

Và ở ngoài có hàm main.py là hàm main gọi các file trong Source để test data đánh giá theo cấu trúc, và có các file output (các file log) ghi lại đánh giá file.

Cách chạy CODE: 
1. CÁCH 1:
Tạo một thư mục trong máy và clone từ github sau về: https://github.com/mjngxwnj/Final-Project-PCA 
Sau đó trong terminal cd để tên thư mục đã tạo trên. Ví dụ tạo thư mục PCA_Nhom3 thì dùng lệnh cd PCA_Nhom3
Và cuối cùng chạy file main_for_teacher.py


2. CÁCH 2:
Nếu không dùng cách clone thì ta phải chỉnh lại đường dẫn đọc file ở trong file main_for_teacher.py rồi chạy
