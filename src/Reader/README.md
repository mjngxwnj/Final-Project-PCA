## 1. Mục tiêu chung của đoạn mã
Lớp DataExpander được thiết kế nhằm mục tiêu tự động phát hiện và trích xuất nội dung dữ liệu từ nhiều loại định dạng tập tin : PDF, DOCX, XLSX, CSV, TXT, JSON, HTML, ảnh (JPEG/PNG), MP3.
Cấu trúc đầu ra tiêu chuẩn:
	type: kiểu dữ liệu (text, table, image, audio, error)
	content: nội dung trích xuất được
	meta: thông tin metadata (tên file, số dòng, số từ, định dạng...)
## 2. Các thư viện sử dụng
	os, json: Xử lý hệ thống file và đọc JSON.
	pandas, numpy: Xử lý dữ liệu dạng bảng.
	docx: Đọc file Word (.docx).
	pdfplumber: Đọc văn bản từ file PDF.
	BeautifulSoup: Phân tích HTML.
	librosa: Trích xuất dữ liệu từ file âm thanh MP3.
	cv2: Đọc và xử lý ảnh.
## 3. Cấu trúc lớp DataExpander
### 3.1. Hàm khởi tạo:__init__
python
CopyEdit
```bash
    def __init__(self):
        pass
```
Không có biến khởi tạo đặc biệt. Đơn giản là tạo một đối tượng lớp.

## 4. Hàm chính: expand(self, file_path, modality=None)
Xác định loại file theo đuôi mở rộng (extension), sau đó gọi các hàm đọc tương ứng.
Luồng xử lý:
1.	Tách phần mở rộng của file từ file_path.
2.	Kiểm tra và gọi đúng hàm đọc: _read_pdf, _read_docx, _read_csv,...
3.	Nếu không hỗ trợ định dạng → báo lỗi.
## 5. Các hàm xử lý định dạng cụ thể
### 5.1. _read_pdf(self, file_path)
	Dùng pdfplumber để đọc từng trang PDF, ghép lại nội dung.
	Trả về type = text, content = dạng văn bản
### 5.2. _read_docx(self, file_path)
	Dùng docx.Document() đọc nội dung từng đoạn (paragraph).
	Trả về type = text, content = dạng văn bản
### 5.3. _read_xlsx(self, file_path, sheet_name=0)
	Đọc file excel pandas.read_excel() để đọc file Excel.
	Trả về type = table, content = pd.DataFrame()
### 5.4. _read_csv(self, file_path, delimiter=',')
	Đọc file CSV với pandas.read_csv.
	Trả về type = table, content = pd.DataFrame()
### 5.5. _read_json(self, file_path)
	Đọc JSON 
	Trả về type = table, content = pd.DataFrame()
### 5.6. _read_txt(self, file_path)
	Đọc file văn bản thuần (.txt) 
	Trả về type = text, content = dạng văn bản
 
### 5.7. _read_html(self, file_path)
	Sử dụng Beautiful Soup để đọc các thẻ văn bản trong html
	Trả về type = text, content = dạng văn bản.
### 5.8. _read_image(self, file_path)
	Dùng OpenCV (cv2) đọc ảnh → chuyển sang RGB.
	Trả về type = image, content = chuỗi ma trận 3x3 dạng numpy
### 5.9. _read_mp3(self, file_path)
	Dùng librosa để load file MP3.
	Trả về type = audio, content = chuỗi ma trận n phần tử dạng numpy
