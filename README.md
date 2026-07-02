# 📊 Ứng dụng Học Máy Đánh giá Rủi ro Khách hàng (Mô hình 5C)

Ứng dụng web tác nghiệp này giúp tự động hóa quy trình phân tích và chấm điểm tín dụng dựa trên cấu trúc mô hình **5C (Character - Capacity - Condition - Capital - Collateral)** truyền thống kết hợp với thuật toán học máy **Hồi quy Logistic (Logistic Regression)**. 

Hệ thống được phát triển trực tiếp từ kịch bản phân tích dữ liệu nghiên cứu trong Notebook huấn luyện máy học gốc.

## 🛠️ Tính năng chính ứng dụng
- **Cấu hình tham số học máy động:** Tinh chỉnh trực tiếp tỷ lệ phân chia tập học/kiểm định và hệ số regularization $C$ của mô hình hồi quy ngay trên giao diện Sidebar.
- **Thống kê tổng quan tự động:** Tổng hợp phân phối, thông số phương sai, giá trị trung vị của toàn bộ 24 tiêu chí đặc trưng đầu vào.
- **Trực quan hóa đồ họa tương tác:** Vẽ biểu đồ tần suất Histogram phân tách trực quan dựa trên nền tảng thư viện đồ họa Plotly cao cấp.
- **Vận hành tác nghiệp linh hoạt:** - Chế độ nhập liệu đơn lẻ: Nhập nhanh điểm chấm hồ sơ qua các thanh trượt phân bổ khoa học theo cấu trúc nghiệp vụ 5C để nhận phản hồi xác suất rủi ro tức thì.
  - Chế độ dự báo hàng loạt: Cho phép nạp tệp danh sách hồ sơ khách hàng mới để chấm điểm đồng thời và hỗ trợ xuất báo cáo định dạng CSV Excel.

## 📁 Cấu trúc dữ liệu đầu vào chuẩn hóa
Tệp dữ liệu nạp vào hệ thống để chạy phân tích cần đảm bảo tối thiểu có **24 trường thuộc tính đánh giá từ thang điểm 1 đến 5** và 1 cột mục tiêu phân lớp nhãn nhị phân:
- **TC1 đến TC5**: Các chỉ số đánh giá về Tư cách (Character) của khách hàng.
- **NL1 đến NL4**: Các chỉ số đo lường Năng lực hoạt động (Capacity).
- **DK1 đến DK5**: Các tiêu chí đo lường Điều kiện kinh doanh kinh tế vĩ mô (Conditions).
- **V1 đến V6**: Nhóm chỉ số cấu trúc Vốn tự có nội tại (Capital).
- **TS1 đến TS4**: Nhóm thuộc tính giá trị pháp lý Tài sản đảm bảo (Collateral).
- **PD (Biến mục tiêu)**: Nhãn trạng thái hồ sơ thực tế (`0` - Hồ sơ an toàn, `1` - Hồ sơ phát sinh rủi ro tiềm ẩn).

## 🚀 Hướng dẫn triển khai cục bộ

### Bước 1: Sao chép mã nguồn và chuẩn bị môi trường máy trạm
Đảm bảo bạn đã cài đặt Python phiên bản từ 3.9 đến 3.12 trên hệ thống của mình.

### Bước 2: Cài đặt gói thư viện phụ thuộc bắt buộc
Di chuyển Terminal/Command Prompt vào thư mục chứa mã nguồn và thực thi câu lệnh:
```bash
pip install -r requirements.txt
