# Đồ án: Tiền xử lý và nhị phân hóa ảnh tài liệu (Binarization)

Dự án xử lý 4 loại ảnh tài liệu bị suy biến thực tế: đốm bẩn tài liệu cổ (`dom_ban`), hóa đơn in nhiệt mờ đứt nét (`dut_net`), ảnh nhòe tiêu cự (`nhieu_hat`), và ảnh chụp bị bóng đổ (`sang_khong_deu`).

---

## 1. Cài đặt thư viện

Mở terminal và chạy lệnh:
pip install opencv-python numpy scikit-image pillow pandas streamlit

---

## 2. Thứ tự chạy chương trình

Lưu ý: Mở terminal đứng tại thư mục gốc của dự án.

- Bước 1 (Tiền xử lý & phẳng nền):
  python src/preprocessing.py

- Bước 2 (Phân ngưỡng đa ứng viên - Thresholding):
  python src/thresholding.py

- Bước 3 (Lọc rác, nối nét & hoàn thiện - Morphology):
  python src/morphology.py
  (Ảnh kết quả cuối cùng sẽ lưu tại: data/processed/)

- Bước 4 (Đánh giá kết quả không cần Ground Truth):
  python src/evaluation.py
  (Bảng điểm chi tiết được xuất ra: evaluation_no_reference_report.csv)

- Chạy giao diện demo xem trực quan:
  streamlit run app.py

---

## 3. Cấu trúc thư mục chính

- data/raw/: Chứa ảnh gốc ban đầu theo 4 nhóm lỗi.
- data/processed/: Chứa ảnh sạch cuối cùng sau khi chạy xong pipeline.
- src/: Chứa 4 file code xử lý chính (preprocessing.py, thresholding.py, morphology.py, evaluation.py).
- evaluation_no_reference_report.csv: Bảng số liệu kết quả đánh giá 40 ảnh.

---

## 4. Tóm tắt kết quả thực nghiệm

Do tập dữ liệu là tài liệu thực tế không có sẵn ảnh gốc sạch (Ground Truth), nhóm sử dụng phương pháp đánh giá No-Reference (đo độ bám biên nét chữ và độ sạch nền):
- Tổng số ảnh xử lý: 40/40 ảnh.
- Độ bám biên chữ trung bình (Edge Recall): 89.84% (giữ nét chữ tốt, không rụng chữ).
- Độ sạch nền (Background Purity): 100.0% (sạch hoàn toàn hạt tiêu và bóng viền).
- Điểm chất lượng trung bình: 99.73 / 100 điểm.