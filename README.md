# Đồ án cuối kỳ CS231 — Hệ thống Nhận diện Khuôn mặt

Nghiên cứu so sánh nhiều pipeline nhận diện khuôn mặt, thực hiện cho môn CS231.

## Các hướng tiếp cận

| Thư mục | Pipeline | Mô tả |
|---------|----------|-------|
| `HOG_SVM/` | HOG + SVM | Đặc trưng thủ công truyền thống kết hợp bộ phân loại SVM |
| `MTCNN_LBP-SOBELLBP_SVM/` | MTCNN + LBP/Sobel-LBP + SVM | Đặc trưng kết cấu (texture) với SVM, có hiệu chỉnh ngưỡng |
| `MTCNN_FACENET/` | MTCNN + FaceNet | Phát hiện bằng deep learning + nhận diện dựa trên embedding |
| `RETINAFACE_ARCFACE/` | RetinaFace + ArcFace | Phát hiện hiện đại + ArcFace margin loss |
| `SRCFD_ARCFACE_FAISS/` | SCRFD + ArcFace + FAISS | Truy hồi nhanh quy mô lớn bằng chỉ mục FAISS |

## Cấu trúc dự án

```
Final_CS231/
├── HOG_SVM/
│   ├── hog_svm.ipynb           # Notebook huấn luyện & đánh giá
│   └── model.py                # Định nghĩa model
├── MTCNN_LBP-SOBELLBP_SVM/
│   ├── feature_utils.py        # Trích đặc trưng LBP, Sobel-LBP
│   ├── train_svm.py            # Script huấn luyện
│   ├── calibrate_thresholds.py # Hiệu chỉnh ngưỡng
│   └── face_recognition_model/
│       ├── svm_face.pkl        # SVM đã huấn luyện
│       ├── scaler.pkl          # Bộ chuẩn hóa đặc trưng
│       ├── label_encoder.pkl   # Bộ mã hóa nhãn
│       ├── thresholds.pkl      # Ngưỡng đã hiệu chỉnh
│       ├── confusion_matrix.png
│       ├── threshold_histograms.png
│       └── threshold_2d_scatter.png
├── MTCNN_FACENET/
│   └── mtcnnxfacenet.ipynb
├── RETINAFACE_ARCFACE/
│   └── RetinaFace_+_ArcFace.ipynb
├── SRCFD_ARCFACE_FAISS/
│   └── SRCFD_ARCFACE_FAISS_CLEAN.ipynb
├── app.py                      # Ứng dụng demo Gradio
├── requirements.txt
└── README.md
```

## Cài đặt

### 1. Cài thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

### 2. Model đã huấn luyện (Git LFS)

Toàn bộ model đã train (`.pkl`, `.pt`, file dlib `.dat`...) được quản lý bằng **Git LFS**, nên khi clone repo các file này **tự được tải về** — không cần tải tay. Chỉ cần cài Git LFS một lần *trước khi* clone:

```bash
git lfs install
git clone https://github.com/diodanghoccs/Face-Recognition.git
```

Nếu bạn đã clone trước khi cài LFS, chạy `git lfs pull` để kéo các file model về.

> **Lưu ý:** Riêng các file embedding khuôn mặt (`face_db*`) **không** có trên repo vì chứa dữ liệu sinh trắc học từ ảnh cá nhân — cần tự sinh lại từ dataset của bạn (xem bước 3).

### 3. Chuẩn bị dataset

Dataset không được chia sẻ công khai do chứa ảnh cá nhân. Để tự tạo lại:

```
data/
└── Dataset Split Filtered/
    ├── train/
    │   └── <tên_người>/   ← ảnh huấn luyện
    ├── val/
    │   └── <tên_người>/
    └── test/
        └── <tên_người>/
```

Mỗi người cần **~80 ảnh train**, **~10 val**, **~10 test**.

### 4. Chạy ứng dụng demo

```bash
python app.py
```

## Dataset

- **5 người**, khoảng **100 ảnh/người**
- Ảnh được thu thập và lọc thủ công
- Dataset **không công khai** do chứa thông tin cá nhân

## Kết quả

Đánh giá trên 50 ảnh test (5 người). Bảng so sánh tổng hợp năm pipeline:

| Pipeline | Detect /50 | Acc trên mặt detect | Macro-F1 | Acc /50 (end-to-end) |
|----------|:----------:|:-------------------:|:--------:|:--------------------:|
| HOG + SVM | 44 (88%) | 0.97 | 0.97 | **0.86** |
| MTCNN + LBP/Sobel + SVM | 48 (96%) | 0.94 | 0.94 | **0.90** |
| MTCNN + FaceNet | 50 (100%) | 1.00 | 1.00 | **1.00** |
| RetinaFace + ArcFace | 50 (100%) | 1.00 | 1.00 | **1.00** |
| SCRFD + ArcFace + FAISS | 50 (100%) | 0.94 | 0.94 | **0.94** |

> **Acc /50 (end-to-end):** tính trên toàn bộ 50 ảnh test; khuôn mặt bị detector bỏ sót được tính là dự đoán sai. Khác với cột **"Acc trên mặt detect"** chỉ tính trên số mặt mỗi pipeline phát hiện được.


CS231 — Trường Đại học Công nghệ Thông tin (UIT)
