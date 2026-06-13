# CS231 Final Project — Face Recognition System

A comparative study of multiple face recognition pipelines, built for CS231.

## Approaches

| Folder | Pipeline | Description |
|--------|----------|-------------|
| `HOG_SVM/` | HOG + SVM | Traditional handcrafted features with SVM classifier |
| `MTCNN_FACENET/` | MTCNN + FaceNet | Deep learning detection + embedding-based recognition |
| `MTCNN_LBP-SOBELLBP_SVM/` | MTCNN + LBP/Sobel-LBP + SVM | Texture descriptors with SVM, includes calibrated thresholds |
| `RETINAFACE_ARCFACE/` | RetinaFace + ArcFace | State-of-the-art detection + ArcFace margin loss |
| `SRCFD_ARCFACE_FAISS/` | SCRFD + ArcFace + FAISS | Fast large-scale retrieval with FAISS index |

## Project Structure

```
Final_CS231/
├── HOG_SVM/
│   ├── hog_svm.ipynb           # Training & evaluation notebook
│   └── model.py                # Model definition
├── MTCNN_FACENET/
│   └── mtcnnxfacenet.ipynb
├── MTCNN_LBP-SOBELLBP_SVM/
│   ├── feature_utils.py        # LBP, Sobel-LBP feature extraction
│   ├── train_svm.py            # Training script
│   ├── calibrate_thresholds.py # Threshold calibration
│   └── face_recognition_model/
│       ├── svm_face.pkl        # Trained SVM
│       ├── scaler.pkl          # Feature scaler
│       ├── label_encoder.pkl   # Label encoder
│       ├── thresholds.pkl      # Calibrated thresholds
│       ├── confusion_matrix.png
│       ├── threshold_histograms.png
│       └── threshold_2d_scatter.png
├── RETINAFACE_ARCFACE/
│   └── RetinaFace_+_ArcFace.ipynb
├── SRCFD_ARCFACE_FAISS/
│   └── SRCFD_ARCFACE_FAISS_CLEAN.ipynb
├── app.py                      # Gradio demo app
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download pre-trained models

Some large model files are not included in this repo. Download manually:

**dlib shape predictor** (required for HOG_SVM):
```bash
# Download shape_predictor_68_face_landmarks.dat
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat HOG_SVM/
```

**Trained SVM models** (HOG_SVM, MTCNN_LBP-SOBELLBP_SVM):
> Các file `.pkl` lớn được lưu riêng (liên hệ nhóm để lấy link Google Drive).

### 3. Prepare your dataset

Dataset không được chia sẻ công khai do chứa ảnh cá nhân. Để tự tạo lại:

```
data/
└── Dataset Split Filtered/
    ├── train/
    │   └── <person_name>/   ← ảnh training
    ├── val/
    │   └── <person_name>/
    └── test/
        └── <person_name>/
```

Mỗi người cần **~80 ảnh train**, **~10 val**, **~10 test**.

### 4. Run the demo app

```bash
python app.py
```

## Dataset

- **5 người**, khoảng **100 ảnh/người**
- Ảnh được thu thập và lọc thủ công
- Dataset **không được public** do chứa thông tin cá nhân

## Results

Đánh giá trên tập test (5 người).

| Pipeline | Accuracy | Macro-F1 |
|----------|----------|----------|
| HOG + SVM | 0.97 | 0.97 |
| MTCNN + FaceNet | 1.00 | 1.00 |
| MTCNN + LBP-SobelLBP + SVM | 0.94 | 0.94 |
| RetinaFace + ArcFace | 1.00 | 1.00 |
| SCRFD + ArcFace + FAISS | 0.9714 | 0.97 |

## Team

CS231 — University of Information Technology (UIT)
