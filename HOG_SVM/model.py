import os
import cv2
import dlib
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import pickle
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Đảm bảo terminal hiển thị được tiếng Việt
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Fallback cho các phiên bản Python cũ hơn (mặc dù user dùng 3.14)
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Khởi tạo bộ tìm khuôn mặt và bộ căn chỉnh
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except RuntimeError:
    print("❌ LỖI: Không tìm thấy file shape_predictor_68_face_landmarks.dat!")
    print("Vui lòng tải file http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 giải nén và để vào thư mục dự án.")
    sys.exit(1)

base_dataset_path = r"C:\Users\diosodumb\Documents\Final_CS231\data\Dataset Split Filtered"
train_path = os.path.join(base_dataset_path, "train")
test_path = os.path.join(base_dataset_path, "test")
val_path = os.path.join(base_dataset_path, "val")

def extract_features(dataset_path, augment=False):
    X_features = [] 
    Y_labels = []   
    
    print(f"Đang trích xuất đặc trưng khuôn mặt từ {dataset_path}...")
    if not os.path.exists(dataset_path):
        print(f"Không tìm thấy thư mục: {dataset_path}")
        return X_features, Y_labels

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir): continue
            
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            img = cv2.imread(image_path)
            if img is None: continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Chuyển tham số 1 thành 0 để không upsample ảnh, giúp quét cực kỳ nhanh
            faces = detector(gray, 0) # Detect mặt trong ảnh
            
            # Chỉ lấy ảnh nếu nó tìm thấy ĐÚNG 1 khuôn mặt để đảm bảo độ chính xác
            if len(faces) == 1:
                face = faces[0]
                
                # 1. Căn chỉnh khuôn mặt (Face Alignment) về 112x112 theo chuẩn tài liệu
                shape = predictor(img, face)
                face_chip = dlib.get_face_chip(img, shape, size=112)
                
                # 2. Chuyển về ảnh xám và resize về 64x64 để đưa vào HOG
                gray_chip = cv2.cvtColor(face_chip, cv2.COLOR_BGR2GRAY)
                face_img_resized = cv2.resize(gray_chip, (128, 128))
                
                # Tính toán vector HOG của khuôn mặt (vector dài 1764)
                hog_feature = hog(face_img_resized, orientations=9, pixels_per_cell=(8, 8), 
                                  cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
                
                X_features.append(hog_feature)
                Y_labels.append(person_name)
                
                # Áp dụng Data Augmentation (Lật ngang ảnh) để tăng gấp đôi dữ liệu train
                if augment:
                    flipped_img = cv2.flip(face_img_resized, 1)
                    hog_flipped = hog(flipped_img, orientations=9, pixels_per_cell=(8, 8), 
                                      cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
                    X_features.append(hog_flipped)
                    Y_labels.append(person_name)
    return X_features, Y_labels

# 1. Trích xuất đặc trưng và Train mô hình (có bật Augmentation)
X_train, Y_train = extract_features(train_path, augment=True)
print(f"Đã trích xuất được {len(X_train)} khuôn mặt từ tập train (đã bao gồm ảnh lật ngang).")

# Chuyển đổi tên người (String) thành Tên máy hiểu (Numbers)
le = LabelEncoder()
Y_train_encoded = le.fit_transform(Y_train)

# Đưa vào cho máy học (SVM) kèm Tinh chỉnh tham số (GridSearchCV)
from sklearn.model_selection import GridSearchCV
print("Đang chạy GridSearchCV để tìm siêu tham số tối ưu nhất...")
param_grid = {
    'C': [1, 10, 100],
    'gamma': [1e-3, 5e-4, 1e-4, 'scale'],
    'kernel': ['rbf', 'linear']
}
base_svm = SVC(class_weight='balanced', probability=True)
svm_model = GridSearchCV(base_svm, param_grid, cv=5, n_jobs=-1, verbose=1)
svm_model.fit(X_train, Y_train_encoded)

print(f"⭐ Tham số tốt nhất tìm được: {svm_model.best_params_}")

# 2. Đánh giá trên tập test và val
def predict_with_threshold(X, threshold=CONFIDENCE_THRESHOLD):
    """Dự đoán có ngưỡng: dưới ngưỡng → trả về 'Unknown'."""
    probs = svm_model.predict_proba(X)
    max_probs = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    labels = []
    for prob, pred in zip(max_probs, predictions):
        if prob >= threshold:
            labels.append(le.inverse_transform([pred])[0])
        else:
            labels.append("Unknown")
    return labels

def print_classification_report(Y_true, Y_pred, classes, dataset_name):
    print(f"\n📊 Bảng Classification Report ({dataset_name}) - ngưỡng={CONFIDENCE_THRESHOLD}):")
    all_classes = list(classes) + ["Unknown"]
    print(classification_report(Y_true, Y_pred, labels=all_classes, zero_division=0))

X_test, Y_test = extract_features(test_path)
Y_pred_test = predict_with_threshold(X_test)
print_classification_report(Y_test, Y_pred_test, le.classes_, "Test")

X_val, Y_val = extract_features(val_path)
Y_pred_val = predict_with_threshold(X_val)
print_classification_report(Y_val, Y_pred_val, le.classes_, "Validation")

# 3. Lưu lại mô hình để lần sau dùng (không phải train từ đầu)
with open("models_svm.pkl", "wb") as f:
    pickle.dump((le, svm_model, CONFIDENCE_THRESHOLD), f)

print(f"✅ Đã lưu mô hình vào models_svm.pkl (ngưỡng={CONFIDENCE_THRESHOLD})")
