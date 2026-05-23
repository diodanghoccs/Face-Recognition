# MTCNN + LBP/Sobel-LBP + SVM

Hệ thống nhận diện khuôn mặt **open-set** bằng classical ML:

```
Raw photos
   │  (MTCNN fine-tuned)
   ▼
Aligned crops 128×128
   │  (CLAHE → multi-scale LBP + Sobel-LBP → L2 per cell)
   ▼
Vector 10 800-d
   │  (StandardScaler → PCA whiten 368-d)
   ▼
SVM (RBF / poly, GridSearchCV) → Platt calibration (FrozenEstimator trên val_calib)
   │
   ▼
T_p, T_m + per-class boost (pick trên val_thresh)  →  từ chối Unknown
```

## Cài đặt

```bash
pip install -r requirements.txt
```

`scikit-learn >= 1.6` là **bắt buộc** vì `train_svm.py` dùng
`sklearn.frozen.FrozenEstimator` (chỉ có từ 1.6 trở lên).

## Cấu hình

**Mọi knob** của pipeline đều nằm trong `../config.py` dưới dạng `cfg.LBP_*`:

| Nhóm        | Một số key quan trọng |
|-------------|-----------------------|
| Paths       | `LBP_DATASET_ROOT`, `LBP_CROP_ROOT`, `LBP_MODEL_DIR`, `LBP_WEIGHTS_DIR`, `LBP_UNKNOWN_DIR` |
| Align       | `LBP_OUTPUT_SIZE`, `LBP_EYE_DIST_RATIO`, `LBP_EYE_Y_RATIO`, `LBP_MIN_VALID`, `LBP_USE_OVAL_MASK` |
| Detect      | `LBP_CONF_THR`, `LBP_DET_MIN_SIZE`, `LBP_MTCNN_THRESHOLDS` |
| Feature     | `LBP_IMG_SIZE`, `LBP_CELL`, `LBP_STEP`, `LBP_BIN_R1`, `LBP_BIN_R2`, `LBP_CLAHE_CLIP` |
| Training    | `LBP_RANDOM_STATE`, `LBP_PCA_DIM`, `LBP_HARD_CLASSES`, `LBP_HARD_WEIGHT`, `LBP_GRID_PARAMS` |
| Augment     | `LBP_AUG_NOISE_RANGE`, `LBP_AUG_BRIGHT_HIGH_*`, `LBP_AUG_BRIGHT_LOW_*` |
| Threshold   | `LBP_T_P_PERCENTILE`, `LBP_T_P_CAP`, `LBP_T_P_FLOOR`, `LBP_T_M_*`, `LBP_RULE`, `LBP_PER_CLASS_BOOST`, `LBP_VAL_THRESH_FRAC` |

Không có hardcode trong các script `.py` nữa — sửa `../config.py` là đủ.

## Thứ tự chạy

```bash
# B1 — Detect + align + crop 128×128
python mtcnn_finetuned_crop.py

# B2 — Trích đặc trưng, train SVM, calibrate Platt
python train_svm.py

# B3 — Pick T_p, T_m, per-class boost trên val_thresh, báo cáo TAR trên TEST
python calibrate_thresholds.py
```

Mỗi script đều `seed` ngẫu nhiên (`numpy`, `random`, `torch`) bằng
`cfg.LBP_RANDOM_STATE` ngay khi import.

## Anti-leakage: split val

Trước đây toàn bộ tập **val** được dùng cho cả Platt sigmoid lẫn pick `T_p`/`T_m`/`PER_CLASS_BOOST` — đây là cùng-một-tập làm 3 việc, dễ overfit val.

Phiên bản mới chia val theo `cfg.LBP_VAL_THRESH_FRAC` (mặc định 0.5):

| Slice          | Vai trò                                  | Ai nhìn |
|----------------|------------------------------------------|---------|
| `val_calib`    | Fit Platt sigmoid (`CalibratedClassifierCV`) | `train_svm.py` |
| `val_thresh`   | Pick `T_p`, `T_m`, áp `PER_CLASS_BOOST`  | `calibrate_thresholds.py` |
| `test`         | **Báo cáo HONEST** TAR/per-class TAR     | cả hai (chỉ score, không tune) |

Đặt `LBP_VAL_THRESH_FRAC = 0.0` nếu muốn quay về behaviour cũ (legacy, không khuyến nghị).

## Manifest & stale-cache guard

`train_svm.py` ghi `manifest.json` cạnh các pkl, gồm:
hash, sklearn version, random_state, feature_dim, pca_dim, n_classes, classes, kích thước từng split, augment_variants…

`calibrate_thresholds.py` đọc manifest trước khi pick ngưỡng và sẽ raise nếu danh sách lớp trong manifest không khớp với `label_encoder.pkl`. Nhờ thế bạn không thể vô tình chạy B3 trên cache cũ sau khi đổi `PCA_DIM` mà quên rerun B2.

## Artifact

Sau khi chạy đủ B2 + B3, thư mục `cfg.LBP_MODEL_DIR` sẽ có:

```
svm_face.pkl          CalibratedClassifierCV(FrozenEstimator(SVC))
scaler.pkl            StandardScaler (fit trên train)
pca.pkl               PCA whiten 368-d (fit trên train)
label_encoder.pkl     LabelEncoder
thresholds.pkl        {T_p, T_m, rule, T_p_per_class, split_mode, per_class_boost}
manifest.json         metadata + hash để chống stale cache
train_cache.npz       X_train_p, X_val_cal_p, X_val_thr_p, X_test_p (+ y_*)
confusion_matrix.png  trên TEST
threshold_histograms.png  p_max & margin trên val_thresh
```

## Reproducibility

* `cfg.LBP_RANDOM_STATE = 42` được truyền vào `SVC`, `PCA`, `train_test_split` (split val), `random.Random`, `np.random.default_rng` (augment noise), `torch.manual_seed`.
* Toàn bộ augment noise đi qua một RNG riêng (`np.random.default_rng(SEED)`) — không phụ thuộc global state.
* Manifest lưu sklearn version để truy vết khi load lại artifacts trên môi trường khác.

## Inference

Bạn đã có Gradio demo. Khi load lại pipeline, đảm bảo gọi đúng các hàm preprocess đã chia sẻ ở đây — đừng tự copy code crop/feature riêng:

```python
from feature_utils import extract_features                    # 10 800-d
from mtcnn_finetuned_crop import load_finetuned_mtcnn, align_and_crop
import joblib, numpy as np

mdir   = str(cfg.LBP_MODEL_DIR)
svm    = joblib.load(f"{mdir}/svm_face.pkl")
scaler = joblib.load(f"{mdir}/scaler.pkl")
pca    = joblib.load(f"{mdir}/pca.pkl")
le     = joblib.load(f"{mdir}/label_encoder.pkl")
thr    = joblib.load(f"{mdir}/thresholds.pkl")

# (sau khi detect + align bằng MTCNN, lấy crop_bgr 128×128)
f      = extract_features(crop_bgr).reshape(1, -1)
proba  = svm.predict_proba(pca.transform(scaler.transform(f)))[0]
pred   = int(proba.argmax())
p_max  = float(proba[pred])
margin = float(p_max - np.sort(proba)[-2])
name   = le.classes_[pred]
accept = (p_max >= thr["T_p_per_class"][name]) and (margin >= thr["T_m"])
label  = name if accept else "Unknown"
```
