# ╔══════════════════════════════════════════════════════╗
# ║  BƯỚC 1: Load dataset + Feature extraction + Train  ║
# ║  Multi-scale LBP + Sobel-LBP (Zhao 2008)             ║
# ║  → PCA whiten → SVM (probability=False)              ║
# ║  → CalibratedClassifierCV(FrozenEstimator) trên val   ║
# ║  Chạy: python train_svm.py                           ║
# ╚══════════════════════════════════════════════════════╝
import os, cv2, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from feature_utils import extract_features

# ╔══════════════════════════════════════════════════════╗
# ║                     CONFIG                           ║
# ║  Chỉnh đường dẫn trong config.py ở thư mục gốc      ║
# ╚══════════════════════════════════════════════════════╝
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from config import cfg

CROP_ROOT    = str(cfg.LBP_CROP_ROOT)
MODEL_DIR    = str(cfg.LBP_MODEL_DIR)
RANDOM_STATE = 42
PCA_DIM      = 368  # Số chiều giữ sau PCA (whiten=True)

# Hard pairs hay nhầm lẫn — boost class_weight cao hơn 'balanced'
HARD_CLASSES = cfg.LBP_HARD_CLASSES  # Tắt boost cho dataset balanced (controlled condition)
HARD_WEIGHT  = 2.0

# Hard-negative mining: chạy thêm 1 vòng train với weight tăng cho sample bị nhầm
HARD_NEGATIVE_MINING = False

os.makedirs(MODEL_DIR, exist_ok=True)


# ╔══════════════════════════════════════════════════════╗
# ║  Augmentation — 5 variants / ảnh                    ║
# ║  (controlled condition: bỏ rotation, perspective,   ║
# ║   blur — chỉ giữ flip + brightness + noise)         ║
# ╚══════════════════════════════════════════════════════╝
def augment(img):
    noise = np.clip(img.astype(np.int16) + np.random.randint(-18, 18, img.shape, dtype=np.int16), 0, 255).astype(np.uint8)

    return [
        img,
        cv2.flip(img, 1),
        cv2.convertScaleAbs(img, alpha=1.3,  beta=20),
        cv2.convertScaleAbs(img, alpha=0.7,  beta=-20),
        noise,
    ]

# ╔══════════════════════════════════════════════════════╗
# ║  Load 3 tập đã split + MTCNN-crop sẵn               ║
# ╚══════════════════════════════════════════════════════╝
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def _load_split(split_name):
    imgs, lbls = [], []
    root = Path(CROP_ROOT) / split_name
    if not root.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {root}. Chạy cell MTCNN crop trước."
        )
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        loaded = 0
        for fp in sorted(d.iterdir()):
            if fp.suffix.lower() not in IMG_EXTS:
                continue
            img = cv2.imread(str(fp))
            if img is None:
                continue
            imgs.append(img); lbls.append(d.name); loaded += 1
        print(f"  [{split_name:5s}] {d.name:10s}: {loaded} ảnh")
    return imgs, np.array(lbls)


def _build_class_weight(le):
    """Boost weight cho HARD_CLASSES, các class còn lại = 1.0 (sklearn sẽ kết hợp với 'balanced' qua sample_weight nếu cần)."""
    w = {}
    for i, cls in enumerate(le.classes_):
        w[i] = HARD_WEIGHT if cls in HARD_CLASSES else 1.0
    return w


def _train_svm(X_train, y_train, sample_weight, le, label="initial"):
    N_JOBS = max(1, os.cpu_count() - 6)
    print(f"\n[{label}] GridSearchCV... (dùng {N_JOBS}/{os.cpu_count()} cores)")
    param_grid = [
        {
            "kernel": ["rbf"],
            "C":      [0.1, 1, 10, 50, 100, 500],
            "gamma":  ["scale", "auto", 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        },
        {
            "kernel": ["poly"],
            "degree": [2, 3],
            "C":      [1, 10, 100],
            "gamma":  ["scale", 1e-3, 1e-2],
            "coef0":  [0, 1],
        },
    ]
    cw = _build_class_weight(le) if HARD_CLASSES else None
    grid = GridSearchCV(
        SVC(probability=False, class_weight=cw, random_state=RANDOM_STATE),
        param_grid,
        cv=5,
        n_jobs=N_JOBS,
        verbose=1,
    )
    fit_kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
    grid.fit(X_train, y_train, **fit_kw)
    print(f"[{label}] Best params : {grid.best_params_}")
    print(f"[{label}] CV accuracy : {grid.best_score_*100:.1f}%")
    return grid.best_estimator_


def main():
    print("Đang load dataset đã split + crop...")
    images_train, labels_train = _load_split("train")
    print()
    images_val,   labels_val   = _load_split("val")
    print()
    images_test,  labels_test  = _load_split("test")

    n_aug = len(augment(images_train[0]))
    feat0 = extract_features(images_train[0])
    print(f"\n{'═'*52}")
    print(f"  Train : {len(images_train):>5} ảnh  →  sau aug ×{n_aug} = {len(images_train)*n_aug}")
    print(f"  Val   : {len(images_val):>5} ảnh")
    print(f"  Test  : {len(images_test):>5} ảnh")
    print(f"  Classes: {sorted(set(labels_train.tolist()))}")
    print(f"  Feature dim: {len(feat0)}")
    print(f"{'═'*52}")

    # ── Encode labels ─────────────────────────────────────────────
    le          = LabelEncoder()
    y_train_raw = le.fit_transform(labels_train)
    y_val       = le.transform(labels_val)
    y_test      = le.transform(labels_test)

    # ── Augment chỉ tập train ─────────────────────────────────────
    X_train, y_train = [], []
    for img, lbl in zip(images_train, y_train_raw):
        for aug_img in augment(img):
            X_train.append(extract_features(aug_img))
            y_train.append(lbl)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)

    # ── Val & Test (KHÔNG augment) ────────────────────────────────
    X_val  = np.array([extract_features(img) for img in images_val],  dtype=np.float32)
    X_test = np.array([extract_features(img) for img in images_test], dtype=np.float32)
    print(f"Sau aug — Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ── Scale ─────────────────────────────────────────────────────
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # ── PCA analysis nhanh — tính full để biết elbow ─────────────
    pca_full = PCA(whiten=False, random_state=RANDOM_STATE)
    pca_full.fit(X_train_s)
    cumvar_full = np.cumsum(pca_full.explained_variance_ratio_)
    print("  Variance retention (full PCA):")
    for target in [0.70, 0.80, 0.90, 0.95]:
        n_needed = int(np.searchsorted(cumvar_full, target)) + 1
        marker = "  ← hiện tại" if n_needed == PCA_DIM else ""
        print(f"    {target*100:.0f}%  →  {n_needed:>4} components{marker}")
    idx = min(PCA_DIM - 1, len(cumvar_full) - 1)
    print(f"    [{PCA_DIM} components đang dùng → {cumvar_full[idx]*100:.1f}%]")

    # ── PCA (giữ PCA_DIM chiều, whiten) ───────────────────────────
    n_comp_max = min(PCA_DIM, X_train_s.shape[0], X_train_s.shape[1])
    pca       = PCA(n_components=n_comp_max, whiten=True, random_state=RANDOM_STATE)
    X_train_p = pca.fit_transform(X_train_s)
    X_val_p   = pca.transform(X_val_s)
    X_test_p  = pca.transform(X_test_s)
    print(f"PCA: {X_train_s.shape[1]} → {X_train_p.shape[1]} dims  (whiten=True, "
          f"explained variance={pca.explained_variance_ratio_.sum()*100:.1f}%)")

    # ── SVM lần 1 (probability=False — calibrate sau trên val) ───
    svm = _train_svm(X_train_p, y_train, sample_weight=None, le=le, label="initial")

    # ── Hard-negative mining (mặc định tắt cho dataset balanced) ──
    if HARD_NEGATIVE_MINING:
        y_pred_train = svm.predict(X_train_p)
        wrong        = (y_pred_train != y_train)
        n_wrong      = int(wrong.sum())
        print(f"\nHard-negative mining: {n_wrong}/{len(y_train)} sample sai trên train.")

        if n_wrong > 0:
            sample_w = np.ones(len(y_train), dtype=np.float64)
            sample_w[wrong] = 3.0   # tăng weight 3× cho sample sai
            # boost thêm cho sample thuộc HARD_CLASSES bị nhầm
            hard_idx = np.array([le.classes_[y] in HARD_CLASSES for y in y_train])
            sample_w[wrong & hard_idx] = 5.0
            svm = _train_svm(X_train_p, y_train, sample_weight=sample_w, le=le, label="hard-neg")

    # ── Calibrate xác suất TRÊN VAL SET (không augment, không leakage) ──
    print(f"\nCalibrating Platt sigmoid trên val set ({len(X_val_p)} mẫu)...")
    cal_svm = CalibratedClassifierCV(FrozenEstimator(svm), method="sigmoid")
    cal_svm.fit(X_val_p, y_val)
    print(f"  Calibrated.")

    # ── Evaluate ──────────────────────────────────────────────────
    train_acc = cal_svm.score(X_train_p, y_train)
    val_acc   = cal_svm.score(X_val_p,   y_val)
    test_acc  = cal_svm.score(X_test_p,  y_test)
    print(f"\n{'═'*52}")
    print(f"  Train accuracy : {train_acc*100:.1f}%")
    print(f"  Val   accuracy : {val_acc*100:.1f}%")
    print(f"  Test  accuracy : {test_acc*100:.1f}%")
    print(f"{'═'*52}")

    y_pred = cal_svm.predict(X_test_p)
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix (Test Set)")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    print(f"  Confusion matrix saved to {MODEL_DIR}/confusion_matrix.png")
    plt.close()

    # ── Lưu model ─────────────────────────────────────────────────
    joblib.dump(cal_svm, os.path.join(MODEL_DIR, "svm_face.pkl"))
    joblib.dump(scaler,  os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(pca,     os.path.join(MODEL_DIR, "pca.pkl"))
    joblib.dump(le,      os.path.join(MODEL_DIR, "label_encoder.pkl"))

    # Xóa lda.pkl cũ nếu có (tránh test/calibrate script load nhầm)
    old_lda = os.path.join(MODEL_DIR, "lda.pkl")
    if os.path.exists(old_lda):
        os.remove(old_lda)
        print(f"Đã xóa {old_lda} (không còn dùng LDA)")

    # ── Lưu cache cho calibrate_thresholds ───────────────────────
    np.savez(os.path.join(MODEL_DIR, "train_cache.npz"),
             X_train_p=X_train_p,
             X_val_p=X_val_p,   y_val=