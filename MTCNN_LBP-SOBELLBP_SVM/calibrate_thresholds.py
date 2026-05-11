# ╔══════════════════════════════════════════════════════╗
# ║  Calibrate thresholds — trên VAL SET                ║
# ║  = Cell "calibrate_cell" chuyển ra .py               ║
# ║  Chạy: python calibrate_thresholds.py                ║
# ╚══════════════════════════════════════════════════════╝
import os, cv2, numpy as np, random, joblib, torch
from pathlib import Path
from facenet_pytorch import MTCNN as FacenetMTCNN
from tqdm import tqdm
import matplotlib.pyplot as plt

from feature_utils import extract_features

# ── Đọc cấu hình từ config.py ở thư mục gốc ──────────────────────────────
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from config import cfg

MODEL_DIR    = str(cfg.LBP_MODEL_DIR)
UNKNOWN_DIR  = str(cfg.LBP_UNKNOWN_DIR)
WEIGHTS_DIR  = str(cfg.LBP_WEIGHTS_DIR)
N_SCAN       = 500
N_TARGET     = 300
CONF_THR     = 0.60
MIN_FACE_PX  = 30
OUT_SIZE     = (128, 128)
RANDOM_STATE = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Align params (khớp với cell crop) ────────────────────────
EYE_DIST_RATIO = 0.42
EYE_Y_RATIO    = 0.40
MIN_VALID      = 0.55


def main():
    # ── Load model + cache ────────────────────────────────────────
    print("Loading model + cache...")
    svm    = joblib.load(os.path.join(MODEL_DIR, "svm_face.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    pca    = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
    le     = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    _cache = np.load(os.path.join(MODEL_DIR, "train_cache.npz"))
    X_val_p   = _cache["X_val_p"]
    y_val     = _cache["y_val"]
    print(f"  Loaded: X_val_p={X_val_p.shape}  y_val={y_val.shape}")

    # ╔══════════════════════════════════════════════════════╗
    # ║  PER-CLASS BOOST — chỉnh tại đây                    ║
    # ╚══════════════════════════════════════════════════════╝
    PER_CLASS_BOOST = {
        "Dan"  : 0.11,
        "Vi"   : 0.11,
        #"Tuan" : 0.10,
        "Triet": 0.012,
        "Tri": 0.026
    }

    # ── Tính 2 tín hiệu trên VAL set (known) ────────────────────
    proba_val = svm.predict_proba(X_val_p)
    sp_val    = np.sort(proba_val, axis=1)
    known_p   = sp_val[:, -1]
    known_m   = sp_val[:, -1] - sp_val[:, -2]

    print(f"Known VAL set ({len(known_p)} mẫu):")
    print(f"  p_max  — min={known_p.min():.3f}  mean={known_p.mean():.3f}  max={known_p.max():.3f}")
    print(f"  margin — min={known_m.min():.3f}  mean={known_m.mean():.3f}  max={known_m.max():.3f}")

    # ── Chọn ngưỡng GLOBAL từ val set ───────────────────────────
    # Cap T_p cao hơn (0.92) vì sau fix Sobel-LBP + bỏ LDA + cv='prefit',
    # p_max known dự kiến > 0.85 → percentile 3 vẫn an toàn
    # Ép ngưỡng T_p tối thiểu là 0.63 để chặn bớt Unknown
    T_p  = max(float(min(np.percentile(known_p, 3),  0.92)), 0.63)
    T_m  = float(min(np.percentile(known_m, 5),  0.80))
    rule = "AND"

    # ── Tính T_p per-class (global + boost) ─────────────────────
    T_p_per_class = {}
    for cls in le.classes_:
        boost = PER_CLASS_BOOST.get(cls, 0.0)
        T_p_per_class[cls] = round(min(T_p + boost, 0.97), 4)

    print(f"\nGlobal threshold  (rule={rule}):")
    print(f"   T_p >= {T_p:.4f}  |  T_m >= {T_m:.4f}")
    print(f"\nPer-class T_p:")
    for cls in le.classes_:
        boost = PER_CLASS_BOOST.get(cls, 0.0)
        mark  = f"  (+{boost:.2f} boost  ← unknown attractor)" if boost > 0 else ""
        print(f"   {cls:10s}: {T_p_per_class[cls]:.4f}{mark}")

    # ── Per-class TAR với rule AND (trên val) ───────────────────
    raw_pred_val = le.classes_[svm.predict_proba(X_val_p).argmax(axis=1)]
    tp_arr_val   = np.array([T_p_per_class[c] for c in raw_pred_val])
    mask_val     = (known_p >= tp_arr_val) & (known_m >= T_m)

    print(f"\n🎯 Overall TAR trên VAL (rule=AND, per-class T_p): {mask_val.mean()*100:.1f}%")
    print("📋 Per-class recognition rate (val):")
    for cls_idx in np.unique(y_val):
        m = (y_val == cls_idx)
        if m.sum() == 0:
            continue
        print(f"   {le.classes_[cls_idx]:10s}: {mask_val[m].mean()*100:5.1f}%  ({m.sum()} mẫu)")

    # ── Nạp unknown dataset (optional — chỉ để visualize) ───────
    def _align_unk(img_rgb, box, lmk):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w < MIN_FACE_PX or h < MIN_FACE_PX: return None
        le_, re_ = np.float32(lmk[0]), np.float32(lmk[1])
        dX, dY = re_[0]-le_[0], re_[1]-le_[1]
        ec = ((le_[0]+re_[0])/2, (le_[1]+re_[1])/2)
        sc = (EYE_DIST_RATIO*OUT_SIZE[0]) / (np.sqrt(dX**2+dY**2)+1e-6)
        M = cv2.getRotationMatrix2D(ec, np.degrees(np.arctan2(dY, dX)), sc)
        M[0,2] += OUT_SIZE[0]*0.50 - ec[0]
        M[1,2] += OUT_SIZE[1]*EYE_Y_RATIO - ec[1]
        H_s, W_s = img_rgb.shape[:2]
        mask = cv2.warpAffine(np.ones((H_s,W_s),np.uint8), M, OUT_SIZE,
                              flags=cv2.INTER_NEAREST, borderValue=0)
        if mask.mean() < MIN_VALID: return None
        aligned = cv2.warpAffine(img_rgb, M, OUT_SIZE,
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)

    X_unk_p = None
    unk_p = unk_m = None

    if Path(UNKNOWN_DIR).exists():
        print(f"\nScan unknown (optional, chỉ để visualize)...")
        _det_unk = FacenetMTCNN(
            min_face_size=20,
            thresholds=[0.5, 0.6, 0.70],
            keep_all=True,
            post_process=False,
            device=DEVICE,
        )
        # Load fine-tuned weights
        for net_name, net_obj in [("pnet", _det_unk.pnet), ("rnet", _det_unk.rnet), ("onet", _det_unk.onet)]:
            wp = Path(WEIGHTS_DIR) / f"{net_name}.pt"
            if wp.exists():
                try:
                    net_obj.load_state_dict(torch.load(str(wp), map_location=DEVICE), strict=False)
                except Exception:
                    pass

        files = [f for f in Path(UNKNOWN_DIR).iterdir()
                 if f.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp")]
        random.seed(RANDOM_STATE); random.shuffle(files)
        X_unk_raw = []
        skip = {"no_face":0,"low_conf":0,"bad_crop":0,"bad_read":0}
        for fp in tqdm(files[:N_SCAN]):
            if len(X_unk_raw) >= N_TARGET: break
            img = cv2.imread(str(fp))
            if img is None: skip["bad_read"]+=1; continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes_u, probs_u, lmks_u = _det_unk.detect(rgb, landmarks=True)
            if boxes_u is None: skip["no_face"]+=1; continue
            best_i = int(np.argmax(probs_u))
            if probs_u[best_i] < CONF_THR: skip["low_conf"]+=1; continue
            crop = _align_unk(rgb, boxes_u[best_i], lmks_u[best_i])
            if crop is None: skip["bad_crop"]+=1; continue
            X_unk_raw.append(extract_features(crop))
        if X_unk_raw:
            X_unk_p = pca.transform(scaler.transform(np.array(X_unk_raw, dtype=np.float32)))
            sp_unk  = np.sort(svm.predict_proba(X_unk_p), axis=1)
            unk_p   = sp_unk[:,-1]
            unk_m   = sp_unk[:,-1] - sp_unk[:,-2]
            print(f"  Sạch: {len(X_unk_p)} / quét {N_SCAN}  |  skip: {skip}")
    else:
        print(f"\n Không tìm thấy {UNKNOWN_DIR} — bỏ qua unknown display")

    # ── Bảng TAR vs T_p ──────────────────────────────────────────
    def _mask_g(pm, mg, tp):
        return (pm >= tp) & (mg >= T_m)

    print(f"\n TAR vs global T_p (T_m={T_m:.3f}  rule=AND):")
    print(f"   {'T_p':>5}  {'TAR':>6}  {'FAR':>8}")
    for tp in sorted(set([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, round(T_p, 3)])):
        tar = _mask_g(known_p, known_m, tp).mean()
        marker = "  <- global auto" if abs(tp - T_p) < 0.001 else ""
        if unk_p is not None:
            far = _mask_g(unk_p, unk_m, tp).mean()
            print(f"   {tp:.2f}   {tar*100:5.1f}%   {far*100:6.2f}%{marker}")
        else:
            print(f"   {tp:.2f}   {tar*100:5.1f}%      N/A{marker}")

    # ── Histogram ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (k_vals, u_vals, name, t_val) in zip(axes, [
        (known_p, unk_p, "p_max",  T_p),
        (known_m, unk_m, "margin", T_m),
    ]):
        ax.hist(k_vals, bins=20, alpha=0.7, color="steelblue", label="Known (val)")
        if u_vals is not None:
            ax.hist(u_vals, bins=20, alpha=0.5, color="tomato", label="Unknown")
        ax.axvline(t_val, color="black", ls="--", lw=2, label=f"T={t_val:.3f}")
        ax.set_title(name); ax.legend(fontsize=8)
    plt.suptitle("Signal distributions — Known (val) vs Unknown (optional)")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "threshold_histograms.png"), dpi=150)
    print(f"\n  Histogram saved to {MODEL_DIR}/threshold_histograms.png")
    plt.show()

    # ── Save ──────────────────────────────────────────────────────
    thresholds = {
        "T_p"           : T_p,
        "T_m"           : T_m,
        "rule"          : rule,
        "T_p_per_class" : T_p_