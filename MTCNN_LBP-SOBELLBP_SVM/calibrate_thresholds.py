# ╔══════════════════════════════════════════════════════╗
# ║  BƯỚC 3: Calibrate open-set thresholds T_p, T_m     ║
# ║                                                      ║
# ║  Pick ngưỡng trên VAL_THRESH (nửa val mà Platt CHƯA  ║
# ║  thấy). Báo cáo TAR trên val_thresh và TEST.        ║
# ║                                                      ║
# ║  Chạy: python calibrate_thresholds.py                ║
# ║  Tất cả tham số đọc từ ../config.py (cfg.LBP_*).     ║
# ╚══════════════════════════════════════════════════════╝
import os, json, cv2, numpy as np, random, joblib, torch
from pathlib import Path
from facenet_pytorch import MTCNN as FacenetMTCNN
from tqdm import tqdm
import matplotlib.pyplot as plt

from feature_utils import extract_features

# ── Đọc cấu hình từ config.py ở thư mục gốc ──────────────────────────────
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from config import cfg

MODEL_DIR     = str(cfg.LBP_MODEL_DIR)
UNKNOWN_DIR   = str(cfg.LBP_UNKNOWN_DIR)
WEIGHTS_DIR   = str(cfg.LBP_WEIGHTS_DIR)
N_SCAN        = int(cfg.LBP_N_SCAN)
N_TARGET      = int(cfg.LBP_N_TARGET)
CONF_THR      = float(cfg.LBP_CONF_THR)
MIN_FACE_PX   = int(cfg.LBP_MIN_FACE_PX)
DET_MIN_SIZE  = int(cfg.LBP_DET_MIN_SIZE)
MTCNN_THRESH  = list(cfg.LBP_MTCNN_THRESHOLDS)
OUT_SIZE      = tuple(cfg.LBP_OUTPUT_SIZE)
RANDOM_STATE  = int(cfg.LBP_RANDOM_STATE)
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ── Align params (đồng bộ với crop pipeline) ─────────────────
EYE_DIST_RATIO = float(cfg.LBP_EYE_DIST_RATIO)
EYE_Y_RATIO    = float(cfg.LBP_EYE_Y_RATIO)
MIN_VALID      = float(cfg.LBP_MIN_VALID)

# ── Threshold knobs ──────────────────────────────────────────
T_P_PERCENTILE  = float(cfg.LBP_T_P_PERCENTILE)
T_P_CAP         = float(cfg.LBP_T_P_CAP)
T_P_FLOOR       = float(cfg.LBP_T_P_FLOOR)
T_M_PERCENTILE  = float(cfg.LBP_T_M_PERCENTILE)
T_M_CAP         = float(cfg.LBP_T_M_CAP)
RULE            = str(cfg.LBP_RULE)
PER_CLASS_BOOST = dict(cfg.LBP_PER_CLASS_BOOST)
PER_CLASS_CAP   = float(cfg.LBP_PER_CLASS_BOOST_MAX)

# ── FAR-driven sweep + Mahalanobis ───────────────────────────
USE_FAR_SWEEP        = bool(cfg.LBP_USE_FAR_SWEEP)
FAR_BUDGET           = float(cfg.LBP_FAR_BUDGET)
FAR_BUDGET_FALLBACK  = float(cfg.LBP_FAR_BUDGET_FALLBACK)
_TP_LO, _TP_HI, _TP_STEPS    = cfg.LBP_SWEEP_TP_RANGE
_TM_LO, _TM_HI, _TM_STEPS    = cfg.LBP_SWEEP_TM_RANGE
_TAU_LO, _TAU_HI, _TAU_STEPS = cfg.LBP_SWEEP_TAU_PCT_RANGE
T_P_GRID    = np.linspace(_TP_LO,  _TP_HI,  int(_TP_STEPS))
T_M_GRID    = np.linspace(_TM_LO,  _TM_HI,  int(_TM_STEPS))
TAU_PCT_GRID = np.linspace(_TAU_LO, _TAU_HI, int(_TAU_STEPS))

# ── Reproducibility ──────────────────────────────────────────
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


def _check_manifest(model_dir, classes):
    """Stale-cache guard: kiểm tra manifest có khớp cache + label_encoder không."""
    mpath = Path(model_dir) / "manifest.json"
    if not mpath.exists():
        print("  ⚠️  Không tìm thấy manifest.json — bỏ qua check (cache có thể stale).")
        return None
    with open(mpath, "r", encoding="utf-8") as f:
        m = json.load(f)
    if list(m.get("classes", [])) != list(classes):
        raise RuntimeError(
            f"Manifest classes {m.get('classes')} ≠ label_encoder classes {list(classes)}. "
            f"Hãy chạy lại train_svm.py."
        )
    print(f"  ✓ Manifest OK (hash={m.get('hash')}, sklearn={m.get('sklearn_version')}, "
          f"pca_dim={m.get('pca_dim')}, n_val_thresh={m.get('n_val_thresh')}, "
          f"n_val_calib={m.get('n_val_calib')})")
    return m


def _slice_two_signals(svm, X):
    """Trả về (p_max, margin) cho mỗi mẫu trong X."""
    proba = svm.predict_proba(X)
    sp    = np.sort(proba, axis=1)
    return sp[:, -1], sp[:, -1] - sp[:, -2]


def _mahalanobis_per_sample(X, preds, class_stats):
    """Mahalanobis distance từ mỗi sample tới class mà SVM predict.
    Open-set signal: predict Dan nhưng x xa Dan centroid → reject."""
    out = np.zeros(len(X), dtype=np.float64)
    for c, st in class_stats.items():
        mask = preds == c
        if mask.any():
            diff = X[mask] - st["mu"]
            out[mask] = np.sqrt(np.einsum("ij,jk,ik->i", diff, st["cov_inv"], diff))
    return out


def _far_sweep(known_p, known_m, known_d, unk_p, unk_m, unk_d, far_budget):
    """Sweep grid (T_p, T_m, τ_d), trả best record + all records.
    Pick: max TAR với FAR ≤ far_budget. Fallback nới budget nếu rỗng."""
    tau_d_grid = np.percentile(known_d, TAU_PCT_GRID)
    records = []
    for tp in T_P_GRID:
        for tm in T_M_GRID:
            for td in tau_d_grid:
                tar = ((known_p >= tp) & (known_m >= tm) & (known_d <= td)).mean()
                if unk_p is not None:
                    far = ((unk_p >= tp) & (unk_m >= tm) & (unk_d <= td)).mean()
                else:
                    far = float("nan")
                records.append((tp, tm, td, tar, far))
    records = np.array(records)
    if unk_p is None:
        return None, records

    feasible = records[records[:, 4] <= far_budget]
    if len(feasible) == 0:
        print(f"  ⚠️  Không có điểm nào FAR ≤ {far_budget*100:.1f}%, "
              f"nới lên {FAR_BUDGET_FALLBACK*100:.1f}%")
        feasible = records[records[:, 4] <= FAR_BUDGET_FALLBACK]
    if len(feasible) == 0:
        print("  ⚠️  Vẫn rỗng → pick min-FAR")
        feasible = records[records[:, 4] == records[:, 4].min()]
    # Sort: TAR desc, FAR asc, tau_d desc
    feasible = feasible[np.lexsort((-feasible[:, 2], feasible[:, 4], -feasible[:, 3]))]
    return tuple(feasible[0]), records


def _plot_roc(records, best, far_budget, out_path):
    if records is None or len(records) == 0:
        return
    fars, tars = records[:, 4], records[:, 3]
    valid = ~np.isnan(fars)
    if not valid.any():
        return
    plt.figure(figsize=(7, 5))
    plt.scatter(fars[valid], tars[valid], s=3, alpha=0.18, color="steelblue",
                label="All grid points")
    plt.axvline(far_budget, ls="--", color="gray", label=f"FAR ≤ {far_budget*100:.1f}%")
    if best is not None:
        plt.scatter([best[4]], [best[3]], s=120, marker="*", color="red", zorder=5,
                    label=f"Op point (T_p={best[0]:.2f}, T_m={best[1]:.2f}, τ_d={best[2]:.2f})")
    plt.xlabel("FAR (Unknown lọt qua)")
    plt.ylabel("TAR (Known nhận đúng)")
    plt.title("Open-set sweep — TAR vs FAR")
    plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"  ROC sweep saved → {out_path}")


def main():
    # ── Load model + cache ────────────────────────────────────────
    print("Loading model + cache...")
    svm    = joblib.load(os.path.join(MODEL_DIR, "svm_face.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    pca    = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
    le     = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    # class_stats optional (legacy model có thể chưa có)
    cs_path = os.path.join(MODEL_DIR, "class_stats.pkl")
    if os.path.exists(cs_path):
        class_stats = joblib.load(cs_path)
        print(f"  Loaded class_stats.pkl ({len(class_stats)} classes) — Mahalanobis enabled")
    else:
        class_stats = None
        print(f"  ⚠️  class_stats.pkl không có — chạy lại train_svm.py để bật Mahalanobis")

    _check_manifest(MODEL_DIR, le.classes_)

    _cache = np.load(os.path.join(MODEL_DIR, "train_cache.npz"))
    cache_keys = set(_cache.files)

    # Ưu tiên dùng val_thresh (nửa val mà Platt CHƯA thấy).
    # Fallback về full val nếu train_svm cũ (legacy) không split.
    if "X_val_thr_p" in cache_keys and "y_val_thr" in cache_keys:
        X_val_thr_p = _cache["X_val_thr_p"]
        y_val_thr   = _cache["y_val_thr"]
        X_val_cal_p = _cache.get("X_val_cal_p")
        y_val_cal   = _cache.get("y_val_cal")
        split_mode  = "split"
        print(f"  Loaded: val_thresh={X_val_thr_p.shape}  "
              f"val_calib={None if X_val_cal_p is None else X_val_cal_p.shape}")
    else:
        X_val_thr_p = _cache["X_val_p"]
        y_val_thr   = _cache["y_val"]
        X_val_cal_p = X_val_thr_p
        y_val_cal   = y_val_thr
        split_mode  = "legacy_full_val"
        print(f"  ⚠️  Cache thiếu X_val_thr_p — fallback full val (legacy mode).")
        print(f"  Loaded: X_val_p={X_val_thr_p.shape}  y_val={y_val_thr.shape}")

    X_test_p = _cache["X_test_p"]
    y_test   = _cache["y_test"]

    # ── PER-CLASS BOOST — chỉnh tại config.py ────────────────────
    print(f"\nPER_CLASS_BOOST (đọc từ cfg.LBP_PER_CLASS_BOOST):")
    for k, v in PER_CLASS_BOOST.items():
        print(f"   {k:10s}: +{v:.4f}")

    # ── Tính 2 tín hiệu trên val_thresh (slice Platt CHƯA thấy) ──
    known_p, known_m = _slice_two_signals(svm, X_val_thr_p)

    print(f"\nVal_thresh ({len(known_p)} mẫu):")
    print(f"  p_max  — min={known_p.min():.3f}  mean={known_p.mean():.3f}  max={known_p.max():.3f}")
    print(f"  margin — min={known_m.min():.3f}  mean={known_m.mean():.3f}  max={known_m.max():.3f}")

    # ── Chọn ngưỡng GLOBAL từ val_thresh ────────────────────────
    T_p  = max(float(min(np.percentile(known_p, T_P_PERCENTILE), T_P_CAP)), T_P_FLOOR)
    T_m  = float(min(np.percentile(known_m, T_M_PERCENTILE), T_M_CAP))
    rule = RULE

    # ── Tính T_p per-class (global + boost) ─────────────────────
    T_p_per_class = {}
    for cls in le.classes_:
        boost = PER_CLASS_BOOST.get(cls, 0.0)
        T_p_per_class[cls] = round(min(T_p + boost, PER_CLASS_CAP), 4)

    print(f"\nGlobal threshold  (rule={rule}, pct_p={T_P_PERCENTILE}, "
          f"cap={T_P_CAP}, floor={T_P_FLOOR}):")
    print(f"   T_p >= {T_p:.4f}  |  T_m >= {T_m:.4f}")
    print(f"\nPer-class T_p:")
    for cls in le.classes_:
        boost = PER_CLASS_BOOST.get(cls, 0.0)
        mark  = f"  (+{boost:.3f} boost)" if boost > 0 else ""
        print(f"   {cls:10s}: {T_p_per_class[cls]:.4f}{mark}")

    # ── TAR trên val_thresh (slice "pick ngưỡng") ───────────────
    def _apply_rule(p, m, raw_pred_cls):
        tp_arr = np.array([T_p_per_class[c] for c in raw_pred_cls])
        return (p >= tp_arr) & (m >= T_m)

    raw_pred_thr = le.classes_[svm.predict_proba(X_val_thr_p).argmax(axis=1)]
    mask_thr     = _apply_rule(known_p, known_m, raw_pred_thr)

    print(f"\n🎯 TAR trên val_thresh (rule=AND): {mask_thr.mean()*100:.1f}%  "
          f"(slice dùng để pick ngưỡng — báo cáo này lạc quan)")

    # ── TAR HONEST trên TEST set (chưa hề được nhìn để pick) ────
    test_p, test_m = _slice_two_signals(svm, X_test_p)
    raw_pred_test  = le.classes_[svm.predict_proba(X_test_p).argmax(axis=1)]
    mask_test      = _apply_rule(test_p, test_m, raw_pred_test)
    correct_test   = (le.transform(raw_pred_test) == y_test)
    accept_correct = mask_test & correct_test

    print(f"\n📊 TAR HONEST trên TEST (rule=AND): {mask_test.mean()*100:.1f}%  "
          f"(trong đó accept-đúng-class: {accept_correct.mean()*100:.1f}%)")
    print("   Per-class TAR trên TEST:")
    for cls_idx in np.unique(y_test):
        m_ = (y_test == cls_idx)
        if m_.sum() == 0:
            continue
        print(f"      {le.classes_[cls_idx]:10s}: {mask_test[m_].mean()*100:5.1f}%  "
              f"({m_.sum()} mẫu)")

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
            min_face_size=DET_MIN_SIZE,
            thresholds=MTCNN_THRESH,
            keep_all=True,
            post_process=False,
            device=DEVICE,
        )
        for net_name, net_obj in [("pnet", _det_unk.pnet),
                                   ("rnet", _det_unk.rnet),
                                   ("onet", _det_unk.onet)]:
            wp = Path(WEIGHTS_DIR) / f"{net_name}.pt"
            if wp.exists():
                try:
                    net_obj.load_state_dict(torch.load(str(wp), map_location=DEVICE),
                                            strict=False)
                except Exception:
                    pass

        files = [f for f in Path(UNKNOWN_DIR).iterdir()
                 if f.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp")]
        random.shuffle(files)
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
            unk_p, unk_m = _slice_two_signals(svm, X_unk_p)
            print(f"  Sạch: {len(X_unk_p)} / quét {N_SCAN}  |  skip: {skip}")
    else:
        print(f"\n Không tìm thấy {UNKNOWN_DIR} — bỏ qua unknown display")

    # ── Bảng TAR vs T_p (trên TEST) ──────────────────────────────
    def _mask_g(pm, mg, tp):
        return (pm >= tp) & (mg >= T_m)

    print(f"\n TAR (TEST) vs global T_p   (T_m={T_m:.3f}  rule=AND):")
    print(f"   {'T_p':>5}  {'TAR':>6}  {'FAR':>8}")
    for tp in sorted(set([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, round(T_p, 3)])):
        tar = _mask_g(test_p, test_m, tp).mean()
        marker = "  <- global auto" if abs(tp - T_p) < 0.001 else ""
        if unk_p is not None:
            far = _mask_g(unk_p, unk_m, tp).mean()
            print(f"   {tp:.2f}   {tar*100:5.1f}%   {far*100:6.2f}%{marker}")
        else:
            print(f"   {tp:.2f}   {tar*100:5.1f}%      N/A{marker}")

    # ── Histogram (vẽ trên val_thresh để không tự lừa mình) ──────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (k_vals, u_vals, name, t_val) in zip(axes, [
        (known_p, unk_p, "p_max",  T_p),
        (known_m, unk_m, "margin", T_m),
    ]):
        ax.hist(k_vals, bins=20, alpha=0.7, color="steelblue", label="Known (val_thresh)")
        if u_vals is not None:
            ax.hist(u_vals, bins=20, alpha=0.5, color="tomato", label="Unknown")
        ax.axvline(t_val, color="black", ls="--", lw=2, label=f"T={t_val:.3f}")
        ax.set_title(name); ax.legend(fontsize=8)
    plt.suptitle("Signal distributions — val_thresh vs Unknown (optional)")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "threshold_histograms.png"), dpi=150)
    print(f"\n  Histogram saved to {MODEL_DIR}/threshold_histograms.png")
    plt.show()

    # ── FAR-driven sweep (nếu có class_stats + USE_FAR_SWEEP) ────
    #
    # Logic: dùng 3 tín hiệu open-set AND-ed (p_max, margin, Mahalanobis),
    # sweep grid 3D, pick TAR max với FAR ≤ FAR_BUDGET trên val_thresh + unknown set.
    # Override T_p, T_m, tau_d trong thresholds.pkl.
    tau_d = None
    sweep_meta = None
    if class_stats is not None and USE_FAR_SWEEP:
        # Mahalanobis trên Known (val_thresh) — predict bằng SVM Frozen + Platt-cal
        pred_thr_idx = svm.predict_proba(X_val_thr_p).argmax(axis=1)
        known_d = _mahalanobis_per_sample(X_val_thr_p, pred_thr_idx, class_stats)
        print(f"\nMahalanobis (val_thresh): min={known_d.min():.2f}  "
              f"mean={known_d.mean():.2f}  max={known_d.max():.2f}")

        # Mahalanobis trên Unknown
        if X_unk_p is not None:
            pred_unk_idx = svm.predict_proba(X_unk_p).argmax(axis=1)
            unk_d = _mahalanobis_per_sample(X_unk_p, pred_unk_idx, class_stats)
            print(f"Mahalanobis (unknown):    min={unk_d.min():.2f}  "
                  f"mean={unk_d.mean():.2f}  max={unk_d.max():.2f}")
        else:
            unk_d = None

        print(f"\n🔍 FAR-driven sweep (grid {len(T_P_GRID)}×{len(T_M_GRID)}×{len(TAU_PCT_GRID)} "
              f"= {len(T_P_GRID)*len(T_M_GRID)*len(TAU_PCT_GRID)} points)...")
        best, records = _far_sweep(known_p, known_m, known_d,
                                   unk_p,   unk_m,   unk_d,
                                   far_budget=FAR_BUDGET)
        if best is not None:
            T_p_new, T_m_new, tau_d, tar_best, far_best = best
            print(f"\n🎯 FAR-sweep operating point:")
            print(f"   T_p   = {T_p_new:.4f}   (cũ percentile-only: {T_p:.4f})")
            print(f"   T_m   = {T_m_new:.4f}   (cũ percentile-only: {T_m:.4f})")
            print(f"   τ_d   = {tau_d:.4f}   (Mahalanobis)")
            print(f"   → TAR (val_thresh) = {tar_best*100:.1f}%   "
                  f"FAR = {far_best*100:.2f}%")
            # Override
            T_p = float(T_p_new)
            T_m = float(T_m_new)
            # Per-class T_p cập nhật cho khớp (giữ boost cũ làm thứ cấp)
            T_p_per_class = {cls: round(min(T_p + PER_CLASS_BOOST.get(cls, 0.0),
                                            PER_CLASS_CAP), 4)
                             for cls in le.classes_}
            sweep_meta = {
                "tar_at_op": float(tar_best),
                "far_at_op": float(far_best) if not np.isnan(far_best) else None,
                "far_budget": FAR_BUDGET,
                "grid_size": int(len(records)),
            }
            _plot_roc(records, best, FAR_BUDGET,
                      os.path.join(MODEL_DIR, "roc_sweep.png"))
        else:
            print("  Không có Unknown set → bỏ qua FAR sweep, dùng percentile-only.")

    # ── Save ──────────────────────────────────────────────────────
    thresholds = {
        "T_p"            : T_p,
        "T_m"            : T_m,
        "tau_d"          : tau_d,      # None nếu legacy / không có class_stats
        "rule"           : rule,
        "T_p_per_class"  : T_p_per_class,
        "split_mode"     : split_mode,
        "per_class_boost": PER_CLASS_BOOST,
        "sweep_meta"     : sweep_meta,  # None nếu không chạy sweep
    }
    joblib.dump(thresholds, os.path.join(MODEL_DIR, "thresholds.pkl"))
    print(f"\nLưu thresholds (T_p, T_m, τ_d, per-class) -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
