# ╔══════════════════════════════════════════════════════╗
# ║  Shared feature extractor                            ║
# ║  Multi-scale LBP^u2 + Sobel-LBP (Zhao et al. 2008)   ║
# ║  Cell 16×16, step 8 (overlap 50%) trên ảnh 128×128  ║
# ║   → 15×15 = 225 cell, L2-normalize per cell          ║
# ║  Dim: 10800-d (LBP1=2250 + LBP2=4050 + SLBPx=2250    ║
# ║                 + SLBPy=2250)                        ║
# ║                                                       ║
# ║  Tất cả tham số đọc từ ../config.py (cfg.LBP_*).     ║
# ╚══════════════════════════════════════════════════════╝
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# ── Đọc cấu hình từ config.py ở thư mục gốc ──────────────────────────────
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from config import cfg

IMG_SIZE   = tuple(cfg.LBP_IMG_SIZE)
CELL       = int(cfg.LBP_CELL)
STEP       = int(cfg.LBP_STEP)
LBP_BIN_R1 = int(cfg.LBP_BIN_R1)
LBP_BIN_R2 = int(cfg.LBP_BIN_R2)
EPS        = 1e-6

_clahe = cv2.createCLAHE(
    clipLimit=float(cfg.LBP_CLAHE_CLIP),
    tileGridSize=tuple(cfg.LBP_CLAHE_TILE),
)


def _cell_grid(h, w):
    """Yield (y, x) top-left của các cell theo grid overlap."""
    for y in range(0, h - CELL + 1, STEP):
        for x in range(0, w - CELL + 1, STEP):
            yield y, x


def _spatial_hist(img2d, bins, value_range):
    """Histogram per cell + L2 normalize per cell."""
    feats = []
    h, w = img2d.shape
    for y, x in _cell_grid(h, w):
        cell = img2d[y:y+CELL, x:x+CELL]
        hist, _ = np.histogram(cell.ravel(), bins=bins, range=value_range, density=False)
        n = np.linalg.norm(hist) + EPS
        feats.extend(hist / n)
    return feats


def extract_features(img_bgr):
    resized = cv2.resize(img_bgr, IMG_SIZE)
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray    = _clahe.apply(gray)

    # ── Multi-scale Uniform LBP trên ảnh gốc ───────────────────
    lbp1 = local_binary_pattern(gray, P=8,  R=1, method="uniform")
    lbp2 = local_binary_pattern(gray, P=16, R=2, method="uniform")

    f_lbp1 = _spatial_hist(lbp1, bins=LBP_BIN_R1, value_range=(0, LBP_BIN_R1))
    f_lbp2 = _spatial_hist(lbp2, bins=LBP_BIN_R2, value_range=(0, LBP_BIN_R2))

    # ── Sobel-LBP (Zhao, Gao, Zhang 2008, eq. 4-5) ─────────────
    # I^x = Sobel_x * I,  I^y = Sobel_y * I  (signed int16)
    # Sobel-LBP^x = LBP(I^x),  Sobel-LBP^y = LBP(I^y)
    # CV_16S → int16 signed: tránh warning của skimage khi LBP trên float
    sx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)

    lbp_sx = local_binary_pattern(sx, P=8, R=1, method="uniform")
    lbp_sy = local_binary_pattern(sy, P=8, R=1, method="uniform")

    f_slbp_x = _spatial_hist(lbp_sx, bins=LBP_BIN_R1, value_range=(0, LBP_BIN_R1))
    f_slbp_y = _spatial_hist(lbp_sy, bins=LBP_BIN_R1, value_range=(0, LBP_BIN_R1))

    return np.concatenate([f_lbp1, f_lbp2, f_slbp_x, f_slbp_y]).astype(np.float32)
