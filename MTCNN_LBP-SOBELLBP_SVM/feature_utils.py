# ╔══════════════════════════════════════════════════════╗
# ║  Shared feature extractor                            ║
# ║  Multi-scale LBP^u2 + Sobel-LBP (Zhao et al. 2008)   ║
# ║  Grid 8x8 với overlap 50% (cell 16, step 8)          ║
# ║  L2-normalize per cell                                ║
# ║  Dim: 10800-d (LBP1=2250 + LBP2=4050 + SLBPx=2250 + SLBPy=2250)
# ╚══════════════════════════════════════════════════════╝
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

IMG_SIZE   = (128, 128)
CELL       = 16
STEP       = 8
LBP_BIN_R1 = 10   # P=8  uniform → bins 0..9
LBP_BIN_R2 = 18   # P=16 uniform → bins 0..17
EPS        = 1e-6

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


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
