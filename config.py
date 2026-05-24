"""
config.py — Cấu hình tập trung cho toàn bộ project CS231.

Khi clone repo về, chỉ cần chỉnh file này, KHÔNG cần sửa các script khác.

Cách dùng:
    from config import cfg
    print(cfg.DATA_TRAIN)
"""

from pathlib import Path

# ===========================================================================
# ROOT — tự động xác định thư mục gốc của project (không cần sửa)
# ===========================================================================
ROOT = Path(__file__).resolve().parent


# ===========================================================================
# ★ DATASET PATHS — THAY ĐỔI NẾU THƯ MỤC DATASET CỦA BẠN KHÁC
# ===========================================================================

# Thư mục chứa dataset đã split (train / val / test)
# Cấu trúc mong đợi:
#   DATASET_ROOT/
#     train/<tên_người>/*.jpg
#     val/<tên_người>/*.jpg
#     test/<tên_người>/*.jpg
DATASET_ROOT = ROOT / "data" / "Dataset Split Filtered"

# Nếu dataset của bạn nằm ở chỗ khác, uncomment dòng dưới và chỉnh lại:
# DATASET_ROOT = Path(r"C:\Users\<tên>\Pictures\Dataset Face Reg")

DATA_TRAIN = DATASET_ROOT / "train"
DATA_VAL   = DATASET_ROOT / "val"
DATA_TEST  = DATASET_ROOT / "test"


# ===========================================================================
# ★ MTCNN_LBP-SOBELLBP_SVM — train_svm.py & calibrate_thresholds.py
# ===========================================================================

# Thư mục ảnh đã crop + align — đầu ra của mtcnn_finetuned_crop.py,
# đầu vào của train_svm.py.  PHẢI KHÁC LBP_DATASET_ROOT, nếu không
# crop sẽ ghi đè cấu trúc dataset → vỡ label encoding.
LBP_CROP_ROOT = ROOT / "data" / "Dataset_crop_finetuned_masked"
# Nếu bạn có thư mục crop riêng ngoài repo, uncomment:
# LBP_CROP_ROOT = Path(r"C:\Users\<tên>\Pictures\Dataset Face Reg_crop_finetuned_masked")

# Thư mục lưu model sau khi train (mặc định lưu trong repo)
LBP_MODEL_DIR = ROOT / "MTCNN_LBP-SOBELLBP_SVM" / "face_recognition_model"
# Để lưu ra ngoài repo, uncomment:
# LBP_MODEL_DIR = Path(r"C:\Users\<tên>\Downloads\face_recognition_model")

# (Tùy chọn) Thư mục ảnh "unknown" dùng để visualize khi calibrate threshold
# Nếu không có, script sẽ bỏ qua bước này
LBP_UNKNOWN_DIR = ROOT / "data" / "unknown"
# LBP_UNKNOWN_DIR = Path(r"C:\Users\<tên>\Downloads\merged_cropped\train")

# (Tùy chọn) Thư mục chứa weights MTCNN finetuned (.pt files)
# Chỉ cần nếu bạn dùng MTCNN finetuned riêng
LBP_WEIGHTS_DIR = ROOT / "MTCNN_LBP-SOBELLBP_SVM" / "weights"
# LBP_WEIGHTS_DIR = Path(r"C:\Users\<tên>\Pictures\mtcnn_finetune\weights")

# Thư mục ảnh GỐC (chưa crop) — dùng cho mtcnn_finetuned_crop.py.
# Mặc định trỏ về DATASET_ROOT ở trên (cùng cấu trúc train/val/test).
LBP_DATASET_ROOT = DATASET_ROOT
# LBP_DATASET_ROOT = Path(r"C:\Users\<tên>\Pictures\Dataset Split Filtered")


# ---------------------------------------------------------------------------
# ★ LBP — Tham số align/crop của MTCNN (mtcnn_finetuned_crop.py)
# ---------------------------------------------------------------------------
LBP_OUTPUT_SIZE         = (128, 128)   # kích thước crop output (phải khớp LBP_IMG_SIZE)
LBP_CONF_THR            = 0.60         # MTCNN confidence tối thiểu
LBP_MIN_FACE_PX         = 30           # cạnh bbox tối thiểu sau detect
LBP_DET_MIN_SIZE        = 15           # min_face_size truyền vào MTCNN
LBP_MTCNN_THRESHOLDS    = [0.5, 0.6, 0.70]   # cascade thresholds P/R/O-Net
LBP_EYE_DIST_RATIO      = 0.42         # khoảng cách mắt / chiều rộng output
LBP_EYE_Y_RATIO         = 0.40         # vị trí Y mắt / chiều cao output
LBP_MIN_VALID           = 0.55         # min phần ảnh hợp lệ sau warp
LBP_MASK_BG_VALUE       = 128          # mid-gray cho oval mask
LBP_MASK_FEATHER        = 10
LBP_MASK_AXES_RX        = 0.42
LBP_MASK_AXES_RY        = 0.55
LBP_MASK_CY_RATIO       = 0.55
LBP_USE_OVAL_MASK       = False        # bật để xoá background
LBP_SKIP_EXISTING       = False        # bỏ qua ảnh đã crop trước
LBP_SAVE_DEBUG_SAMPLES  = 30           # số ảnh random copy sang _debug_samples/

# Pose sanity check (open-set: reject mặt nghiêng quá ngưỡng).
# Áp cho cả crop dataset (training) lẫn inference.
LBP_POSE_CHECK          = True
LBP_NOSE_OFFSET_MAX     = 0.35         # |nose.x - eye_mid_x| / eye_dist
LBP_EYE_Y_ASYM_MAX      = 0.15         # |left_eye.y - right_eye.y| / face_height
LBP_INFER_CONF_THR      = 0.85         # MTCNN confidence ở inference (siết hơn 0.60 ở train)


# ---------------------------------------------------------------------------
# ★ LBP — Tham số feature extractor (feature_utils.py)
# ---------------------------------------------------------------------------
LBP_IMG_SIZE     = (128, 128)
LBP_CELL         = 16
LBP_STEP         = 8                   # 50% overlap với CELL=16 → 15×15=225 cell
LBP_BIN_R1       = 10                  # P=8  uniform → 0..9
LBP_BIN_R2       = 18                  # P=16 uniform → 0..17
LBP_CLAHE_CLIP   = 2.0
LBP_CLAHE_TILE   = (8, 8)


# ---------------------------------------------------------------------------
# ★ LBP — Tham số training (train_svm.py)
# ---------------------------------------------------------------------------
LBP_RANDOM_STATE          = 42
LBP_PCA_DIM               = 368        # chiều giữ sau PCA whiten
LBP_HARD_WEIGHT           = 2.0        # boost class_weight cho HARD_CLASSES
LBP_HARD_NEGATIVE_MINING  = False
LBP_NJOBS_OFFSET          = 6          # n_jobs = max(1, cpu_count() - OFFSET)
LBP_AUG_NOISE_RANGE       = 18         # ±18 cho np.random.randint
LBP_AUG_BRIGHT_HIGH_ALPHA = 1.3
LBP_AUG_BRIGHT_HIGH_BETA  = 20
LBP_AUG_BRIGHT_LOW_ALPHA  = 0.7
LBP_AUG_BRIGHT_LOW_BETA   = -20

# (Tùy chọn) GridSearchCV param grid — None để dùng default trong code.
LBP_GRID_PARAMS = None


# ---------------------------------------------------------------------------
# ★ LBP — Tham số calibrate / open-set threshold (calibrate_thresholds.py)
# ---------------------------------------------------------------------------
LBP_T_P_PERCENTILE   = 3       # percentile của p_max trên known
LBP_T_P_CAP          = 0.92    # cap trên T_p (tránh quá nghiêm khi val đẹp)
LBP_T_P_FLOOR        = 0.63    # floor T_p (chặn unknown confident)
LBP_T_M_PERCENTILE   = 5       # percentile của margin
LBP_T_M_CAP          = 0.80
LBP_RULE             = "AND"
# Per-class T_p boost: legacy manual tuning.
# Sau khi dùng class_weight='balanced' (train_svm) + FAR-driven sweep (calibrate),
# manual boost không còn cần thiết. Để rỗng = không boost (đồng đều mọi class).
LBP_PER_CLASS_BOOST  = {}
LBP_PER_CLASS_BOOST_MAX = 0.97   # cap cuối cùng cho T_p_per_class (vẫn dùng cho safety)

# Split val thành 2 phần: calib (fit Platt) và thresh (pick T_p/T_m/boost).
# Đặt 0.0 để dùng full val cho Platt và full val cho threshold (legacy, có overfit val).
LBP_VAL_THRESH_FRAC  = 0.5     # tỉ lệ val đưa sang nhánh "thresh"

# Số ảnh scan từ LBP_UNKNOWN_DIR khi calibrate (dùng cho FAR sweep + visualize)
LBP_N_SCAN    = 500
LBP_N_TARGET  = 300

# FAR-driven threshold sweep (3 tín hiệu open-set: p_max, margin, Mahalanobis)
# Sweep grid (T_p × T_m × τ_d), pick TAR max với FAR ≤ FAR_BUDGET.
LBP_FAR_BUDGET            = 0.01    # 1% Unknown được phép lọt
LBP_FAR_BUDGET_FALLBACK   = 0.05    # nới khi grid không có điểm thỏa
LBP_SWEEP_TP_RANGE        = (0.50, 0.97, 48)   # (lo, hi, steps)
LBP_SWEEP_TM_RANGE        = (0.05, 0.80, 30)
LBP_SWEEP_TAU_PCT_RANGE   = (90.0, 99.9, 20)   # percentile của Mahalanobis trên known
LBP_USE_FAR_SWEEP         = True    # False = fallback percentile-only (legacy)


# ===========================================================================
# HOG_SVM — app.py & hog_svm.ipynb
# ===========================================================================

HOG_DIR        = ROOT / "HOG_SVM"
HOG_PKL        = HOG_DIR / "models_svm.pkl"
HOG_SHAPE_PRED = HOG_DIR / "shape_predictor_68_face_landmarks.dat"

# URL tải shape_predictor_68_face_landmarks.dat (ưu tiên theo thứ tự)
HOG_SHAPE_URLS = [
    "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2",
    "https://raw.githubusercontent.com/davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2",
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
]


# ===========================================================================
# MTCNN_FACENET
# ===========================================================================

FACENET_DIR     = ROOT / "MTCNN_FACENET"
FACENET_DB_PATH = FACENET_DIR / "face_db.pt"


# ===========================================================================
# RETINAFACE_ARCFACE
# ===========================================================================

RETINA_DIR     = ROOT / "RETINAFACE_ARCFACE"
RETINA_DB_PATH = RETINA_DIR / "face_database.npy"


# ===========================================================================
# SCRFD_ARCFACE_FAISS
# ===========================================================================

SCRFD_DIR      = ROOT / "SRCFD_ARCFACE_FAISS"
SCRFD_DB_L_PATH = SCRFD_DIR / "face_db_0.pkl"   # buffalo_l
SCRFD_DB_S_PATH = SCRFD_DIR / "face_db_1.pkl"   # buffalo_s


# ===========================================================================
# HYPERPARAMETERS — có thể điều chỉnh nếu muốn thử nghiệm
# ===========================================================================

# LBP-SVM: các class cần hard-negative mining (để trống = tắt)
LBP_HARD_CLASSES: set = set()

# Ngưỡng nhận diện mặc định (dùng trong app.py khi chưa có thresholds.pkl)
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Gradio app: port và chế độ share
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE       = False   # Đặt True nếu muốn tạo public URL


# ===========================================================================
# Wrapper object để import gọn — từ config import cfg
# ===========================================================================
class _Config:
    ROOT               = ROOT
    DATASET_ROOT       = DATASET_ROOT
    DATA_TRAIN         = DATA_TRAIN
    DATA_VAL           = DATA_VAL
    DATA_TEST          = DATA_TEST

    LBP_CROP_ROOT      = LBP_CROP_ROOT
    LBP_MODEL_DIR      = LBP_MODEL_DIR
    LBP_UNKNOWN_DIR    = LBP_UNKNOWN_DIR
    LBP_WEIGHTS_DIR    = LBP_WEIGHTS_DIR
    LBP_DATASET_ROOT   = LBP_DATASET_ROOT

    # Align / crop
    LBP_OUTPUT_SIZE        = LBP_OUTPUT_SIZE
    LBP_CONF_THR           = LBP_CONF_THR
    LBP_MIN_FACE_PX        = LBP_MIN_FACE_PX
    LBP_DET_MIN_SIZE       = LBP_DET_MIN_SIZE
    LBP_MTCNN_THRESHOLDS   = LBP_MTCNN_THRESHOLDS
    LBP_EYE_DIST_RATIO     = LBP_EYE_DIST_RATIO
    LBP_EYE_Y_RATIO        = LBP_EYE_Y_RATIO
    LBP_MIN_VALID          = LBP_MIN_VALID
    LBP_MASK_BG_VALUE      = LBP_MASK_BG_VALUE
    LBP_MASK_FEATHER       = LBP_MASK_FEATHER
    LBP_MASK_AXES_RX       = LBP_MASK_AXES_RX
    LBP_MASK_AXES_RY       = LBP_MASK_AXES_RY
    LBP_MASK_CY_RATIO      = LBP_MASK_CY_RATIO
    LBP_USE_OVAL_MASK      = LBP_USE_OVAL_MASK
    LBP_SKIP_EXISTING      = LBP_SKIP_EXISTING
    LBP_SAVE_DEBUG_SAMPLES = LBP_SAVE_DEBUG_SAMPLES
    LBP_POSE_CHECK         = LBP_POSE_CHECK
    LBP_NOSE_OFFSET_MAX    = LBP_NOSE_OFFSET_MAX
    LBP_EYE_Y_ASYM_MAX     = LBP_EYE_Y_ASYM_MAX
    LBP_INFER_CONF_THR     = LBP_INFER_CONF_THR

    # Feature
    LBP_IMG_SIZE   = LBP_IMG_SIZE
    LBP_CELL       = LBP_CELL
    LBP_STEP       = LBP_STEP
    LBP_BIN_R1     = LBP_BIN_R1
    LBP_BIN_R2     = LBP_BIN_R2
    LBP_CLAHE_CLIP = LBP_CLAHE_CLIP
    LBP_CLAHE_TILE = LBP_CLAHE_TILE

    # Training
    LBP_RANDOM_STATE          = LBP_RANDOM_STATE
    LBP_PCA_DIM               = LBP_PCA_DIM
    LBP_HARD_WEIGHT           = LBP_HARD_WEIGHT
    LBP_HARD_NEGATIVE_MINING  = LBP_HARD_NEGATIVE_MINING
    LBP_NJOBS_OFFSET          = LBP_NJOBS_OFFSET
    LBP_AUG_NOISE_RANGE       = LBP_AUG_NOISE_RANGE
    LBP_AUG_BRIGHT_HIGH_ALPHA = LBP_AUG_BRIGHT_HIGH_ALPHA
    LBP_AUG_BRIGHT_HIGH_BETA  = LBP_AUG_BRIGHT_HIGH_BETA
    LBP_AUG_BRIGHT_LOW_ALPHA  = LBP_AUG_BRIGHT_LOW_ALPHA
    LBP_AUG_BRIGHT_LOW_BETA   = LBP_AUG_BRIGHT_LOW_BETA
    LBP_GRID_PARAMS           = LBP_GRID_PARAMS

    # Calibrate / threshold
    LBP_T_P_PERCENTILE      = LBP_T_P_PERCENTILE
    LBP_T_P_CAP             = LBP_T_P_CAP
    LBP_T_P_FLOOR           = LBP_T_P_FLOOR
    LBP_T_M_PERCENTILE      = LBP_T_M_PERCENTILE
    LBP_T_M_CAP             = LBP_T_M_CAP
    LBP_RULE                = LBP_RULE
    LBP_PER_CLASS_BOOST     = LBP_PER_CLASS_BOOST
    LBP_PER_CLASS_BOOST_MAX = LBP_PER_CLASS_BOOST_MAX
    LBP_VAL_THRESH_FRAC     = LBP_VAL_THRESH_FRAC
    LBP_N_SCAN              = LBP_N_SCAN
    LBP_N_TARGET            = LBP_N_TARGET

    LBP_FAR_BUDGET            = LBP_FAR_BUDGET
    LBP_FAR_BUDGET_FALLBACK   = LBP_FAR_BUDGET_FALLBACK
    LBP_SWEEP_TP_RANGE        = LBP_SWEEP_TP_RANGE
    LBP_SWEEP_TM_RANGE        = LBP_SWEEP_TM_RANGE
    LBP_SWEEP_TAU_PCT_RANGE   = LBP_SWEEP_TAU_PCT_RANGE
    LBP_USE_FAR_SWEEP         = LBP_USE_FAR_SWEEP

    HOG_DIR            = HOG_DIR
    HOG_PKL            = HOG_PKL
    HOG_SHAPE_PRED     = HOG_SHAPE_PRED
    HOG_SHAPE_URLS     = HOG_SHAPE_URLS

    FACENET_DIR        = FACENET_DIR
    FACENET_DB_PATH    = FACENET_DB_PATH

    RETINA_DIR         = RETINA_DIR
    RETINA_DB_PATH     = RETINA_DB_PATH

    SCRFD_DIR          = SCRFD_DIR
    SCRFD_DB_L_PATH    = SCRFD_DB_L_PATH
    SCRFD_DB_S_PATH    = SCRFD_DB_S_PATH

    LBP_HARD_CLASSES              = LBP_HARD_CLASSES
    DEFAULT_CONFIDENCE_THRESHOLD  = DEFAULT_CONFIDENCE_THRESHOLD
    GRADIO_SERVER_PORT            = GRADIO_SERVER_PORT
    GRADIO_SHARE                  = GRADIO_SHARE


cfg = _Config()
