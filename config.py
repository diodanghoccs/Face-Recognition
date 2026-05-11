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

# Thư mục ảnh đã crop + finetune dùng để train LBP-SVM
# (có thể giống DATA_TRAIN nếu bạn không crop riêng)
LBP_CROP_ROOT = DATA_TRAIN
# Nếu bạn có thư mục crop riêng, uncomment:
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
