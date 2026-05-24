"""Gradio demo so sanh 5 pipeline Face Recognition.

Chay:
    python app.py                # khoi dong UI
    python app.py --rebuild-db   # rebuild lai DB embedding cho 3 pipeline embedding

Quy tac:
- Moi pipeline tu quan ly artifact trong thu muc rieng cua minh (HOG_SVM/,
  MTCNN_LBP-SOBELLBP_SVM/, MTCNN_FACENET/, RETINAFACE_ARCFACE/, SRCFD_ARCFACE_FAISS/).
- Khong ghi gi vao thu muc data/.
"""

from __future__ import annotations

import argparse
import bz2
import os
import pickle
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_TRAIN = ROOT / "data" / "Dataset Split Filtered" / "train"

HOG_DIR       = ROOT / "HOG_SVM"
LBP_DIR       = ROOT / "MTCNN_LBP-SOBELLBP_SVM"
LBP_MODEL_DIR = LBP_DIR / "face_recognition_model"
FACENET_DIR   = ROOT / "MTCNN_FACENET"
RETINA_DIR    = ROOT / "RETINAFACE_ARCFACE"
SCRFD_DIR     = ROOT / "SRCFD_ARCFACE_FAISS"

HOG_PKL          = HOG_DIR / "models_svm.pkl"
HOG_SHAPE_PRED   = HOG_DIR / "shape_predictor_68_face_landmarks.dat"
# GitHub mirror (nhanh hon dlib.net rat nhieu o VN), fallback ve dlib.net neu fail.
HOG_SHAPE_URLS   = [
    "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2",
    "https://raw.githubusercontent.com/davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2",
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
]

FACENET_DB_PATH  = FACENET_DIR / "face_db.pt"
RETINA_DB_PATH   = RETINA_DIR / "face_database.npy"
SCRFD_DB_L_PATH  = SCRFD_DIR / "face_db_0.pkl"   # buffalo_l
SCRFD_DB_S_PATH  = SCRFD_DIR / "face_db_1.pkl"   # buffalo_s

# Sub-label hien thi duoi ten moi pipeline (UI design)
PIPELINE_SUBS = {
    "HOG + SVM":                  "Baseline · CPU",
    "MTCNN + LBP/Sobel-LBP + SVM":    "Hand-crafted features · CPU",
    "MTCNN + FaceNet":            "Inception-ResNet · embedding",
    "RetinaFace + ArcFace":       "ResNet50 · 512-d embedding",
    "SCRFD + ArcFace + FAISS":    "buffalo_s + buffalo_l · FAISS-IVF",
}

# Cho phep import feature_utils tu LBP_DIR
if str(LBP_DIR) not in sys.path:
    sys.path.insert(0, str(LBP_DIR))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_bgr(img: np.ndarray | None) -> np.ndarray | None:
    """Gradio Image (RGB ndarray) -> BGR cho OpenCV. None passthrough."""
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def to_rgb(img: np.ndarray | None) -> np.ndarray | None:
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def annotate(img_bgr: np.ndarray, bbox, label: str, color=(0, 255, 0)) -> np.ndarray:
    out = img_bgr.copy()
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2)
    else:
        cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a / na, b / nb))


def list_train_identities() -> list[str]:
    if not DATA_TRAIN.exists():
        return []
    return sorted(d.name for d in DATA_TRAIN.iterdir() if d.is_dir())


# ---------------------------------------------------------------------------
# Pipeline 1: HOG + SVM
# ---------------------------------------------------------------------------
class HogSvmPipeline:
    name = "HOG + SVM"

    def __init__(self):
        self.detector = None
        self.predictor = None
        self.svm = None
        self.le = None
        self.classes: list[str] = []
        self.error: str | None = None
        self.ready = False

    def _ensure_shape_predictor(self):
        if HOG_SHAPE_PRED.exists():
            return
        bz2_path = HOG_SHAPE_PRED.with_suffix(".dat.bz2")
        # Xoa partial download cu (neu user huy giua chung lan truoc)
        bz2_path.unlink(missing_ok=True)

        last_err: Exception | None = None
        for url in HOG_SHAPE_URLS:
            try:
                print(f"[HOG] Downloading from {url} ...")
                self._download_with_progress(url, bz2_path, timeout=20)
                break
            except Exception as e:
                last_err = e
                print(f"[HOG] Failed: {e}. Thu mirror tiep theo...")
                bz2_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Khong tai duoc shape_predictor tu bat ky mirror nao. "
                               f"Last error: {last_err}\n"
                               f"Tai thu cong: https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2\n"
                               f"giai nen va luu vao: {HOG_SHAPE_PRED}")

        print(f"[HOG] Decompressing -> {HOG_SHAPE_PRED}")
        with bz2.open(bz2_path, "rb") as fz, open(HOG_SHAPE_PRED, "wb") as fo:
            fo.write(fz.read())
        bz2_path.unlink(missing_ok=True)
        print(f"[HOG] Done.")

    @staticmethod
    def _download_with_progress(url: str, dest: Path, timeout: int = 20):
        import socket
        socket.setdefaulttimeout(timeout)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            chunk = 1024 * 64
            t0 = time.time()
            last_print = 0.0
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                now = time.time()
                if now - last_print > 0.5 or done == total:
                    elapsed = now - t0 + 1e-6
                    speed = done / elapsed / 1024
                    pct = (done / total * 100) if total else 0
                    print(f"\r  [{pct:5.1f}%] {done/1024:.0f} KB / "
                          f"{(total/1024) if total else 0:.0f} KB  ({speed:.0f} KB/s)",
                          end="", flush=True)
                    last_print = now
            print()

    def load(self):
        try:
            import dlib  # type: ignore
            import joblib  # noqa
            from sklearn.preprocessing import LabelEncoder
            self._ensure_shape_predictor()
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(str(HOG_SHAPE_PRED))
            with open(HOG_PKL, "rb") as f:
                raw = pickle.load(f)
            # File co the la tuple (svm, le) hoac dict {'svm':..., 'le':...} hoac estimator truc tiep
            self.svm = self._unwrap(raw, lambda o: hasattr(o, "predict_proba"))
            self.le  = self._unwrap(raw, lambda o: isinstance(o, LabelEncoder))
            if self.svm is None:
                raise RuntimeError(f"Khong tim thay estimator co predict_proba trong {HOG_PKL} "
                                   f"(file structure: {type(raw).__name__})")
            self.classes = list_train_identities()
            self.ready = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print(f"[HOG] load error: {self.error}")

    @staticmethod
    def _unwrap(obj, predicate):
        if predicate(obj):
            return obj
        if isinstance(obj, (tuple, list)):
            for item in obj:
                got = HogSvmPipeline._unwrap(item, predicate)
                if got is not None:
                    return got
        if isinstance(obj, dict):
            for v in obj.values():
                got = HogSvmPipeline._unwrap(v, predicate)
                if got is not None:
                    return got
        return None

    def _hog_feature(self, face_aligned_gray_128: np.ndarray) -> np.ndarray:
        from skimage.feature import hog
        return hog(face_aligned_gray_128, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm="L2-Hys", visualize=False)

    def _detect_align(self, img_bgr: np.ndarray):
        import dlib  # type: ignore
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        if len(faces) == 0:
            return None, None
        # Lay mat to nhat
        face = max(faces, key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()))
        shape = self.predictor(img_bgr, face)
        chip = dlib.get_face_chip(img_bgr, shape, size=112)
        gray_chip = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_chip, (128, 128))
        bbox = (face.left(), face.top(), face.right(), face.bottom())
        return resized, bbox

    def predict(self, ref_bgr, snap_bgr) -> dict:
        if not self.ready:
            return {"error": self.error or "not loaded"}
        t0 = time.time()
        if snap_bgr is None:
            return {"error": "Khong co snapshot"}
        face, bbox = self._detect_align(snap_bgr)
        if face is None:
            return {"error": "Khong detect duoc khuon mat trong snapshot",
                    "annotated": annotate(snap_bgr, None, "No face", (0, 0, 255))}
        feat = self._hog_feature(face).reshape(1, -1)
        probs = self.svm.predict_proba(feat)[0]
        pred_idx = int(np.argmax(probs))
        pred_prob = float(probs[pred_idx])
        # Map sang ten class
        raw_classes = getattr(self.svm, "classes_", None)
        if raw_classes is None:
            label = self.classes[pred_idx] if pred_idx < len(self.classes) else f"class_{pred_idx}"
        else:
            cls_val = raw_classes[pred_idx]
            if self.le is not None and np.issubdtype(np.asarray(raw_classes).dtype, np.integer):
                label = str(self.le.inverse_transform([cls_val])[0])
            elif np.issubdtype(np.asarray(raw_classes).dtype, np.integer):
                label = self.classes[cls_val] if cls_val < len(self.classes) else f"class_{cls_val}"
            else:
                label = str(cls_val)
        threshold = 0.5
        final_label = label if pred_prob >= threshold else "Unknown"

        sim = None
        if ref_bgr is not None:
            ref_face, _ = self._detect_align(ref_bgr)
            if ref_face is not None:
                ref_feat = self._hog_feature(ref_face)
                sim = cosine(ref_feat, feat[0])

        color = (0, 200, 0) if final_label != "Unknown" else (0, 0, 255)
        annotated = annotate(snap_bgr, bbox, f"{final_label} ({pred_prob:.2f})", color)
        return {
            "label": final_label,
            "confidence": pred_prob,
            "similarity": sim,
            "annotated": annotated,
            "infer_ms": (time.time() - t0) * 1000,
        }


# ---------------------------------------------------------------------------
# Pipeline 2: MTCNN + LBP/Sobel-LBP + SVM (calibrated, open-set 3-signal)
# ---------------------------------------------------------------------------
# Đọc params từ config (đồng bộ với mtcnn_finetuned_crop.py + train_svm.py)
try:
    from config import cfg as _cfg_lbp
    EYE_DIST_RATIO     = float(_cfg_lbp.LBP_EYE_DIST_RATIO)
    EYE_Y_RATIO        = float(_cfg_lbp.LBP_EYE_Y_RATIO)
    MIN_VALID          = float(_cfg_lbp.LBP_MIN_VALID)
    MIN_FACE_PX        = int(_cfg_lbp.LBP_MIN_FACE_PX)
    LBP_OUT_SIZE       = tuple(_cfg_lbp.LBP_OUTPUT_SIZE)
    LBP_INFER_CONF_THR = float(_cfg_lbp.LBP_INFER_CONF_THR)
    LBP_POSE_CHECK     = bool(_cfg_lbp.LBP_POSE_CHECK)
    LBP_NOSE_OFFSET_MAX = float(_cfg_lbp.LBP_NOSE_OFFSET_MAX)
    LBP_EYE_Y_ASYM_MAX  = float(_cfg_lbp.LBP_EYE_Y_ASYM_MAX)
except Exception:  # fallback nếu config chưa có key mới
    EYE_DIST_RATIO = 0.42
    EYE_Y_RATIO    = 0.40
    MIN_VALID      = 0.55
    MIN_FACE_PX    = 30
    LBP_OUT_SIZE   = (128, 128)
    LBP_INFER_CONF_THR = 0.85
    LBP_POSE_CHECK = True
    LBP_NOSE_OFFSET_MAX = 0.35
    LBP_EYE_Y_ASYM_MAX  = 0.15


def _lbp_pose_ok(landmarks, face_h):
    """Reject profile / heavily tilted faces — khớp với mtcnn_finetuned_crop._pose_ok."""
    if not LBP_POSE_CHECK:
        return True
    left_eye  = np.float32(landmarks[0])
    right_eye = np.float32(landmarks[1])
    nose      = np.float32(landmarks[2])
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_dist  = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist < 1.0 or face_h < 1.0:
        return False
    nose_offset = abs(nose[0] - eye_mid_x) / eye_dist
    eye_y_asym  = abs(left_eye[1] - right_eye[1]) / face_h
    return (nose_offset <= LBP_NOSE_OFFSET_MAX) and (eye_y_asym <= LBP_EYE_Y_ASYM_MAX)


def lbp_align_and_crop(img_rgb, box, landmarks, out_size=LBP_OUT_SIZE):
    """Sao y mtcnn_finetuned_crop.align_and_crop + pose sanity check."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < MIN_FACE_PX or h < MIN_FACE_PX:
        return None
    if not _lbp_pose_ok(landmarks, h):
        return None
    left_eye  = np.float32(landmarks[0])
    right_eye = np.float32(landmarks[1])
    dX = right_eye[0] - left_eye[0]
    dY = right_eye[1] - left_eye[1]
    eye_center = ((left_eye[0] + right_eye[0]) / 2,
                  (left_eye[1] + right_eye[1]) / 2)
    scale = (EYE_DIST_RATIO * out_size[0]) / (np.sqrt(dX**2 + dY**2) + 1e-6)
    M = cv2.getRotationMatrix2D(eye_center, np.degrees(np.arctan2(dY, dX)), scale)
    M[0, 2] += out_size[0] * 0.50 - eye_center[0]
    M[1, 2] += out_size[1] * EYE_Y_RATIO - eye_center[1]
    H_src, W_src = img_rgb.shape[:2]
    mask = cv2.warpAffine(np.ones((H_src, W_src), dtype=np.uint8),
                          M, out_size, flags=cv2.INTER_NEAREST, borderValue=0)
    if mask.mean() < MIN_VALID:
        return None
    aligned = cv2.warpAffine(img_rgb, M, out_size,
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)


class LbpSvmPipeline:
    name = "MTCNN + LBP/Sobel-LBP + SVM"

    def __init__(self):
        self.detector = None
        self.svm = None
        self.scaler = None
        self.pca = None
        self.le = None
        self.thresholds = None
        self.class_stats = None
        self.extract_features = None
        self.error: str | None = None
        self.ready = False

    def load(self):
        try:
            import joblib
            import torch
            from facenet_pytorch import MTCNN as FacenetMTCNN
            from feature_utils import extract_features  # type: ignore

            self.extract_features = extract_features
            self.svm        = joblib.load(LBP_MODEL_DIR / "svm_face.pkl")
            self.scaler     = joblib.load(LBP_MODEL_DIR / "scaler.pkl")
            self.pca        = joblib.load(LBP_MODEL_DIR / "pca.pkl")
            self.le         = joblib.load(LBP_MODEL_DIR / "label_encoder.pkl")
            try:
                self.thresholds = joblib.load(LBP_MODEL_DIR / "thresholds.pkl")
            except Exception:
                self.thresholds = None
            try:
                self.class_stats = joblib.load(LBP_MODEL_DIR / "class_stats.pkl")
            except Exception:
                self.class_stats = None
                print("[LBP] class_stats.pkl không có — Mahalanobis OOD signal tắt")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.detector = FacenetMTCNN(min_face_size=15,
                                         thresholds=[0.5, 0.6, 0.7],
                                         keep_all=True, post_process=False,
                                         device=device)
            self.ready = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print(f"[LBP] load error: {self.error}")

    def _detect_crop(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs, lmks = self.detector.detect(rgb, landmarks=True)
        if boxes is None or len(boxes) == 0:
            return None, None, "no_face"
        best = int(np.argmax(probs))
        if probs[best] < LBP_INFER_CONF_THR:
            return None, None, "low_conf"
        crop = lbp_align_and_crop(rgb, boxes[best], lmks[best])
        if crop is None:
            return None, None, "bad_pose_or_crop"
        return crop, boxes[best], "ok"

    def _mahalanobis(self, feat_p_1d, pred_idx):
        if self.class_stats is None or int(pred_idx) not in self.class_stats:
            return None
        st = self.class_stats[int(pred_idx)]
        diff = feat_p_1d - st["mu"]
        return float(np.sqrt(diff @ st["cov_inv"] @ diff))

    def _embed(self, crop_bgr):
        feat = self.extract_features(crop_bgr).reshape(1, -1)
        feat_s = self.scaler.transform(feat)
        feat_p = self.pca.transform(feat_s)
        return feat_p

    def predict(self, ref_bgr, snap_bgr) -> dict:
        if not self.ready:
            return {"error": self.error or "not loaded"}
        t0 = time.time()
        if snap_bgr is None:
            return {"error": "Khong co snapshot"}
        crop, bbox, reason = self._detect_crop(snap_bgr)
        if crop is None:
            reason_msg = {"no_face":  "No face",
                          "low_conf": "Low conf",
                          "bad_pose_or_crop": "Bad pose"}.get(reason, "No face")
            return {"label": "Unknown",
                    "error": f"MTCNN reject: {reason}",
                    "annotated": annotate(snap_bgr, None, reason_msg, (0, 0, 255))}
        feat_p = self._embed(crop)
        probs = self.svm.predict_proba(feat_p)[0]
        pred_idx = int(np.argmax(probs))
        p_max = float(probs[pred_idx])
        sorted_probs = np.sort(probs)
        margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) >= 2 else 1.0
        cls_name = str(self.le.inverse_transform([pred_idx])[0])
        d_M = self._mahalanobis(feat_p[0], pred_idx)

        # Triple-threshold AND: p_max ≥ T_p, margin ≥ T_m, Mahalanobis ≤ τ_d
        if self.thresholds is not None:
            T_p = self.thresholds.get("T_p_per_class", {}).get(cls_name,
                  self.thresholds.get("T_p", 0.5))
            T_m = self.thresholds.get("T_m", 0.0)
            tau_d = self.thresholds.get("tau_d")  # None nếu legacy model
            ok = (p_max >= T_p) and (margin >= T_m)
            if d_M is not None and tau_d is not None:
                ok = ok and (d_M <= tau_d)
        else:
            ok = p_max >= 0.5
        final = cls_name if ok else "Unknown"

        sim = None
        if ref_bgr is not None:
            ref_crop, _, _ = self._detect_crop(ref_bgr)
            if ref_crop is not None:
                sim = cosine(self._embed(ref_crop)[0], feat_p[0])

        color = (0, 200, 0) if final != "Unknown" else (0, 0, 255)
        d_M_str = f" dM={d_M:.1f}" if d_M is not None else ""
        annotated = annotate(snap_bgr, bbox,
                             f"{final} ({p_max:.2f} m={margin:.2f}{d_M_str})", color)
        return {
            "label": final,
            "confidence": p_max,
            "margin": margin,
            "mahalanobis": d_M,
            "similarity": sim,
            "annotated": annotated,
            "infer_ms": (time.time() - t0) * 1000,
        }


# ---------------------------------------------------------------------------
# Pipeline 3: MTCNN + FaceNet
# ---------------------------------------------------------------------------
class FacenetPipeline:
    name = "MTCNN + FaceNet"
    L2_THRESHOLD = 0.95

    def __init__(self):
        self.mtcnn = None
        self.resnet = None
        self.device = None
        self.db: dict[str, "any"] = {}
        self.error: str | None = None
        self.ready = False

    def load(self, force_rebuild: bool = False):
        try:
            import torch
            from facenet_pytorch import MTCNN, InceptionResnetV1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
                               thresholds=[0.6, 0.7, 0.7], factor=0.709,
                               post_process=True, device=self.device)
            self.resnet = InceptionResnetV1(pretrained="casia-webface").eval().to(self.device)

            if FACENET_DB_PATH.exists() and not force_rebuild:
                self.db = torch.load(str(FACENET_DB_PATH), map_location="cpu", weights_only=False)
                print(f"[FaceNet] Loaded DB ({len(self.db)} identity) tu {FACENET_DB_PATH}")
            else:
                self._build_db()
            self.ready = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print(f"[FaceNet] load error: {self.error}")

    def _build_db(self):
        import torch
        from PIL import Image, ImageOps
        if not DATA_TRAIN.exists():
            raise FileNotFoundError(f"Khong thay {DATA_TRAIN}")
        print(f"[FaceNet] Building DB tu {DATA_TRAIN} ...")
        db = {}
        for person_dir in sorted(DATA_TRAIN.iterdir()):
            if not person_dir.is_dir():
                continue
            embs = []
            for fp in person_dir.iterdir():
                if fp.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                try:
                    img = Image.open(fp)
                    img = ImageOps.exif_transpose(img).convert("RGB")
                except Exception:
                    continue
                face = self.mtcnn(img)
                if face is None:
                    continue
                with torch.no_grad():
                    emb = self.resnet(face.unsqueeze(0).to(self.device)).detach().cpu()
                embs.append(emb)
            if embs:
                db[person_dir.name] = torch.mean(torch.stack(embs), dim=0)
                print(f"   + {person_dir.name}: {len(embs)} anh")
        FACENET_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(db, str(FACENET_DB_PATH))
        self.db = db
        print(f"[FaceNet] Saved {FACENET_DB_PATH} ({len(db)} identity)")

    def _embed(self, img_bgr):
        import torch
        from PIL import Image
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        # Co lay them bbox de annotate
        boxes, _ = self.mtcnn.detect(pil)
        face = self.mtcnn(pil)
        if face is None:
            return None, None
        with torch.no_grad():
            emb = self.resnet(face.unsqueeze(0).to(self.device)).detach().cpu()
        bbox = boxes[0] if boxes is not None and len(boxes) > 0 else None
        return emb, bbox

    def predict(self, ref_bgr, snap_bgr):
        import torch
        if not self.ready:
            return {"error": self.error or "not loaded"}
        t0 = time.time()
        if snap_bgr is None:
            return {"error": "Khong co snapshot"}
        emb, bbox = self._embed(snap_bgr)
        if emb is None:
            return {"error": "MTCNN khong detect duoc",
                    "annotated": annotate(snap_bgr, None, "No face", (0, 0, 255))}
        if not self.db:
            return {"error": "Database trong"}
        dists = {name: float(torch.dist(emb, db_emb).item()) for name, db_emb in self.db.items()}
        best = min(dists, key=dists.get)
        d = dists[best]
        ok = d <= self.L2_THRESHOLD
        final = best if ok else "Unknown"
        # Confidence quy doi tu L2
        sim_pct = max(0.0, (1 - (d ** 2) / 2)) * 100

        sim = None
        if ref_bgr is not None:
            ref_emb, _ = self._embed(ref_bgr)
            if ref_emb is not None:
                d_ref = float(torch.dist(ref_emb, emb).item())
                sim = max(0.0, (1 - (d_ref ** 2) / 2))

        color = (0, 200, 0) if ok else (0, 0, 255)
        annotated = annotate(snap_bgr, bbox, f"{final} (L2={d:.2f})", color)
        return {
            "label": final,
            "confidence": sim_pct / 100,
            "similarity": sim,
            "annotated": annotated,
            "infer_ms": (time.time() - t0) * 1000,
        }


# ---------------------------------------------------------------------------
# Pipeline 4: RetinaFace + ArcFace (InsightFace buffalo_s)
# ---------------------------------------------------------------------------
class RetinaArcfacePipeline:
    name = "RetinaFace + ArcFace"
    THRESHOLD = 0.5

    def __init__(self):
        self.app = None
        self.det_model = None
        self.rec_model = None
        self.db: dict[str, np.ndarray] = {}
        self.error: str | None = None
        self.ready = False

    def load(self, force_rebuild: bool = False):
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name="buffalo_s",
                                    allowed_modules=["detection", "recognition"],
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.det_model = self.app.models["detection"]
            self.rec_model = self.app.models["recognition"]
            if RETINA_DB_PATH.exists() and not force_rebuild:
                self.db = np.load(str(RETINA_DB_PATH), allow_pickle=True).item()
                print(f"[RetinaArc] Loaded DB ({len(self.db)} identity)")
            else:
                self._build_db()
            self.ready = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print(f"[RetinaArc] load error: {self.error}")

    def _detect(self, img_bgr, threshold=0.5):
        h, w = img_bgr.shape[:2]
        pad_h, pad_w = h // 4, w // 4
        padded = cv2.copyMakeBorder(img_bgr, pad_h, pad_h, pad_w, pad_w,
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        self.det_model.det_thresh = threshold
        bboxes, lmks = self.det_model.detect(padded, max_num=0)
        if bboxes is None or len(bboxes) == 0:
            return [], []
        bboxes[:, [0, 2]] -= pad_w
        bboxes[:, [1, 3]] -= pad_h
        lmks -= np.array([pad_w, pad_h])
        return bboxes, lmks

    def _embed(self, img_bgr, landmark):
        from insightface.utils import face_align
        aligned = face_align.norm_crop(img_bgr, landmark=landmark)
        emb = self.rec_model.get_feat(aligned).flatten()
        return emb / (np.linalg.norm(emb) + 1e-9)

    def _largest_idx(self, bboxes):
        if len(bboxes) == 0:
            return -1
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        return int(np.argmax(areas))

    def _build_db(self):
        if not DATA_TRAIN.exists():
            raise FileNotFoundError(f"Khong thay {DATA_TRAIN}")
        print(f"[RetinaArc] Building DB tu {DATA_TRAIN} ...")
        db = {}
        for person_dir in sorted(DATA_TRAIN.iterdir()):
            if not person_dir.is_dir():
                continue
            embs = []
            for fp in person_dir.iterdir():
                if fp.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                bboxes, lmks = self._detect(img)
                if len(bboxes) == 0:
                    continue
                idx = self._largest_idx(bboxes)
                embs.append(self._embed(img, lmks[idx]))
            if embs:
                mean = np.mean(embs, axis=0)
                mean = mean / (np.linalg.norm(mean) + 1e-9)
                db[person_dir.name] = mean
                print(f"   + {person_dir.name}: {len(embs)} anh")
        RETINA_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(RETINA_DB_PATH), db, allow_pickle=True)
        self.db = db
        print(f"[RetinaArc] Saved {RETINA_DB_PATH}")

    def predict(self, ref_bgr, snap_bgr):
        if not self.ready:
            return {"error": self.error or "not loaded"}
        t0 = time.time()
        if snap_bgr is None:
            return {"error": "Khong co snapshot"}
        bboxes, lmks = self._detect(snap_bgr)
        if len(bboxes) == 0:
            return {"error": "Khong detect duoc",
                    "annotated": annotate(snap_bgr, None, "No face", (0, 0, 255))}
        idx = self._largest_idx(bboxes)
        emb = self._embed(snap_bgr, lmks[idx])
        bbox = bboxes[idx][:4]

        best, best_sim = "Unknown", -1.0
        for name, db_emb in self.db.items():
            s = float(np.dot(emb, db_emb))
            if s > best_sim:
                best_sim = s
                best = name
        ok = best_sim >= self.THRESHOLD
        final = best if ok else "Unknown"

        sim = None
        if ref_bgr is not None:
            r_bb, r_lm = self._detect(ref_bgr)
            if len(r_bb) > 0:
                ref_emb = self._embed(ref_bgr, r_lm[self._largest_idx(r_bb)])
                sim = float(np.dot(ref_emb, emb))

        color = (0, 200, 0) if ok else (0, 0, 255)
        annotated = annotate(snap_bgr, bbox, f"{final} ({best_sim:.2f})", color)
        return {
            "label": final,
            "confidence": best_sim,
            "similarity": sim,
            "annotated": annotated,
            "infer_ms": (time.time() - t0) * 1000,
        }


# ---------------------------------------------------------------------------
# Pipeline 5: SCRFD + ArcFace + FAISS
# ---------------------------------------------------------------------------
class ScrfdFaissPipeline:
    name = "SCRFD + ArcFace + FAISS"
    THRESHOLD = 0.5

    def __init__(self, model_name: str = "buffalo_s"):
        self.model_name = model_name
        self.app = None
        self.faiss_index = None
        self.names: list[str] = []
        self.error: str | None = None
        self.ready = False

    @property
    def db_path(self) -> Path:
        return SCRFD_DB_S_PATH if self.model_name == "buffalo_s" else SCRFD_DB_L_PATH

    def load(self, force_rebuild: bool = False):
        try:
            import faiss
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name=self.model_name,
                                    allowed_modules=["detection", "recognition"],
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            if not self.db_path.exists() or force_rebuild:
                self._build_db()
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
            self.names = data["names"]
            vectors = np.array(data["vectors"]).astype("float32")
            # Chuan hoa neu chua
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
            vectors = vectors / norms
            self.faiss_index = faiss.IndexFlatIP(vectors.shape[1])
            self.faiss_index.add(vectors)
            print(f"[SCRFD] Loaded {len(self.names)} identity ({self.model_name})")
            self.ready = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print(f"[SCRFD] load error: {self.error}")

    def _build_db(self):
        if not DATA_TRAIN.exists():
            raise FileNotFoundError(f"Khong thay {DATA_TRAIN}")
        print(f"[SCRFD] Building DB ({self.model_name}) tu {DATA_TRAIN} ...")
        names, vectors = [], []
        for person_dir in sorted(DATA_TRAIN.iterdir()):
            if not person_dir.is_dir():
                continue
            person_vecs = []
            for fp in person_dir.iterdir():
                if fp.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                faces = self.app.get(img)
                if not faces:
                    continue
                faces = sorted(faces,
                               key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                               reverse=True)
                person_vecs.append(faces[0].embedding)
            if person_vecs:
                m = np.mean(person_vecs, axis=0)
                m = m / (np.linalg.norm(m) + 1e-9)
                names.append(person_dir.name)
                vectors.append(m)
                print(f"   + {person_dir.name}: {len(person_vecs)} anh")
        SCRFD_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump({"names": names, "vectors": vectors}, f)
        print(f"[SCRFD] Saved {self.db_path}")

    def predict(self, ref_bgr, snap_bgr):
        if not self.ready:
            return {"error": self.error or "not loaded"}
        t0 = time.time()
        if snap_bgr is None:
            return {"error": "Khong co snapshot"}
        faces = self.app.get(snap_bgr)
        if not faces:
            return {"error": "Khong detect duoc",
                    "annotated": annotate(snap_bgr, None, "No face", (0, 0, 255))}
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-9)
        D, I = self.faiss_index.search(np.asarray([emb], dtype="float32"), k=1)
        score = float(D[0][0])
        idx = int(I[0][0])
        ok = score >= self.THRESHOLD
        final = self.names[idx] if ok else "Unknown"

        sim = None
        if ref_bgr is not None:
            ref_faces = self.app.get(ref_bgr)
            if ref_faces:
                rf = max(ref_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                ref_emb = rf.embedding / (np.linalg.norm(rf.embedding) + 1e-9)
                sim = float(np.dot(ref_emb, emb))

        bbox = face.bbox.astype(int)
        color = (0, 200, 0) if ok else (0, 0, 255)
        annotated = annotate(snap_bgr, bbox, f"{final} ({score:.2f})", color)
        return {
            "label": final,
            "confidence": score,
            "similarity": sim,
            "annotated": annotated,
            "infer_ms": (time.time() - t0) * 1000,
        }


# ---------------------------------------------------------------------------
# Pipeline 5 combined: SCRFD chay ca 2 variant buffalo_s + buffalo_l
# trong 1 logical pipeline, hien thi 1 card voi metrics so sanh 2 variant.
# ---------------------------------------------------------------------------
class ScrfdCombinedPipeline:
    name = "SCRFD + ArcFace + FAISS"

    def __init__(self):
        self.inner_s = ScrfdFaissPipeline(model_name="buffalo_s")
        self.inner_l = ScrfdFaissPipeline(model_name="buffalo_l")
        self.error: str | None = None
        self.ready = False

    def load(self, force_rebuild: bool = False):
        try:
            self.inner_s.load(force_rebuild=force_rebuild)
        except Exception as e:
            self.inner_s.error = f"{type(e).__name__}: {e}"
            self.inner_s.ready = False
        try:
            self.inner_l.load(force_rebuild=force_rebuild)
        except Exception as e:
            self.inner_l.error = f"{type(e).__name__}: {e}"
            self.inner_l.ready = False
        # Ready neu it nhat 1 variant load OK
        self.ready = self.inner_s.ready or self.inner_l.ready
        errs = []
        if not self.inner_s.ready:
            errs.append(f"S: {self.inner_s.error or 'unknown'}")
        if not self.inner_l.ready:
            errs.append(f"L: {self.inner_l.error or 'unknown'}")
        self.error = " | ".join(errs) if errs else None

    def predict(self, ref_bgr, snap_bgr) -> dict:
        if not self.ready:
            return {"error": self.error or "not loaded"}
        rs = self.inner_s.predict(ref_bgr, snap_bgr) if self.inner_s.ready else {"error": "S not loaded"}
        rl = self.inner_l.predict(ref_bgr, snap_bgr) if self.inner_l.ready else {"error": "L not loaded"}
        # Uu tien dung ket qua cua L lam main annotated; fallback ve S.
        main = rl if rl.get("annotated") is not None else rs
        if main.get("annotated") is None:
            return {"error": f"Ca 2 variant deu loi (S: {rs.get('error')}, L: {rl.get('error')})"}
        total_ms = (rs.get("infer_ms") or 0) + (rl.get("infer_ms") or 0)
        return {
            "label": main.get("label", "Unknown"),
            "confidence": main.get("confidence"),
            "annotated": main["annotated"],
            "infer_ms": total_ms,
            "variants": [
                {"name": "buffalo_s", **rs},
                {"name": "buffalo_l", **rl},
            ],
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class App:
    def __init__(self, force_rebuild: bool = False):
        self.force_rebuild = force_rebuild
        self.pipes: list[Any] = []

    def init_all(self):
        # 5 logical pipeline. SCRFD chay ca buffalo_s + buffalo_l trong 1 wrapper.
        constructors = [
            ("HOG + SVM",                   lambda: HogSvmPipeline(),         {}),
            ("MTCNN + LBP/Sobel-LBP + SVM",     lambda: LbpSvmPipeline(),         {}),
            ("MTCNN + FaceNet",             lambda: FacenetPipeline(),        {"force_rebuild": self.force_rebuild}),
            ("RetinaFace + ArcFace",        lambda: RetinaArcfacePipeline(),  {"force_rebuild": self.force_rebuild}),
            ("SCRFD + ArcFace + FAISS",     lambda: ScrfdCombinedPipeline(),  {"force_rebuild": self.force_rebuild}),
        ]
        for label, factory, load_kw in constructors:
            print(f"\n=== Loading: {label} ===")
            inst = factory()
            inst.name = label  # override de phan biet 2 variant SCRFD
            try:
                inst.load(**load_kw) if load_kw else inst.load()
            except Exception as e:
                inst.error = f"{type(e).__name__}: {e}"
                inst.ready = False
                print(f"[{label}] FAIL: {inst.error}")
                traceback.print_exc()
            self.pipes.append(inst)

    def status_table(self) -> str:
        rows = ["| Pipeline | Status |", "|----------|--------|"]
        for p in self.pipes:
            rows.append(f"| {p.name} | {'OK' if p.ready else ('ERROR: ' + (p.error or 'unknown'))} |")
        return "\n".join(rows)

    def run_all(self, ref_rgb, snap_rgb):
        ref_bgr = to_bgr(ref_rgb)
        snap_bgr = to_bgr(snap_rgb)
        results = []
        for p in self.pipes:
            try:
                r = p.predict(ref_bgr, snap_bgr)
            except Exception as e:
                r = {"error": f"{type(e).__name__}: {e}"}
                traceback.print_exc()
            results.append((p.name, r))
        return results


# ---------------------------------------------------------------------------
# Gradio UI — Claude Design (face-demo) styling
# ---------------------------------------------------------------------------
import html as _html


def _esc(s) -> str:
    """HTML-escape, gracefully xu ly None."""
    return _html.escape(str(s)) if s is not None else ""


def _result_kind(r: dict) -> str:
    """Map result dict -> kind: success | unknown | error."""
    if r is None:
        return "empty"
    if "annotated" not in r and r.get("error"):
        return "error"
    if r.get("label") == "Unknown":
        return "unknown"
    return "success"


def _status_badge(kind: str, text: str) -> str:
    """Render <span class='badge ...'>...</span>."""
    return f'<span class="badge {kind}"><span class="b-dot"></span>{_esc(text)}</span>'


def _status_table_html(pipes) -> str:
    """Render status card hien thi tinh trang load 6 pipeline."""
    ok = sum(1 for p in pipes if p.ready)
    err = len(pipes) - ok
    err_pill = f' <span class="badge err"><span class="b-dot"></span>{err} lỗi</span>' if err else ""
    rows = []
    for i, p in enumerate(pipes, 1):
        sub = PIPELINE_SUBS.get(p.name, "")
        if p.ready:
            badge = _status_badge("ok", "OK")
            err_small = ""
            row_cls = ""
        else:
            badge = _status_badge("err", "ERROR")
            err_small = (f'<small style="color:var(--red-700);font-family:var(--mono);">'
                         f'{_esc(p.error or "unknown")}</small>')
            row_cls = " err-row"
        rows.append(
            f'<div class="status-row{row_cls}">'
            f'  <span class="num">P{i}</span>'
            f'  <div class="name">{_esc(p.name)}'
            f'    <small>{_esc(sub)}</small>'
            f'    {err_small}'
            f'  </div>'
            f'  {badge}'
            f'</div>'
        )
    return (
        '<div class="card status-card" id="status_card">'
        '  <div class="card-head">'
        '    <div class="card-title">'
        '      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" width="16" height="16">'
        '        <rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
        '        <path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 15h3M1 9h3M1 15h3"/>'
        '      </svg>'
        f'      Trạng thái khởi tạo pipeline <span class="sub">· {ok}/{len(pipes)} sẵn sàng</span>'
        '    </div>'
        f'    <div style="display:flex;gap:8px;align-items:center;">{err_pill}'
        '      <button type="button" class="btn btn-ghost" id="status_toggle"'
        '        style="height:30px;padding:0 12px;font-size:12px;">Thu gọn</button>'
        '    </div>'
        '  </div>'
        f'  <div class="status-list" id="status_list">{"".join(rows)}</div>'
        '</div>'
    )


def _result_header_html(idx: int, pipe_name: str, kind: str) -> str:
    """Header card kem badge trang thai. idx tu 0."""
    sub = PIPELINE_SUBS.get(pipe_name, "")
    if kind == "empty":
        badge = _status_badge("ok", "Sẵn sàng")
    elif kind == "loading":
        badge = _status_badge("info", "Đang xử lý…")
    elif kind == "success":
        badge = (
            '<span class="badge ok"><span class="b-dot"></span>'
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12">'
            '<path d="M20 6L9 17l-5-5"/></svg>Nhận diện</span>'
        )
    elif kind == "unknown":
        badge = _status_badge("warn", "Unknown")
    else:  # error
        badge = (
            '<span class="badge err"><span class="b-dot"></span>'
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" width="12" height="12">'
            '<path d="M10.3 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
            '<path d="M12 9v4"/><path d="M12 17h.01"/></svg>Lỗi</span>'
        )
    return (
        '<div class="result-head">'
        '  <div class="result-title">'
        f'    <div class="pipe-num">P{idx + 1}</div>'
        f'    <div><div class="pipe-name">{_esc(pipe_name)}<small>{_esc(sub)}</small></div></div>'
        '  </div>'
        f'  {badge}'
        '</div>'
    )


def _result_metrics_html(r: dict | None, kind: str) -> str:
    """Render metrics row duoi anh: Predict / Confidence / Time / Note."""
    if kind == "empty":
        return (
            '<div class="result-metrics">'
            '  <div class="metric"><span class="k">Predict</span><span class="v muted">—</span></div>'
            '  <div class="metric"><span class="k">Confidence</span><span class="v muted">—</span></div>'
            '  <div class="metric" style="grid-column:1/-1;"><span class="k">Time</span><span class="v muted">—</span></div>'
            '</div>'
        )
    if kind == "loading":
        return (
            '<div class="result-metrics">'
            '  <div class="metric"><span class="k">Predict</span><span class="skel" style="width:64px;height:14px;"></span></div>'
            '  <div class="metric"><span class="k">Confidence</span><span class="skel" style="width:48px;height:14px;"></span></div>'
            '  <div class="metric" style="grid-column:1/-1;"><span class="k">Time</span><span class="skel" style="width:48px;height:14px;"></span></div>'
            '</div>'
        )
    if kind == "error":
        msg = r.get("error", "Pipeline loi") if r else "Pipeline loi"
        return (
            '<div class="result-metrics">'
            '  <div class="metric"><span class="k">Predict</span><span class="v muted">—</span></div>'
            '  <div class="metric"><span class="k">Confidence</span><span class="v muted">—</span></div>'
            '  <div class="metric" style="grid-column:1/-1;"><span class="k">Time</span><span class="v muted">—</span></div>'
            f'  <div class="note-row"><span>⚠</span><span>{_esc(msg)}</span></div>'
            '</div>'
        )
    # success / unknown
    label = r.get("label", "?")
    conf = r.get("confidence")
    conf_s = f"{conf:.3f}" if conf is not None else "—"
    t = r.get("infer_ms")
    t_s = f"{t:.0f} ms" if t is not None else "—"
    tag_cls = "tag-unknown" if kind == "unknown" else "tag-name"
    note_html = ""
    if r.get("error"):
        note_html = (
            f'<div class="note-row warn"><span>ⓘ</span><span>{_esc(r["error"])}</span></div>'
        )

    # Neu pipeline co "variants" (vd: SCRFD S+L) -> render block so sanh variant
    variants_html = ""
    if r.get("variants"):
        rows = []
        for v in r["variants"]:
            v_name = v.get("name", "?")
            if v.get("error") and v.get("annotated") is None:
                rows.append(
                    f'<div style="display:flex;justify-content:space-between;padding:5px 0;font-size:11px;border-top:1px dashed var(--neutral-3);">'
                    f'  <span style="font-family:var(--mono);color:var(--base-text-second);font-weight:700;">{_esc(v_name)}</span>'
                    f'  <span style="color:var(--red-700);">— {_esc(v.get("error", "loi"))}</span>'
                    f'</div>'
                )
            else:
                v_label = v.get("label", "?")
                v_conf = v.get("confidence")
                v_conf_s = f"{v_conf:.3f}" if v_conf is not None else "—"
                v_t = v.get("infer_ms")
                v_t_s = f"{v_t:.0f} ms" if v_t is not None else "—"
                v_unk = v_label == "Unknown"
                v_tag_cls = "tag-unknown" if v_unk else "tag-name"
                rows.append(
                    f'<div style="display:grid;grid-template-columns:auto 1fr auto auto;gap:8px;align-items:center;padding:6px 0;font-size:11px;border-top:1px dashed var(--neutral-3);">'
                    f'  <span style="font-family:var(--mono);color:var(--base-text-second);font-weight:700;">{_esc(v_name)}</span>'
                    f'  <span class="v {v_tag_cls}" style="font-family:var(--mono);font-size:10px;justify-self:start;">{_esc(v_label)}</span>'
                    f'  <span style="font-family:var(--mono);font-weight:700;">{v_conf_s}</span>'
                    f'  <span style="font-family:var(--mono);color:var(--base-text-second);">{v_t_s}</span>'
                    f'</div>'
                )
        variants_html = (
            '<div style="grid-column:1/-1;margin-top:6px;">'
            '  <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.06em;color:var(--base-text-second);font-weight:700;margin-bottom:2px;">So sánh 2 variant</div>'
            + "".join(rows) +
            '</div>'
        )

    return (
        '<div class="result-metrics">'
        f'  <div class="metric"><span class="k">Predict</span>'
        f'    <span class="v {tag_cls}">{_esc(label)}</span></div>'
        f'  <div class="metric"><span class="k">Confidence</span><span class="v">{conf_s}</span></div>'
        f'  <div class="metric" style="grid-column:1/-1;"><span class="k">Time</span><span class="v">{t_s}</span></div>'
        f'  {variants_html}'
        f'  {note_html}'
        '</div>'
    )


def _result_card_class(kind: str) -> str:
    """Map kind -> class boi them cho .result-card (de mau border-top)."""
    if kind in ("success", "unknown", "error"):
        return kind
    return ""


def _summary_table_html(results: list | None, view_state: str) -> str:
    """Render bang tong ket so sanh pipeline (plain table, khong highlight)."""
    head = (
        '<div class="card summary-card">'
        '  <div class="card-head">'
        '    <div class="card-title">'
        '      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" width="18" height="18">'
        '        <path d="M12 2 2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>'
        '      </svg>'
        '      Bảng tổng kết <span class="sub">· so sánh các pipeline trên cùng ảnh</span>'
        '    </div>'
        '  </div>'
        '  <div style="overflow-x:auto;">'
        '    <table class="summary"><thead><tr>'
        '      <th style="width:50px;">#</th>'
        '      <th>Pipeline</th>'
        '      <th>Predict</th>'
        '      <th class="num">Confidence</th>'
        '      <th class="num">Time</th>'
        '    </tr></thead><tbody>'
    )
    if not results or view_state != "results":
        body_rows = []
        for i, name in enumerate(PIPELINE_SUBS.keys(), 1):
            sub = PIPELINE_SUBS[name]
            body_rows.append(
                f'<tr><td class="num" style="color:var(--base-text-second);">P{i}</td>'
                f'<td><div style="font-weight:700;color:var(--neutral-9);">{_esc(name)}</div>'
                f'<div style="font-size:11px;color:var(--base-text-second);">{_esc(sub)}</div></td>'
                '<td><span style="color:var(--base-text-second);">—</span></td>'
                '<td class="num">—</td><td class="num">—</td></tr>'
            )
        return head + "".join(body_rows) + '</tbody></table></div></div>'

    body_rows = []
    for i, (name, r) in enumerate(results):
        kind = _result_kind(r)
        sub = PIPELINE_SUBS.get(name, "")
        row_cls = "err-row" if kind == "error" else ""
        # Predict cell
        if kind == "error":
            predict_cell = '<span style="color:var(--base-text-second);">—</span>'
        elif kind == "unknown":
            predict_cell = '<span class="v tag-unknown" style="font-family:var(--mono);">Unknown</span>'
        else:
            predict_cell = f'<span class="v tag-name" style="font-family:var(--mono);">{_esc(r.get("label", "?"))}</span>'
        conf = r.get("confidence")
        conf_cell = f'{conf:.3f}' if conf is not None else "—"
        t = r.get("infer_ms")
        time_cell = f'{t:.0f} ms' if t is not None else "—"
        body_rows.append(
            f'<tr class="{row_cls}">'
            f'  <td class="num" style="color:var(--base-text-second);">P{i + 1}</td>'
            f'  <td><div style="font-weight:700;color:var(--neutral-9);">{_esc(name)}</div>'
            f'    <div style="font-size:11px;color:var(--base-text-second);">{_esc(sub)}</div></td>'
            f'  <td>{predict_cell}</td>'
            f'  <td class="num">{conf_cell}</td>'
            f'  <td class="num">{time_cell}</td>'
            f'</tr>'
        )
    return head + "".join(body_rows) + '</tbody></table></div></div>'


# ---------------------------------------------------------------------------
# CSS tu Claude Design (tokens.css + face-demo.html). Embed nguyen ban de
# Gradio render. Mot vai overrides cuoi cung dieu chinh layout container.
# ---------------------------------------------------------------------------
_FACE_DEMO_CSS = r"""
/* ============ tokens.css ============ */
:root {
  --primary: #418dff;
  --primary-hover: #1b63f5;
  --primary-active: #144ee1;
  --primary-active-hover: #173fb6;
  --primary-bg: #d9ebff;
  --primary-foreground: #ffffff;
  --secondary: #ffa921;
  --secondary-hover: #ea9f25;
  --secondary-bg: #ffefc6;
  --secondary-foreground: #ffffff;
  --success: #4caf50;
  --success-hover: #45a049;
  --warning: #ff9800;
  --warning-hover: #e68a00;
  --danger: #f44336;
  --danger-hover: #e53935;
  --info: #2196f3;
  --info-hover: #1976d2;
  --base-text: #212529;
  --base-text-second: #6c757d;
  --base-text-disabled: #adb5bd;
  --base-text-link: #418dff;
  --base-text-link-hover: #1b63f5;
  --background: #ffffff;
  --foreground: #333333;
  --card: #ffffff;
  --muted: #f9fafb;
  --muted-foreground: #6b7280;
  --accent: #e0f2fe;
  --accent-foreground: #418dff;
  --border: #e5e7eb;
  --input: #e5e7eb;
  --ring: var(--primary);
  --blue-50:#e6f4ff; --blue-100:#bae0ff; --blue-200:#91caff; --blue-700:#003eb3;
  --geekblue-500:#2f54eb;
  --green-50:#f6ffed; --green-200:#b7eb8f; --green-400:#73d13d; --green-500:#52c41a; --green-700:#237804;
  --gold-50:#fffbe6; --gold-200:#ffe58f; --gold-400:#ffc53d; --gold-500:#faad14; --gold-700:#ad6800; --gold-800:#874d00;
  --red-50:#fff1f0; --red-200:#ffa39e; --red-700:#a8071a;
  --neutral-2:#fafafa; --neutral-3:#f5f5f5; --neutral-4:#f0f0f0; --neutral-5:#d9d9d9; --neutral-8:#595959; --neutral-9:#262626;
  --radius: 0.5rem;
  --radius-sm: 0.25rem;
  --radius-md: 0.375rem;
  --radius-xl: 0.75rem;
  --radius-2xl: 1rem;
  --radius-full: 9999px;
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.08), 0 1px 2px -1px rgb(0 0 0 / 0.06);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.08), 0 2px 4px -2px rgb(0 0 0 / 0.05);
  --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.08), 0 4px 6px -4px rgb(0 0 0 / 0.05);
  --font-sans: "Nunito", "Roboto", Inter, Avenir, Helvetica, Arial, sans-serif;
  --font-body: "Roboto", "Nunito", Inter, Helvetica, Arial, sans-serif;
  --page-bg: #f4f6fa;
  --shell-bg: #ffffff;
  --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

/* ============ Gradio container overrides ============ */
html, body {
  background:
    radial-gradient(1200px 600px at 50% -10%, rgba(65,141,255,0.06) 0%, transparent 60%),
    var(--page-bg) !important;
  color: var(--base-text) !important;
  margin: 0; padding: 0;
}
gradio-app {
  background: transparent !important;
  display: block;
  width: 100%;
}
.gradio-container {
  max-width: 1600px !important;
  margin: 0 auto !important;
  background: transparent !important;
  font-family: var(--font-body) !important;
  color: var(--base-text) !important;
  padding: 28px 32px 60px !important;
}
.gradio-container .main, .gradio-container > .main > .wrap { background: transparent !important; }
.gradio-container .prose h1, .gradio-container .prose h2,
.gradio-container .prose h3, .gradio-container .prose p { margin: 0 !important; }
footer { display: none !important; }  /* hide gradio footer */

/* ============ Base typography ============ */
h1,h2,h3,h4,h5,h6 { font-family: var(--font-sans); margin: 0; color: var(--base-text); }
.h-display { font-family: var(--font-sans); font-weight: 800; letter-spacing: -0.01em; }
.mono { font-family: var(--mono); }

/* ============ Hero ============ */
.hero { display:flex; align-items:flex-end; justify-content:space-between; gap:24px; }
.hero h1 { font-size: 52px; line-height: 1.12; font-weight: 800; color: var(--neutral-9) !important; }
.hero p.lede {
  margin-top: 14px !important;
  color: var(--base-text-second) !important;
  font-size: 20px;
  max-width: 920px;
  line-height: 1.6;
  background: transparent !important;
}
.hero p.lede b, .hero p.lede strong {
  color: var(--primary) !important;
  background: transparent !important;
  font-weight: 700;
}

/* ============ Card / workbench ============ */
.card {
  background: var(--shell-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-sm);
  overflow: hidden;
}
.card-head {
  display:flex; align-items:center; justify-content:space-between;
  padding: 18px 22px;
  border-bottom: 1px solid var(--border);
}
.card-title {
  font-family: var(--font-sans); font-weight: 700; font-size: 18px; color: var(--neutral-9);
  display:flex; align-items:center; gap: 10px;
}
.card-title .sub { font-weight: 500; color: var(--base-text-second); font-size: 14px; }
.card-title svg { width: 20px !important; height: 20px !important; }

.pill {
  display:inline-flex; align-items:center; gap:6px;
  padding: 4px 10px; border-radius: 999px;
  background: var(--muted); border:1px solid var(--border);
  font-size: 12px; color: var(--neutral-8); font-weight: 600;
}
.pill .dot { width:6px; height:6px; border-radius:999px; background: var(--success);
             box-shadow: 0 0 0 3px rgba(76,175,80,0.18); }

/* ============ Input card / dark stage ============ */
#input_card_wrap { display: flex; flex-direction: column; gap: 0; }
#input_card_wrap > .card { padding: 0 !important; }
#snap_in_component {
  border-radius: 0 !important;
  background: linear-gradient(135deg, #11192a 0%, #1d2a47 100%) !important;
  border: 0 !important;
  min-height: 680px;
}
#snap_in_component .upload-container, #snap_in_component .image-container {
  background: transparent !important;
  border: 0 !important;
}
#snap_in_component label { color: #cdd6e6 !important; }

/* ============ Action bar — 3 nut chia deu 1 hang ============ */
.actionbar-wrap {
  padding: 16px 20px !important;
  background: var(--muted) !important;
  border-top: 1px solid var(--border);
  display: flex !important;
  align-items: center !important;
  gap: 12px !important;
  flex-wrap: nowrap !important;
}
.actionbar-wrap > * {
  flex: 1 1 0 !important;
  min-width: 0 !important;
}
.actionbar-wrap button {
  width: 100% !important;
  white-space: nowrap !important;
}

/* Override Gradio default button styles. Tang specificity bang nested selectors. */
.gradio-container .btn-capture button,
.gradio-container .btn-analyze button,
.gradio-container .btn-clear button,
.gradio-container .btn-capture > button,
.gradio-container .btn-analyze > button,
.gradio-container .btn-clear > button {
  border-radius: var(--radius-md) !important;
  font-family: var(--font-body) !important;
  font-weight: 700 !important;
  border: 1px solid transparent !important;
  transition: filter 0.12s ease, transform 0.06s ease, background 0.12s ease !important;
  text-shadow: none !important;
}

/* CHUP — orange */
.gradio-container .btn-capture button,
.gradio-container .btn-capture > button {
  background: #ffa921 !important;
  background-color: #ffa921 !important;
  background-image: linear-gradient(135deg, #ffb851 0%, #ffa921 100%) !important;
  color: #fff !important;
  height: 52px !important; min-height: 52px !important;
  padding: 0 22px !important; font-size: 16px !important;
  box-shadow: 0 6px 14px -3px rgba(255,169,33,0.55) !important;
  border-color: transparent !important;
}
.gradio-container .btn-capture button:hover,
.gradio-container .btn-capture > button:hover {
  background: #ea9f25 !important;
  background-image: linear-gradient(135deg, #ffb851 0%, #ea9f25 100%) !important;
  filter: brightness(1.05);
}

/* PHAN TICH — blue hero */
.gradio-container .btn-analyze button,
.gradio-container .btn-analyze > button {
  height: 58px !important; min-height: 58px !important;
  padding: 0 28px !important; font-size: 17px !important;
  background: #418dff !important;
  background-image: linear-gradient(135deg, #418dff 0%, #144ee1 100%) !important;
  color: #fff !important;
  box-shadow: 0 10px 20px -4px rgba(65,141,255,0.55) !important;
  border-color: transparent !important;
}
.gradio-container .btn-analyze button:hover,
.gradio-container .btn-analyze > button:hover {
  filter: brightness(1.08) !important;
  background-image: linear-gradient(135deg, #1b63f5 0%, #144ee1 100%) !important;
}

/* CLEAR — ghost / outline grey */
.gradio-container .btn-clear button,
.gradio-container .btn-clear > button {
  height: 52px !important; min-height: 52px !important;
  padding: 0 22px !important; font-size: 16px !important;
  background: #ffffff !important;
  background-image: none !important;
  color: #595959 !important;
  border: 1px solid #d9d9d9 !important;
  box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.04) !important;
}
.gradio-container .btn-clear button:hover,
.gradio-container .btn-clear > button:hover {
  background: #f5f5f5 !important;
  border-color: #bfbfbf !important;
  color: #262626 !important;
}

/* ============ Status table ============ */
.status-card .card-head { background: linear-gradient(180deg, #fafbfd, #ffffff); }
.status-list { padding: 6px 0; }
.status-row {
  display:grid; grid-template-columns: 34px 1fr auto; gap: 14px; align-items:center;
  padding: 14px 22px; border-bottom: 1px solid var(--neutral-3); font-size: 15px;
}
.status-row:last-child { border-bottom: 0; }
.status-row .num {
  font-size: 13px; color: var(--base-text-second); font-weight: 700;
  font-family: var(--mono); text-align: right;
}
.status-row .name { color: var(--neutral-9); font-weight: 600; }
.status-row .name small { display:block; color: var(--base-text-second); font-weight: 500; font-size: 13px; margin-top: 3px; }
.status-row.err-row { background: var(--red-50); }
.status-list.collapsed { display: none; }

#status_toggle {
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
  background: transparent;
  color: var(--neutral-8);
  cursor: pointer;
  font-weight: 700;
  height: 30px; padding: 0 12px; font-size: 12px;
  font-family: var(--font-body);
}
#status_toggle:hover { background: var(--neutral-3); }

/* ============ Badge ============ */
.badge {
  display:inline-flex; align-items:center; gap:7px;
  padding: 5px 12px; border-radius: 999px;
  font-size: 13px; font-weight: 700; border:1px solid transparent;
}
.badge .b-dot { width:8px; height:8px; border-radius: 999px; }
.badge.ok   { background: var(--green-50); color: var(--green-700); border-color: var(--green-200); }
.badge.ok   .b-dot { background: var(--success); }
.badge.err  { background: var(--red-50); color: var(--red-700); border-color: var(--red-200); }
.badge.err  .b-dot { background: var(--danger); }
.badge.warn { background: var(--gold-50); color: var(--gold-700); border-color: var(--gold-200); }
.badge.warn .b-dot { background: var(--warning); }
.badge.info { background: var(--blue-50); color: var(--blue-700); border-color: var(--blue-200); }
.badge.info .b-dot { background: var(--info); }

/* ============ Section head + legend ============ */
.section-head {
  margin-top: 28px;
  display:flex; align-items: baseline; justify-content: space-between; gap: 16px;
}
.section-head h2 { font-size: 32px; font-family: var(--font-sans); font-weight: 800; color: var(--neutral-9) !important; }
.section-head .meta { font-size: 15px; color: var(--base-text-second); margin-top: 8px; }
.section-head .meta b, .section-head .meta strong {
  color: var(--primary) !important;
  background: transparent !important;
  font-weight: 700;
}
.legend { display:inline-flex; gap: 18px; align-items:center; }
.legend-item {
  display:inline-flex; gap: 6px; align-items:center;
  font-size: 14px; color: var(--base-text-second); font-weight: 600;
}
.legend-swatch { width: 10px; height: 10px; border-radius: 2px; display:inline-block; }

/* ============ Result cards ============ */
.result-card-wrap {
  background: var(--shell-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-xl);
  overflow: hidden;
  transition: box-shadow 0.15s ease, transform 0.15s ease;
}
.result-card-wrap:hover { box-shadow: var(--shadow-md); transform: translateY(-1px); }
.empty-spacer { visibility: hidden !important; }
.result-card-wrap.success { border-top: 3px solid var(--success); }
.result-card-wrap.unknown { border-top: 3px solid var(--gold-500); }
.result-card-wrap.error   { border-top: 3px solid var(--danger);
                            background: linear-gradient(180deg, #fff5f5 0%, #ffffff 70%); }

.result-head {
  display:flex; align-items:center; justify-content: space-between;
  padding: 16px 18px 14px;
}
.result-title { display:flex; align-items:center; gap: 14px; }
.pipe-num {
  width: 38px; height: 38px; border-radius: 8px;
  display:inline-flex; align-items:center; justify-content:center;
  font-family: var(--mono); font-weight: 700; font-size: 13px;
  background: var(--muted); color: var(--base-text-second);
  border: 1px solid var(--border);
}
.pipe-name { font-family: var(--font-sans); font-weight: 800; font-size: 18px; color: var(--neutral-9); line-height: 1.25; }
.pipe-name small { display:block; font-family: var(--font-body); font-weight: 500; color: var(--base-text-second); font-size: 13px; margin-top: 4px; }

/* Anh ket qua trong card — gioi han trong card */
/* Card styling — KHONG ap dung khi fullscreen (de Gradio's native CSS handle). */
.result-card-wrap .image-container:not(:fullscreen),
.result-card-wrap .image-frame:not(:fullscreen) {
  border-radius: 10px;
  overflow: hidden;
  margin: 0 14px;
  background: #11192a;
}
.result-card-wrap img { object-fit: contain; }

/* Fullscreen — Gradio 6 da co san CSS chuan cho .image-container:fullscreen
   (black bg, flex-center, img max-width:90vw max-height:90vh object-fit:scale-down).
   Minh chi can override 4 properties cua card-custom styling cu de no khong
   chen vao khi fullscreen. KHONG dung den width/height/display/img sizing. */
.image-container:fullscreen {
  background: #000 !important;
  margin: 0 !important;
  border-radius: 0 !important;
  padding: 0 !important;
}

/* AN nut download / share / clear / FULLSCREEN cua gr.Image trong result cards.
   An nut fullscreen cua Gradio vi minh dung custom lightbox JS thay the. */
.result-card-wrap button[aria-label*="ownload" i],
.result-card-wrap button[aria-label*="hare" i],
.result-card-wrap button[aria-label*="lear" i],
.result-card-wrap button[aria-label*="emove" i],
.result-card-wrap button[aria-label*="ullscreen" i],
.result-card-wrap a[download],
.result-card-wrap .icon-button-wrapper button[title*="ownload" i],
.result-card-wrap .icon-button-wrapper button[title*="hare" i],
.result-card-wrap .icon-button-wrapper button[title*="ullscreen" i] {
  display: none !important;
}
/* Cung an nut download/share trong snapshot input */
#snap_in_component button[aria-label*="ownload" i],
#snap_in_component button[aria-label*="hare" i] { display: none !important; }

/* Custom lightbox — bypass Gradio's native fullscreen vi no conflict CSS gay flicker */
.result-card-wrap img { cursor: zoom-in; }
.fr-lightbox {
  position: fixed; inset: 0;
  z-index: 9999;
  background: rgba(8, 12, 22, 0.95);
  backdrop-filter: blur(8px);
  display: flex; align-items: center; justify-content: center;
  padding: 40px;
  cursor: zoom-out;
  animation: fr-fade-in 0.18s ease;
}
@keyframes fr-fade-in { from { opacity: 0; } to { opacity: 1; } }
.fr-lightbox img {
  max-width: calc(100vw - 80px);
  max-height: calc(100vh - 80px);
  width: auto; height: auto;
  object-fit: contain;
  display: block;
  border-radius: 8px;
  box-shadow: 0 30px 80px -10px rgba(0,0,0,0.6);
}
.fr-lightbox .fr-close {
  position: absolute; top: 24px; right: 24px;
  width: 44px; height: 44px; border-radius: 10px;
  background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
  color: #fff; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px; font-weight: 700;
  transition: background 0.12s ease;
}
.fr-lightbox .fr-close:hover { background: rgba(255,255,255,0.2); }
.fr-lightbox .fr-title {
  position: absolute; top: 28px; left: 28px;
  color: #fff; font-family: var(--font-sans); font-weight: 700; font-size: 16px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.16);
  padding: 8px 14px; border-radius: 10px;
}

.result-metrics {
  padding: 16px 18px 18px;
  display:grid; grid-template-columns: 1fr 1fr;
  gap: 4px 20px;
  font-size: 15px;
}
.metric {
  display:flex; align-items:center; justify-content: space-between;
  padding: 9px 0;
  border-bottom: 1px dashed var(--neutral-3);
}
.metric:nth-last-child(-n+2):not(.note-row) { border-bottom: 0; }
.metric .k { color: var(--base-text-second); font-weight: 600; font-size: 14px; }
.metric .v { font-family: var(--mono); font-weight: 700; color: var(--neutral-9); font-size: 14px; }
.metric .v.tag-name {
  padding: 4px 12px; border-radius: 4px;
  background: var(--green-50); color: var(--green-700); border: 1px solid var(--green-200);
  font-family: var(--mono); font-size: 13px;
}
.metric .v.tag-unknown {
  background: var(--gold-50); color: var(--gold-700); border:1px solid var(--gold-200);
  padding: 4px 12px; border-radius: 4px; font-size: 13px;
}
.metric .v.muted { color: var(--base-text-second); font-weight: 500; }
.note-row {
  grid-column: 1 / -1;
  margin-top: 6px;
  padding: 8px 10px;
  background: var(--red-50);
  border: 1px solid var(--red-200);
  border-radius: 6px;
  font-size: 12px;
  color: var(--red-700);
  display:flex; gap: 8px; align-items:flex-start;
}
.note-row.warn { background: var(--gold-50); border-color: var(--gold-200); color: var(--gold-800); }

.skel {
  background: linear-gradient(90deg, var(--neutral-3) 0%, var(--neutral-4) 50%, var(--neutral-3) 100%);
  background-size: 200% 100%;
  animation: shimmer 1.4s ease-in-out infinite;
  border-radius: 4px;
  display:inline-block;
}
@keyframes shimmer { 0%{background-position: 200% 0;} 100%{background-position: -200% 0;} }

/* ============ Summary table ============ */
.summary-card { margin-top: 32px; }
table.summary { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 15px; }
table.summary thead th {
  text-align: left; background: #fafbfd;
  color: var(--base-text-second); font-weight: 700; font-size: 13px;
  text-transform: uppercase; letter-spacing: 0.06em;
  padding: 16px 20px; border-bottom: 1px solid var(--border);
}
table.summary thead th.num, table.summary tbody td.num {
  text-align: right; font-variant-numeric: tabular-nums; font-family: var(--mono);
}
table.summary tbody td {
  padding: 16px 20px; border-bottom: 1px solid var(--neutral-3);
  color: var(--neutral-9); vertical-align: middle;
  font-size: 15px;
}
table.summary tbody tr:last-child td { border-bottom: 0; }
table.summary tbody tr.err-row { background: linear-gradient(90deg, var(--red-50) 0%, transparent 30%); }
table.summary tbody tr:hover { background: var(--muted); }
.rank-trophy {
  display:inline-flex; align-items:center; gap:4px;
  padding: 2px 6px; border-radius: 4px;
  font-size: 10px; font-weight: 700; margin-left: 6px;
}
.rank-trophy.fastest   { background: var(--green-50); color: var(--green-700); border:1px solid var(--green-200); }
.rank-trophy.slowest   { background: var(--gold-50);  color: var(--gold-700);  border:1px solid var(--gold-200); }
.rank-trophy.best-conf { background: var(--blue-50);  color: var(--blue-700);  border:1px solid var(--blue-200); }

/* ============ Footer ============ */
.foot {
  margin-top: 44px; padding: 20px 0;
  border-top: 1px solid var(--border);
  color: var(--base-text-second); font-size: 13px;
  display:flex; justify-content: space-between; align-items: center;
}
.kbd {
  display:inline-flex; align-items:center; justify-content:center;
  min-width: 18px; height: 18px; padding: 0 5px;
  border-radius: 4px; background: var(--neutral-3);
  font-family: var(--font-body); font-size: 11px; font-weight: 700; color: var(--neutral-8);
  margin: 0 2px;
}
"""

_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800&family=Roboto:wght@400;500;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
"""

_HERO_HTML = """
<div class="hero">
  <div>
    <h1 class="h-display">Face Recognition — Demo 5 Pipeline</h1>
    <p class="lede">
      Chụp ảnh từ webcam hoặc tải ảnh lên, sau đó bấm <b>Phân tích</b> để chạy song song 5 pipeline nhận diện khuôn mặt
      trên cùng một ảnh. So sánh kết quả predict, confidence và thời gian inference trực tiếp ở phần dưới.
    </p>
  </div>
</div>
"""

_INPUT_CARD_HEAD = """
<div class="card-head">
  <div class="card-title">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" width="16" height="16">
      <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3Z"/>
      <circle cx="12" cy="13" r="3.5"/>
    </svg>
    Snapshot đầu vào <span class="sub">· chụp hoặc tải ảnh</span>
  </div>
  <span class="pill"><span class="dot"></span>Webcam OK</span>
</div>
"""

_SECTION_HEAD_HTML = """
<div class="section-head">
  <div>
    <h2>Kết quả</h2>
    <div class="meta">5 pipeline · bấm <b>Phân tích</b> để bắt đầu, click icon fullscreen để xem ảnh to.</div>
  </div>
  <div class="legend">
    <span class="legend-item"><span class="legend-swatch" style="background:var(--success);"></span>Match</span>
    <span class="legend-item"><span class="legend-swatch" style="background:var(--gold-500);"></span>Unknown</span>
    <span class="legend-item"><span class="legend-swatch" style="background:var(--danger);"></span>Error</span>
  </div>
</div>
"""

_FOOTER_HTML = """
<div class="foot">
  <div>FaceLab · CS231 — Đồ án nhận diện khuôn mặt</div>
  <div>Phím tắt: <span class="kbd">Esc</span> thoát fullscreen</div>
</div>
"""


def build_ui(app: App):
    import gradio as gr

    N_PIPES = len(app.pipes)

    # CSS + head duoc truyen vao launch() o Gradio 6 (khong con o Blocks init).
    # Luu them vao demo attribute de main() lay ra.
    with gr.Blocks(title="Face Recognition — Demo 6 Pipeline") as demo:
        demo._face_demo_css = _FACE_DEMO_CSS
        demo._face_demo_head = _HEAD
        gr.HTML(_HERO_HTML)

        # ===== Workbench: input card (left) + status card (right) =====
        with gr.Row(equal_height=False):
            with gr.Column(scale=145, min_width=560, elem_id="input_card_wrap"):
                with gr.Column(elem_classes="card"):
                    gr.HTML(_INPUT_CARD_HEAD)
                    snap_in = gr.Image(
                        label="",
                        sources=["webcam", "upload"],
                        type="numpy",
                        height=680,
                        show_label=False,
                        elem_id="snap_in_component",
                    )
                    with gr.Row(elem_classes="actionbar-wrap"):
                        capture_btn = gr.Button("📷  Chụp", elem_classes="btn-capture")
                        analyze_btn = gr.Button("▶  Phân tích 5 pipeline", elem_classes="btn-analyze")
                        clear_btn = gr.Button("🗑  Chụp lại / Clear", elem_classes="btn-clear")

            with gr.Column(scale=100, min_width=380):
                status_html = gr.HTML(_status_table_html(app.pipes))

        # ===== Section head + result grid =====
        gr.HTML(_SECTION_HEAD_HTML)

        result_headers: list = []
        result_imgs: list = []
        result_metrics: list = []
        # Row 1: 3 card. Row 2: 2 card + 1 spacer (de chieu rong card dong nhat).
        layout = [
            [0, 1, 2],
            [3, 4, None],   # None = spacer empty column
        ]
        for row in layout:
            with gr.Row(equal_height=True):
                for slot in row:
                    if slot is None or slot >= N_PIPES:
                        with gr.Column(scale=1, min_width=380, elem_classes="empty-spacer"):
                            gr.HTML("&nbsp;")  # invisible spacer giu lai 1/3 width
                        continue
                    pipe = app.pipes[slot]
                    with gr.Column(scale=1, min_width=380, elem_classes="result-card-wrap"):
                        hdr = gr.HTML(_result_header_html(slot, pipe.name, "empty"))
                        img = gr.Image(
                            label="",
                            interactive=False,
                            height=380,
                            show_label=False,
                            container=False,
                        )
                        met = gr.HTML(_result_metrics_html(None, "empty"))
                        result_headers.append(hdr)
                        result_imgs.append(img)
                        result_metrics.append(met)

        summary_html = gr.HTML(_summary_table_html(None, "empty"))
        gr.HTML(_FOOTER_HTML)

        # ===== Callbacks =====
        def _run(snap):
            if snap is None:
                # Khong co snapshot -> giu nguyen empty
                hdrs = [_result_header_html(i, app.pipes[i].name, "empty") for i in range(N_PIPES)]
                imgs = [None] * N_PIPES
                mets = [_result_metrics_html(None, "empty")] * N_PIPES
                return (*hdrs, *imgs, *mets, _summary_table_html(None, "empty"))
            results = app.run_all(None, snap)
            hdrs, imgs, mets = [], [], []
            for i, (name, r) in enumerate(results):
                kind = _result_kind(r)
                hdrs.append(_result_header_html(i, name, kind))
                imgs.append(to_rgb(r["annotated"]) if r.get("annotated") is not None else None)
                mets.append(_result_metrics_html(r, kind))
            return (*hdrs, *imgs, *mets, _summary_table_html(results, "results"))

        def _clear():
            hdrs = [_result_header_html(i, app.pipes[i].name, "empty") for i in range(N_PIPES)]
            imgs = [None] * N_PIPES
            mets = [_result_metrics_html(None, "empty")] * N_PIPES
            return (None, *hdrs, *imgs, *mets, _summary_table_html(None, "empty"))

        # JS: nut "Chup" -> click nut camera noi bo cua Gradio webcam de commit frame
        _capture_js = r"""
        () => {
            const container = document.getElementById('snap_in_component');
            if (!container) return;
            const video = container.querySelector('video');
            if (!video) return;
            const selectors = [
                'button[aria-label="Capture photo"]',
                'button[aria-label*="apture" i]',
                'button[title*="apture" i]',
                'button[aria-label*="hoto" i]',
                '.controls button',
                '.webcam button',
            ];
            for (const sel of selectors) {
                const cands = container.querySelectorAll(sel);
                for (const b of cands) {
                    if (b && b.offsetParent !== null) { b.click(); return; }
                }
            }
        }
        """
        # JS: toggle thu gon status table
        _status_toggle_js = r"""
        () => {
            if (window.__statusToggleInstalled) return;
            window.__statusToggleInstalled = true;
            document.addEventListener('click', (e) => {
                const btn = e.target.closest('#status_toggle');
                if (!btn) return;
                const list = document.getElementById('status_list');
                if (!list) return;
                const collapsed = list.classList.toggle('collapsed');
                btn.textContent = collapsed ? 'Mở rộng' : 'Thu gọn';
            });
        }
        """

        # Custom lightbox: click anh result -> mo modal full screen voi anh.
        # Bypass Gradio's native fullscreen API vi no conflict CSS gay flicker.
        _lightbox_js = r"""
        () => {
            if (window.__lightboxInstalled) return;
            window.__lightboxInstalled = true;
            const openLightbox = (imgSrc, title) => {
                const existing = document.querySelector('.fr-lightbox');
                if (existing) existing.remove();
                const box = document.createElement('div');
                box.className = 'fr-lightbox';
                const titleEl = document.createElement('div');
                titleEl.className = 'fr-title';
                titleEl.textContent = title || 'Result';
                const img = document.createElement('img');
                img.src = imgSrc;
                img.alt = title || '';
                const close = document.createElement('button');
                close.className = 'fr-close';
                close.setAttribute('aria-label', 'Đóng (Esc)');
                close.textContent = '✕';
                close.onclick = (e) => { e.stopPropagation(); box.remove(); };
                box.onclick = (e) => { if (e.target === box) box.remove(); };
                box.appendChild(titleEl);
                box.appendChild(img);
                box.appendChild(close);
                document.body.appendChild(box);
            };
            // Esc -> dong lightbox
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    const lb = document.querySelector('.fr-lightbox');
                    if (lb) lb.remove();
                }
            });
            // Delegate click vao anh trong .result-card-wrap
            document.addEventListener('click', (e) => {
                const img = e.target.closest('.result-card-wrap img');
                if (!img || !img.src) return;
                // Lay ten pipeline tu header trong cung card
                const card = img.closest('.result-card-wrap');
                let title = '';
                if (card) {
                    const nameEl = card.querySelector('.pipe-name');
                    if (nameEl) {
                        title = (nameEl.childNodes[0]?.textContent || '').trim();
                    }
                }
                openLightbox(img.src, title);
            });
        }
        """

        capture_btn.click(fn=None, inputs=None, outputs=None, js=_capture_js)
        analyze_btn.click(
            fn=_run,
            inputs=[snap_in],
            outputs=[*result_headers, *result_imgs, *result_metrics, summary_html],
        )
        clear_btn.click(
            fn=_clear,
            inputs=None,
            outputs=[snap_in, *result_headers, *result_imgs, *result_metrics, summary_html],
        )
        demo.load(fn=None, js=_status_toggle_js)
        demo.load(fn=None, js=_lightbox_js)

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _check_numpy_version():
    """Canh bao neu numpy >= 2 — facenet-pytorch chua fully support."""
    try:
        major = int(np.__version__.split(".")[0])
    except Exception:
        return
    if major >= 2:
        print("=" * 70)
        print(f"[WARN] numpy {np.__version__} co the gay loi voi facenet-pytorch")
        print("       (loi 'Could not infer dtype of numpy.uint8' o MTCNN+LBP, MTCNN+FaceNet).")
        print("       Fix: pip install \"numpy<2\"")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-db", action="store_true",
                        help="Rebuild lai DB embedding cho 3 pipeline embedding")
    parser.add_argument("--check", action="store_true",
                        help="Chi load model va in status, khong khoi dong UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    _check_numpy_version()

    app = App(force_rebuild=args.rebuild_db)
    app.init_all()

    print("\n" + app.status_table())

    if args.check:
        return

    import gradio as gr
    demo = build_ui(app)
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        css=getattr(demo, "_face_demo_css", None),
        head=getattr(demo, "_face_demo_head", None),
    )


if __name__ == "__main__":
    main()
