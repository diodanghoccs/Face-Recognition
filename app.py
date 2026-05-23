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
# Pipeline 2: MTCNN + LBP/Sobel-LBP + SVM (calibrated)
# ---------------------------------------------------------------------------
EYE_DIST_RATIO = 0.42
EYE_Y_RATIO    = 0.40
MIN_VALID      = 0.55
MIN_FACE_PX    = 30
LBP_OUT_SIZE   = (128, 128)


def lbp_align_and_crop(img_rgb, box, landmarks, out_size=LBP_OUT_SIZE):
    """Sao y mtcnn_finetuned_crop.align_and_crop, tra ve BGR 128x128."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < MIN_FACE_PX or h < MIN_FACE_PX:
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
            return None, None
        best = int(np.argmax(probs))
        if probs[best] < 0.6:
            return None, None
        crop = lbp_align_and_crop(rgb, boxes[best], lmks[best])
        if crop is None:
            return None, None
        bbox = boxes[best]
        return crop, bbox

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
        crop, bbox = self._detect_crop(snap_bgr)
        if crop is None:
            return {"error": "MTCNN khong detect duoc",
                    "annotated": annotate(snap_bgr, None, "No face", (0, 0, 255))}
        feat_p = self._embed(crop)
        probs = self.svm.predict_proba(feat_p)[0]
        pred_idx = int(np.argmax(probs))
        p_max = float(probs[pred_idx])
        sorted_probs = np.sort(probs)
        margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) >= 2 else 1.0
        cls_name = str(self.le.inverse_transform([pred_idx])[0])

        if self.thresholds is not None:
            T_p = self.thresholds.get("T_p_per_class", {}).get(cls_name,
                  self.thresholds.get("T_p", 0.5))
            T_m = self.thresholds.get("T_m", 0.0)
            ok = (p_max >= T_p) and (margin >= T_m)
        else:
            ok = p_max >= 0.5
        final = cls_name if ok else "Unknown"

        sim = None
        if ref_bgr is not None:
            ref_crop, _ = self._detect_crop(ref_bgr)
            if ref_crop is not None:
                sim = cosine(self._embed(ref_crop)[0], feat_p[0])

        color = (0, 200, 0) if final != "Unknown" else (0, 0, 255)
        annotated = annotate(snap_bgr, bbox, f"{final} ({p_max:.2f})", color)
        return {
            "label": final,
            "confidence": p_max,
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
# Orchestrator
# ---------------------------------------------------------------------------
class App:
    def __init__(self, force_rebuild: bool = False):
        self.force_rebuild = force_rebuild
        self.pipes: list[Any] = []

    def init_all(self):
        constructors = [
            ("HOG + SVM",                 HogSvmPipeline,      {}),
            ("MTCNN + LBP/Sobel + SVM",   LbpSvmPipeline,      {}),
            ("MTCNN + FaceNet",           FacenetPipeline,     {"force_rebuild": self.force_rebuild}),
            ("RetinaFace + ArcFace",      RetinaArcfacePipeline, {"force_rebuild": self.force_rebuild}),
            ("SCRFD + ArcFace + FAISS",   ScrfdFaissPipeline,  {"force_rebuild": self.force_rebuild}),
        ]
        for label, cls, load_kw in constructors:
            print(f"\n=== Loading: {label} ===")
            inst = cls()
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
# Gradio UI
# ---------------------------------------------------------------------------
def format_caption(name: str, r: dict) -> str:
    """One-line caption for Gallery item (shown in fullscreen preview)."""
    if "error" in r and "annotated" not in r:
        return f"{name} — ERROR: {r['error']}"
    parts = [name]
    if "label" in r:
        parts.append(f"Predict: {r['label']}")
    if r.get("confidence") is not None:
        parts.append(f"Conf: {r['confidence']:.3f}")
    if "infer_ms" in r:
        parts.append(f"{r['infer_ms']:.0f} ms")
    if r.get("error"):
        parts.append(f"warn: {r['error']}")
    return " | ".join(parts)


def format_row(name: str, r: dict) -> str:
    """Markdown table row summarizing one pipeline result."""
    if "error" in r and "annotated" not in r:
        return f"| {name} | — | — | — | ERROR: {r['error']} |"
    label = f"`{r['label']}`" if "label" in r else "—"
    conf = f"{r['confidence']:.3f}" if r.get("confidence") is not None else "—"
    t = f"{r['infer_ms']:.0f} ms" if "infer_ms" in r else "—"
    note = r.get("error") or ""
    return f"| {name} | {label} | {conf} | {t} | {note} |"


def build_ui(app: App):
    import gradio as gr

    with gr.Blocks(title="Face Recognition - 5 Pipeline Demo") as demo:
        gr.Markdown("# Face Recognition - 5 Pipeline Demo\n"
                    "Chup **snapshot tu webcam/OBS** (hoac upload anh) roi nhan "
                    "**Chup & Phan tich** de chay 5 pipeline song song.")

        snap_in = gr.Image(
            label="Snapshot (Webcam / OBS)",
            sources=["webcam", "upload"],
            type="numpy",
            height=480,
        )

        with gr.Row():
            capture_btn = gr.Button("Chup & Phan tich", variant="primary", scale=3)
            clear_btn = gr.Button("Chup lai / Clear", variant="secondary", scale=1)

        status_md = gr.Markdown(app.status_table())

        gr.Markdown("## Ket qua")
        result_gallery = gr.Gallery(
            label="Ket qua 5 pipeline (click de fullscreen, dung mui ten de chuyen)",
            columns=5,
            rows=1,
            height=320,
            object_fit="contain",
            show_label=True,
            preview=False,
            allow_preview=True,
            buttons=["download", "fullscreen"],
        )
        summary_md = gr.Markdown()

        _SUMMARY_HEADER = (
            "| Pipeline | Predict | Confidence | Time | Note |\n"
            "|---|---|---|---|---|"
        )

        def _run(snap):
            if snap is None:
                return [], "**Chua co snapshot.** Hay chup webcam hoac upload anh."
            results = app.run_all(None, snap)
            gallery_items = []
            rows = [_SUMMARY_HEADER]
            for (name, r) in results:
                img = to_rgb(r["annotated"]) if r.get("annotated") is not None else None
                if img is not None:
                    gallery_items.append((img, format_caption(name, r)))
                rows.append(format_row(name, r))
            return gallery_items, "\n".join(rows)

        capture_btn.click(fn=_run, inputs=[snap_in], outputs=[result_gallery, summary_md])
        clear_btn.click(
            fn=lambda: (None, [], ""),
            inputs=None,
            outputs=[snap_in, result_gallery, summary_md],
        )

        # Fix Gradio Gallery: trong fullscreen, nut X / phim X khong thoat duoc.
        # Inject JS de:
        #   1) Bat ky click vao nut co aria-label chua "close"/"exit" khi dang
        #      fullscreen -> exitFullscreen() truoc.
        #   2) Phim 'x' / 'X' -> exit fullscreen va dong preview (click nut close).
        _fullscreen_fix_js = r"""
        () => {
            if (window.__fsFixInstalled) return;
            window.__fsFixInstalled = true;

            const exitFs = () => {
                if (document.fullscreenElement) {
                    document.exitFullscreen().catch(() => {});
                }
            };

            const findCloseBtn = () => {
                const sels = [
                    'button[aria-label*="lose" i]',
                    'button[aria-label*="exit" i]',
                    'button[title*="lose" i]',
                    '.gallery .icon-button[aria-label*="lose" i]',
                ];
                for (const s of sels) {
                    const b = document.querySelector(s);
                    if (b && b.offsetParent !== null) return b;
                }
                return null;
            };

            document.addEventListener('click', (e) => {
                if (!document.fullscreenElement) return;
                const btn = e.target.closest(
                    'button[aria-label*="lose" i],' +
                    'button[aria-label*="exit" i],' +
                    'button[title*="lose" i]'
                );
                if (btn) {
                    exitFs();
                }
            }, true);

            document.addEventListener('keydown', (e) => {
                if (e.key === 'x' || e.key === 'X') {
                    exitFs();
                    setTimeout(() => {
                        const b = findCloseBtn();
                        if (b) b.click();
                    }, 50);
                }
            });
        }
        """
        demo.load(fn=None, js=_fullscreen_fix_js)

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
    demo.launch(server_port=args.port, share=args.share, inbrowser=True,
                theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
