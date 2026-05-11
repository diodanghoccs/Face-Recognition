# ╔══════════════════════════════════════════════════════╗
# ║  BƯỚC 4: Crop pipeline dùng MTCNN đã fine-tune      ║
# ║  Load weights mới vào facenet-pytorch MTCNN          ║
# ║  Crop + align giống pipeline gốc nhưng detect tốt   ║
# ║  hơn (không chỉ crop phần mũi nữa)                  ║
# ╚══════════════════════════════════════════════════════╝
import os
import random
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from facenet_pytorch import MTCNN as FacenetMTCNN

# ── CONFIG ────────────────────────────────────────────────────
SPLIT_ROOT   = r"C:\Users\diosodumb\Pictures\Dataset Split Filtered"
CROP_ROOT    = r"C:\Users\diosodumb\Pictures\Dataset Face Reg_crop_finetuned_masked"
WEIGHTS_DIR  = r"C:\Users\diosodumb\Pictures\mtcnn_finetune\weights"
OUTPUT_SIZE  = (128, 128) #đã thay đổi từ (160, 160) để phù hợp với crop siết hơn, tránh background nhiều

# ── Detect params ─────────────────────────────────────────────
CONF_THR    = 0.60     # was 0.80 — hạ để bắt thêm 1 ảnh no_face
MIN_FACE_PX = 30      # was 40
DET_MIN_SIZE = 15     # was 20 — detect mặt nhỏ hơn

# ── Align params (siết để mặt lấp khung, kèm oval mask) ──────
EYE_DIST_RATIO = 0.42  # was 0.32 — mặt to hơn ~31%, bớt background
EYE_Y_RATIO    = 0.40  # was 0.38 — bớt trán/tóc
MIN_VALID      = 0.55  # was 0.50 — siết → có thể chạm biên ảnh gốc
SKIP_EXISTING  = False  # tạm tắt để chạy lại toàn bộ

# ── Background mask params ───────────────────────────────────
MASK_BG_VALUE  = 128   # mid-gray: LBP/Sobel ít nhiễu, đồng nhất giữa ảnh
MASK_FEATHER   = 10    # blur viền mask cho mượt
MASK_AXES_RX   = 0.42  # bán trục ngang / W
MASK_AXES_RY   = 0.55  # bán trục dọc / H
MASK_CY_RATIO  = 0.55  # tâm ellipse Y / H (hơi xuống dưới mắt)
USE_OVAL_MASK  = False  # bật lại để xóa background, tránh SVM học theo nền

# ── Debug ────────────────────────────────────────────────────
SAVE_DEBUG_SAMPLES = 30  # số ảnh ngẫu nhiên copy ra _debug_samples/

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _build_oval_mask(out_size, feather=MASK_FEATHER):
    """Mask hình oval mềm (1 ở vùng mặt, 0 ở góc)."""
    W, H = out_size
    mask = np.zeros((H, W), dtype=np.float32)
    cv2.ellipse(
        mask,
        center=(W // 2, int(H * MASK_CY_RATIO)),
        axes=(int(W * MASK_AXES_RX), int(H * MASK_AXES_RY)),
        angle=0, startAngle=0, endAngle=360,
        color=1.0, thickness=-1,
    )
    if feather > 0:
        k = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask[:, :, None]   # (H, W, 1) broadcast 3 channels


_OVAL_MASK = _build_oval_mask(OUTPUT_SIZE)


def load_finetuned_mtcnn():
    """Load MTCNN từ facenet-pytorch, thay thế weights bằng bản fine-tuned."""
    detector = FacenetMTCNN(
        min_face_size=DET_MIN_SIZE,
        thresholds=[0.5, 0.6, 0.70],  # nới lỏng cascade thresholds
        keep_all=True,
        post_process=False,
        device=DEVICE,
    )

    weights_dir = Path(WEIGHTS_DIR)

    # Load fine-tuned PNet weights
    pnet_path = weights_dir / "pnet.pt"
    if pnet_path.exists():
        state = torch.load(str(pnet_path), map_location=DEVICE)
        # facenet-pytorch PNet có thêm softmax4_1 — skip nếu không có trong saved
        try:
            detector.pnet.load_state_dict(state, strict=False)
            print(f"  ✅ Loaded fine-tuned PNet from {pnet_path}")
        except Exception as e:
            print(f"  ⚠️ PNet load failed: {e}")
    else:
        print(f"  ⚠️ PNet weights not found: {pnet_path} — dùng pretrained")

    # Load fine-tuned RNet weights
    rnet_path = weights_dir / "rnet.pt"
    if rnet_path.exists():
        state = torch.load(str(rnet_path), map_location=DEVICE)
        try:
            detector.rnet.load_state_dict(state, strict=False)
            print(f"  ✅ Loaded fine-tuned RNet from {rnet_path}")
        except Exception as e:
            print(f"  ⚠️ RNet load failed: {e}")
    else:
        print(f"  ⚠️ RNet weights not found: {rnet_path} — dùng pretrained")

    # Load fine-tuned ONet weights
    onet_path = weights_dir / "onet.pt"
    if onet_path.exists():
        state = torch.load(str(onet_path), map_location=DEVICE)
        try:
            detector.onet.load_state_dict(state, strict=False)
            print(f"  ✅ Loaded fine-tuned ONet from {onet_path}")
        except Exception as e:
            print(f"  ⚠️ ONet load failed: {e}")
    else:
        print(f"  ⚠️ ONet weights not found: {onet_path} — dùng pretrained")

    return detector


def align_and_crop(img_rgb, box, landmarks, out_size=OUTPUT_SIZE):
    """Align và crop mặt, giống pipeline gốc."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < MIN_FACE_PX or h < MIN_FACE_PX:
        return None

    left_eye   = np.float32(landmarks[0])
    right_eye  = np.float32(landmarks[1])
    dX         = right_eye[0] - left_eye[0]
    dY         = right_eye[1] - left_eye[1]
    eye_center = ((left_eye[0]+right_eye[0])/2, (left_eye[1]+right_eye[1])/2)
    scale      = (EYE_DIST_RATIO * out_size[0]) / (np.sqrt(dX**2+dY**2) + 1e-6)
    M          = cv2.getRotationMatrix2D(eye_center, np.degrees(np.arctan2(dY, dX)), scale)
    M[0, 2]   += out_size[0] * 0.50 - eye_center[0]
    M[1, 2]   += out_size[1] * EYE_Y_RATIO - eye_center[1]

    H_src, W_src = img_rgb.shape[:2]
    mask = cv2.warpAffine(
        np.ones((H_src, W_src), dtype=np.uint8),
        M, out_size, flags=cv2.INTER_NEAREST, borderValue=0
    )
    if mask.mean() < MIN_VALID:
        return None

    aligned = cv2.warpAffine(img_rgb, M, out_size,
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    if USE_OVAL_MASK:
        bg = np.full_like(aligned, MASK_BG_VALUE)
        aligned = (aligned.astype(np.float32) * _OVAL_MASK
                   + bg.astype(np.float32) * (1.0 - _OVAL_MASK)).astype(np.uint8)

    return cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)


def process_frame(img_bgr, out_dir, stem, detector):
    """Process 1 ảnh: detect → align → crop → save."""
    dst = out_dir / f"{stem}.jpg"
    if SKIP_EXISTING and dst.exists():
        return "existing"
    if img_bgr is None:
        return "bad_read"
    H, W = img_bgr.shape[:2]
    if H < 40 or W < 40:
        return "too_small"

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = detector.detect(img_rgb, landmarks=True)

    if boxes is None:
        return "no_face"

    best_idx = int(np.argmax(probs))
    if probs[best_idx] < CONF_THR:
        return "low_conf"

    crop = align_and_crop(img_rgb, boxes[best_idx], landmarks[best_idx])
    if crop is None:
        return "bad_crop"

    cv2.imwrite(str(dst), crop)
    return "saved"


def main():
    print("Loading fine-tuned MTCNN...")
    detector = load_finetuned_mtcnn()

    grand_saved = grand_existing = 0
    grand_skip = {}

    for split_name in ("train", "val", "test"):
        split_root = Path(SPLIT_ROOT) / split_name
        if not split_root.exists():
            print(f"  ⚠️  Không tìm thấy {split_root}, bỏ qua")
            continue
        print(f"\n{'━'*52} {split_name.upper()}")

        for person_dir in sorted(split_root.iterdir()):
            if not person_dir.is_dir():
                continue
            out_dir = Path(CROP_ROOT) / split_name / person_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            imgs = [f for f in person_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
            counts = {"saved": 0, "existing": 0}
            p_skips = {}

            def _tally(r):
                if r in counts:
                    counts[r] += 1
                else:
                    p_skips[r] = p_skips.get(r, 0) + 1

            for fp in tqdm(imgs, desc=f"  {split_name}/{person_dir.name}", leave=False):
                _tally(process_frame(cv2.imread(str(fp)), out_dir, fp.stem, detector))

            grand_saved    += counts["saved"]
            grand_existing += counts["existing"]
            for k, v in p_skips.items():
                grand_skip[k] = grand_skip.get(k, 0) + v

            skip_str  = "  ".join(f"{k}={v}" for k, v in p_skips.items()) or "—"
            exist_str = f"  |  {counts['existing']} existed" if counts["existing"] else ""
            print(f"  📁 {split_name}/{person_dir.name}: "
                  f"{counts['saved']} saved{exist_str}  |  skip: {skip_str}")

    # ── Tổng kết ─────────────────────────────────────────────
    print(f"\n{'═'*52}")
    print(f"    Mới saved    : {grand_saved}")
    if grand_existing:
        print(f"    Đã có sẵn   : {grand_existing}  (bỏ qua)")
    if grand_skip:
        print("  ❌  Skipped:")
        labels = {
            "no_face": "không detect", "low_conf": "confidence thấp",
            "bad_crop": "crop kém (mặt cắt/quá xa)", "too_small": "ảnh nhỏ",
            "bad_read": "lỗi đọc",
        }
        for k, v in sorted(grand_skip.items(), key=lambda x: -x[1]):
            print(f"     • {labels.get(k, k)}: {v}")
    print(f"{'═'*52}")

    print("\n📊 Phân bố sau crop:")
    for split_name in ("train", "val", "test"):
        split_root = Path(CROP_ROOT) / split_name
        if not split_root.exists():
            continue
        print(f"  [{split_name}]")
        for d in sorted(split_root.iterdir()):
            if d.is_dir():
                n = len(list(d.glob("*.jpg")))
                print(f"    {d.name:12s}: {n:>4} ảnh")
    print(f"\n  Aligned crops: {CROP_ROOT}")

    # ── Debug samples (random N ảnh từ train) ────────────────
    if SAVE_DEBUG_SAMPLES > 0:
        train_root = Path(CROP_ROOT) / "train"
        if train_root.exists():
            all_jpgs = list(train_root.rglob("*.jpg"))
            if all_jpgs:
                rng = random.Random(42)
                picks = rng.sample(all_jpgs,
                                   min(SAVE_DEBUG_SAMPLES, len(all_jpgs)))
                debug_dir = Path(CROP_ROOT) / "_debug_samples"
                debug_dir.mkdir(parents=True, exist_ok=True)
                for i, src in enumerate(picks):
                    rel = src.relative_to(train_root)
                    out_name = f"{i:03d}_{str(rel).replace(os.sep, '_')}"
                    img = cv2.imread(str(src))
                    if img is not None:
                        cv2.imwrite(str(debug_dir / out_name), img)
                print(f"  Debug samples: {debug_dir} ({len(picks)} ảnh)")


if __name__ == "__main__":
    main()
