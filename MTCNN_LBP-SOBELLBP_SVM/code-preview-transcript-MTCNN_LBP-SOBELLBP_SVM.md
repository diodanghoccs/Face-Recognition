# Code Preview Transcript — MTCNN_LBP-SOBELLBP_SVM

**Project:** `C:\Users\diosodumb\Documents\Final_CS231\MTCNN_LBP-SOBELLBP_SVM`
**Date:** 2026-05-18
**User question (VN):** *"Xem qua pipeline này, giải thích lí do cài đặt config như thế, đánh giá"*
**Classification:** AI/ML project — classical ML (handcrafted features + SVM with PyTorch MTCNN frontend)

---

## Analysis Brief

**Files in scope:**
- `feature_utils.py` — Multi-scale LBP + Sobel-LBP feature extractor
- `mtcnn_finetuned_crop.py` — Fine-tuned MTCNN face detection + alignment + crop
- `train_svm.py` — Augment, scale, PCA-whiten, SVM GridSearchCV, Platt calibration with FrozenEstimator
- `calibrate_thresholds.py` — Pick open-set thresholds (T_p, T_m) on val + per-class boost

**Out of scope:** `../config.py` (parent folder, outside connected workspace) — only attribute names `cfg.LBP_*` are visible from imports.

**Frameworks:** PyTorch (facenet-pytorch), scikit-learn ≥ 1.6 (`sklearn.frozen.FrozenEstimator`), scikit-image, OpenCV, joblib, matplotlib, seaborn, tqdm.

**Artifacts in `face_recognition_model/`:** svm_face.pkl, scaler.pkl, pca.pkl, label_encoder.pkl, thresholds.pkl, train_cache.npz, confusion_matrix.png, threshold_histograms.png, threshold_2d_scatter.png — plus an orphan `knn.pkl` with no training script in this folder.

---

## Anonymization Map (revealed)

| Anonymized | True identity |
|---|---|
| Analysis A | The Pipeline Tracer |
| Analysis B | The Environment Cartographer |
| Analysis C | The Architect |
| Analysis D | The Model Anatomist |
| Analysis E | The Entry Point Navigator |

---

## Analyst Responses

### The Architect

Pipeline có 4 module rõ ràng tương ứng 4 file. Tách biệt train/eval/inference tốt: aug chỉ trên train (train_svm.py L154–158), không lạm val/test; calibrate_thresholds.py dùng val cho Platt sigmoid; artifacts pickle/npz đầy đủ. **Không có script inference/webcam trong thư mục** — phải load .pkl ngoài, lỗ hổng kiến trúc cuối.

Config 2 lớp: (1) hardcoded inline (CELL=16, STEP=8, CLAHE clipLimit=2.0 ở feature_utils.py L12–19; align params EYE_DIST_RATIO=0.42, EYE_Y_RATIO=0.40 lặp ở 3 file — mtcnn_finetuned_crop.py L28–29, calibrate_thresholds.py L31–32); (2) tập trung ở ../config.py. Lỗi coupling: train_svm.py dùng `sys.path.insert` động (L27–28), bố trí kỳ lạ; hardcoded path trong mtcnn_finetuned_crop.py không qua config.

ML patterns: ✅ Tốt: aug chỉ train, scaler fit train, class_weight per-class (L114), hard-neg mining (tắt), GridSearchCV cv=5, Platt trên val.
❌ Anti-pattern: (a) extract_features() tính lại 5 lần cho augmented (L157) — lãng phí, nên cache aug idx; (b) PCA_DIM=368 hardcoded (L34), PCA analysis in-flight tính nhưng kết quả không dùng để chọn dim; (c) PER_CLASS_BOOST cứng trong dict (calibrate_thresholds.py L51–57); (d) MTCNN weight loading 3 chỗ — DRY violation (mtcnn_finetuned_crop.py L68–117 vs calibrate_thresholds.py L137–143).

Khuyến nghị: viết inference pipeline, gộp config, cache features cho aug, tách MTCNN weight loader thành utility.

### The Pipeline Tracer

**Stage 1 (MTCNN Crop + Align)** — config siết để bớt background:
- `OUTPUT_SIZE=(128,128)` (L20): hạ từ (160,160), comment ghi "tránh background nhiều"
- `EYE_DIST_RATIO=0.42` (L28): tăng từ 0.32, mặt to hơn ~31%
- `EYE_Y_RATIO=0.40` (L29): bớt trán/tóc
- `CONF_THR=0.60` (L23): hạ từ 0.80
- `DET_MIN_SIZE=15`, `MIN_VALID=0.55`
- `MASK_BG_VALUE=128` mid-gray (LBP ít nhiễu), `USE_OVAL_MASK=False` đang tắt

**Stage 2 (Feature)** — 10800-d:
- IMG_SIZE 128, CELL 16, STEP 8 → 15×15=225 cell
- LBP P=8 R=1 (10 bins) + P=16 R=2 (18 bins) + Sobel-LBP_x,y (P=8 R=1 mỗi, 10 bins)
- CV_16S signed để tránh skimage warn trên float
- L2-normalize per cell

**Stage 3 (Train)**:
- Augment 5 variants chỉ trên train (orig, flip, brightness ±, noise ±18) — controlled condition, bỏ rotation/perspective/blur
- Scaler fit train only (L169–172), PCA whiten 368 dims
- GridSearchCV cv=5, RBF + poly, probability=False
- `CalibratedClassifierCV(FrozenEstimator(svm), method="sigmoid").fit(X_val_p, y_val)` (L215–216) — FrozenEstimator khóa SVM, val không augment → **ZERO leakage**

**Stage 4 (Threshold)**:
- `T_p = max(min(percentile(p_max,3), 0.92), 0.63)` — cap 0.92, floor 0.63 (L73)
- `T_m = min(percentile(margin,5), 0.80)` (L74)
- `PER_CLASS_BOOST = {Dan:0.11, Vi:0.11, Triet:0.012, Tri:0.026}` (L51–57)
- Rule AND (L75): cả 2 signal phải qua

**Data quality**: ✓ aug train only, ✓ scaler/PCA fit train only, ✓ FrozenEstimator → val không retrain. **Cảnh báo**: val dùng 2 mục đích (calibrate + chọn threshold) — nếu tune hyperparam dựa val sẽ overfitting. Per-class boost hardcode 4 class.

### The Model Anatomist

**Feature design**: 10800-d (225 cell × (10+18+10+10) = 10800). CLAHE clipLimit=2.0 tile 8×8 bù sáng cục bộ trước LBP. Multi-scale LBP: R=1 bắt chi tiết tinh (vết, nếp), R=2 bắt pattern lớn (cấu trúc gương). Sobel-LBP (Zhao Gao Zhang 2008): LBP trên Sobel-x, Sobel-y → bắt gradient direction, mạnh khi mặt quay/đổi sáng. CV_16S (signed int16) cho LBP xử lý gradient âm. L2-normalize per-cell tạo bất biến scaling, tránh cell tương phản cao chi phối.

**SVM**: `probability=False` (L116) tiết kiệm Platt nội-fold trong GridSearch; calibrate Platt sau trên val. GridSearch RBF 42 + Poly 36 = 78 combos × cv=5 = 390 fits. PCA_DIM=368 giữ ~85–90% variance (qua elbow visual, không tune). `whiten=True` chuẩn hóa eigenvector → covariance = I, tối ưu cho RBF SVM (Euclidean distance). `class_weight HARD_WEIGHT=2.0` boost class hay nhầm lẫn. Calibrate trên VAL dùng `FrozenEstimator(svm)` để khóa SVM → tránh leakage.

**MTCNN**: thresholds=[0.5, 0.6, 0.70] (L72) nới từ default [0.7, 0.8, 0.9] để bắt mặt nhỏ/low-light. CONF_THR=0.60 hạ từ 0.80. `strict=False` (L86, L98, L110) vì facenet-pytorch có softmax4_1 không có trong checkpoint. EYE_DIST_RATIO=0.42 từ 0.32 → mặt to hơn 31%. USE_OVAL_MASK=False tạm tắt để test SVM tự học ranh giới mặt.

**Open-set rejection**: T_p = max(min(pct(p_max,3),0.92),0.63) — cap 0.92 tránh threshold quá nghiêm khi val tốt, floor 0.63 chặn unknown confident sai. PER_CLASS_BOOST: Dan/Vi +0.11 (unknown attractor), Triet/Tri +0.012/0.026. Rule AND yêu cầu cả p_max và margin cao.

**Tóm**: pipeline cân bằng tốt giữa robustness ánh sáng/quay (multi-scale LBP+Sobel), overfitting (PCA 368 + augment 5×), open-set (2-signal calibration). Tham số được chọn qua thực nghiệm có kiểm soát (balanced dataset, fixed random_state=42).

### The Environment Cartographer

**A. Inconsistency hardcoded paths vs config**: mtcnn_finetuned_crop.py L17–19 dùng `r"C:\Users\diosodumb\Pictures\..."`; train_svm.py (L31–32) và calibrate_thresholds.py (L19–21) dùng `cfg.LBP_*`. Đổi config.py không sync sang crop script. ⚠️ Medium.

**B. Thiếu requirements.txt**: `sklearn.frozen.FrozenEstimator` cần sklearn ≥ 1.6. 🔴 High.

**C. N_JOBS = cpu_count - 6**: trên máy ≤6 core → N_JOBS=1; trên 8-core → N_JOBS=2. 🟡 Medium.

**D. [BỊA SỐ] Claim EYE_DIST_RATIO khác nhau giữa 2 file (0.42 vs 0.31)** — **không đúng**, kiểm tra calibrate_thresholds.py L31 thực ra cũng 0.42. *(Chairman: loại khỏi report cuối.)*

**E. Reproducibility**: RANDOM_STATE=42 ở GridSearchCV/PCA, thiếu torch.manual_seed, np.random.seed global, deterministic flag. Augmentation noise (np.random.randint) không deterministic. 🟡 Medium.

**F. Cache side-channel train_cache.npz**: đổi PCA_DIM mà quên rerun train → cache cũ → silent shape mismatch.

**G. Augmentation 5 variants** hợp lý, không over-aug. Multi-scale LBP + Sobel-LBP (Zhao 2008) classical mạnh. L2 per-cell tốt cho SVM.

**H. GridSearchCV**: RBF 6×7=42 + Poly 2×3×3×2=36 = 78 combos × cv=5 = 390 fits trên ≥ 5N samples. Trên CPU yếu vài giờ.

**I. PER_CLASS_BOOST hardcoded** → ad-hoc, khó tái hiện.

Tổng đánh giá: feature engineering cẩn thận, train/val/test sạch không leakage. Dự kiến 85–92% accuracy.

### The Entry Point Navigator

**Bước 1: `python mtcnn_finetuned_crop.py`** — Tiền xử lý. Iterate train/val/test ở SPLIT_ROOT, detect+align+crop 128×128 → CROP_ROOT. Inputs: SPLIT_ROOT raw, WEIGHTS_DIR pt (optional fallback pretrained). Outputs: cropped jpg + 30 debug samples. Failure: bad_read/too_small(<40)/no_face/low_conf(<0.60)/bad_crop(mask.mean<0.55).

**Bước 2: `python train_svm.py`** — Train. Load 3 splits từ cfg.LBP_CROP_ROOT, aug train ×5, extract 10800-d, scale, PCA 368 whiten, GridSearchCV SVM, Platt calibrate val FrozenEstimator. Output: svm_face.pkl, scaler.pkl, pca.pkl, label_encoder.pkl, train_cache.npz, confusion_matrix.png. Cũng xoá legacy lda.pkl nếu có. ~78 combos × cv=5 = 390 fits.

**Bước 3: `python calibrate_thresholds.py`** — Threshold. Load model + train_cache.npz, compute T_p/T_m trên val signals, apply PER_CLASS_BOOST {Dan/Vi/Triet/Tri}, optional unknown scan UNKNOWN_DIR plot FAR vs TAR, save thresholds.pkl + histogram.

**Missing**:
- Không có requirements.txt / README
- Không có inference / webcam script (orphan face_recognition_model/knn.pkl trong folder gợi ý từng có)
- Không có standalone test script (eval chỉ trong train_svm.py)
- config.py ở parent folder — ngoài scope ở đây, nhưng pipeline không chạy được nếu thiếu

**Thứ tự chạy**: (1) data sẵn ở SPLIT_ROOT, (2) crop, (3) train, (4) calibrate. Không có orchestration script.

---

## Cross-Reviews

### Reviewer 1

**1. Most accurate/comprehensive: D.** Explains the *why* behind every config choice (CV_16S for signed Sobel, whiten=True making covariance=I for RBF Euclidean, FrozenEstimator preventing leakage, AND-rule rationale). A is close but more descriptive; D ties config to ML theory.

**2. Biggest gap/inaccuracy: B.** Item D is a fabricated number (it self-flags but still asserts "High risk") — hallucinated config drift undermines trust. Misframes N_JOBS=cpu_count−6 severity without checking actual core count. Conversely, B is the only one catching the sklearn ≥1.6 FrozenEstimator dependency, which is genuinely critical.

**3. All missed:** Val triple-duty (Platt + threshold + boost tuning); class imbalance / per-class sample counts; MTCNN det threshold loosening + confidence floor 0.60 creates selection bias if inference uses different thresholds; no test-set isolation audit — does augmentation leak via subject identity across splits?

**4. AI/ML violations none flagged:** Identity leakage risk (splits should be by subject, not random); Platt sigmoid on ~N val samples per class is unstable below ~50/class; PCA fit includes augmented train (augmentations inflate variance estimates); no fixed numpy/cv2 RNG seed; threshold floor 0.63 is dataset-specific.

### Reviewer 2

**1. Most accurate/comprehensive: D.** Correctly anatomizes every model choice. A close second — strong pipeline tracing and the only one that explicitly flags val-double-use risk.

**2. Biggest gap/inaccuracy: B.** Fabricates EYE_DIST_RATIO mismatch — hallucinated bug undermines credibility. Real findings (hardcoded paths, missing requirements.txt, cache shape side channel) are valid but buried under the false claim.

**3. All missed:** No held-out test split is ever audited; `confusion_matrix.png` produced but no test ROC or open-set TPR@FPR; MTCNN crop is non-deterministic across reruns yet feeds cache; class imbalance handling beyond `HARD_WEIGHT=2.0` unexamined.

**4. Unflagged violations:** Val-set triple-dipping (calibrate + threshold + boost) is textbook leakage masquerading as clean because of FrozenEstimator; PER_CLASS_BOOST tuned on the same val it's evaluated on is overfit-by-construction; no seed for cv2/MTCNN/aug RNG; pickle artifacts have no version/hash pinning; no drift monitoring or threshold-decay plan for the open-set rejector.

### Reviewer 3

**1. Most accurate/comprehensive: A.** Traces every stage with exact numerics (crop ratios, 225 cells × multi-LBP = 10800-d, PCA 368, T_p/T_m formulas) and is the only one that flags the subtle val-set double-use.

**2. Biggest gap: B.** Self-admits a fabricated config mismatch (claim (d)) and otherwise lists generic env gripes. Also misses that GridSearch with probability=False is cheap, so "many hours" claim is overstated.

**3. All missed:** No test set held out for final reporting; no class-imbalance / per-class sample counts discussed despite HARD_WEIGHT and per-class boosts; no MTCNN crop failure rate quantified — silently drops training data; no inter-stage versioning.

**4. Unflagged issues:** Augmentation before scaler fit drifts scaler statistics; PCA fit on augmented train biases whitening basis; Platt calibration on hundreds of val samples statistically unstable; no drift/OOD monitoring; non-determinism (torch/np seeds + cuDNN); no model card / eval on demographic slices.

---

## Chairman Synthesis

The HTML report at `code-preview-MTCNN_LBP-SOBELLBP_SVM.html` is the consolidated synthesis. Key conclusions:

1. **Pipeline is structurally sound** with explicit train/val/test separation and a clever use of `FrozenEstimator` to prevent Platt calibration from leaking val into the SVM weights.

2. **The biggest *real* risk** is val being used three times (Platt + threshold percentile + PER_CLASS_BOOST tuning). This makes any val-reported metric optimistic. The user should ensure the test-set evaluation in `train_svm.py:main()` is only computed once and never used to iterate on thresholds.

3. **Identity leakage** — a face-specific failure mode — was not detectable from this code alone, as splits live in `SPLIT_ROOT` (outside this folder). Worth verifying that the same person doesn't appear across train/val/test.

4. **Config is partially centralized**. The crop script `mtcnn_finetuned_crop.py` is the outlier — it uses hardcoded Windows paths instead of `cfg`. This should be normalized.

5. **Missing pieces**: requirements.txt (especially `scikit-learn>=1.6`), README, inference/webcam script, standalone evaluation script. The orphan `knn.pkl` artifact suggests a previous KNN baseline whose training code has been removed.

6. **The config choices themselves are well-justified**. Every parameter has a comment explaining what was tried before and why the current value was chosen — this is unusually disciplined for a research codebase. The cap/floor logic on T_p (max(min(...,0.92), 0.63)) is a thoughtful guard against percentile estimation noise in both directions.

7. **Reviewer B's claim of EYE_DIST_RATIO drift between files is incorrect** — both files use 0.42. The configs are merely duplicated, which is still a future drift risk but not a current bug.
