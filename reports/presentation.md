# Presentation — Children's Handwriting Recognition

Slide-by-slide outline for the final course presentation. Each `---` is one slide.
Renders as a deck with Marp/reveal.js, or copy into PowerPoint/Google Slides.
Fill `[FILL IN]` from the M3 run.

---

## Children's Handwriting Recognition (OCR)

Deep-learning OCR: handwritten word image → text

Course image-processing project · [names]

---

## The Problem

- Transcribe handwritten words to digital text
- Target: **children's** handwriting — high variability
  - inconsistent letter sizes, slanted/skewed baselines, poor spacing, mirrored letters
- Variable-length output, no character segmentation → **CTC** recognizer

---

## Data & Preprocessing

- **IAM word-level** dataset: 69k train / 23k val / 23k test
- Caveat: adult handwriting → we report a **transfer baseline**
- Preprocessing: grayscale → adaptive threshold → denoise → resize 32×128 → normalize

---

## Approach — 3 Milestones

1. **Baseline** — classical CRNN (CNN → BiLSTM → CTC)
2. **Advanced** — + Spatial Transformer Network + custom augmentation
3. **Final** — fine-tune + test-set evaluation + inference pipeline

---

## M1 — Baseline CRNN

- CNN → BiLSTM → CTC, 8.2M params, 53 classes
- Establishes reference metrics
- **Val CER 0.1065 · WER 0.2460 · 75.4% words perfect**

---

## M2 — Handling Variability

- **STN** front-end: learns an affine warp to rectify slant/scale/baseline
  (identity-init, +80k params only)
- **Augmentation** (train only): rotation, shear, elastic warp, wavy baseline —
  simulates child-like distortions
- Show: augmentation before/after + STN input-vs-rectified figures

---

## M2 — Results

| | Val CER | Val WER | Perfect |
|---|---|---|---|
| M1 baseline | 0.1065 | 0.2460 | 75.4% |
| **M2 STN+aug** | **0.0867** | **0.2012** | **79.9%** |

**~18.6% relative CER reduction**, +1,036 words read perfectly

---

## M3 — Fine-Tuning & Test Evaluation

- Fine-tune from M2 weights at lr 1e-4 (short, early-stopped)
- First evaluation on the **held-out test set**
- **Test CER `[FILL IN]` · WER `[FILL IN]`**

---

## Inference Pipeline

- `python -m src.inference --checkpoint checkpoints_m3/best_model.pt --image word.png`
- Live demo: sample test images → predictions
- (show 4–6 GT vs Pred examples)

---

## Error Analysis

- Worst CER = single-char / punctuation crops → **metric artifact** (edit distance ÷ length-1)
- Real words ≈ 80% exact
- Genuine failures: degraded / ambiguous crops

---

## Conclusions & Future Work

- STN + augmentation improve a strong baseline by ~18.6% at negligible cost
- Architecture targets the geometric variability of children's handwriting
- **Next:** labelled children's dataset, CTC beam search + language model

---

## Thank You

Questions?

Repo: github.com/yotamc19/Image-processing-project
