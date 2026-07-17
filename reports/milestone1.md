# Milestone 1 — Data, Preprocessing & Baseline CRNN

> Numbers from the full run (`QUICK_TEST = False`) of `notebooks/milestone1_baseline.ipynb`.

## 1. Dataset

We use the **IAM word-level handwriting** dataset (`priyank-m/IAM_words_text_recognition` on HuggingFace) as the baseline data source.

| Split | Samples |
|-------|---------|
| Train | 69,200 |
| Val   | 23,100 |
| Test  | 23,100 |

- **Task:** given an image of a single handwritten word, output the text.
- **Label stats (train):** avg length **4.1** chars, max length **53** chars.
- **Character set:** lowercase letters, digits, and common punctuation (see `ALPHABET` in `src/dataset.py`).

> **Important caveat / limitation:** IAM is **adult** handwriting. The project goal is *children's* handwriting, which has far higher variability (inconsistent letter sizes, skewed baselines, reversed/mirrored letters). This milestone therefore reports a **transfer baseline** on adult data; sourcing a children's handwriting dataset is the primary open task for Milestones 2–3.

## 2. Preprocessing

Pipeline in `src/preprocessing.py` (see Section 4 for before/after figures):

1. **Grayscale** conversion.
2. **Adaptive thresholding** (Gaussian, block 11, C=2) → binarization robust to uneven lighting.
3. **Median blur** (3×3) → removes speckle noise.
4. **Resize** to height 32 preserving aspect ratio, capped/padded to width 128.
5. **Normalize** to [-1, 1].

## 3. Baseline Model — CRNN

Architecture (`src/model.py`): **CNN feature extractor → BiLSTM → CTC**.

- 7-layer CNN (channels 64→128→256→256→512→512→512) reducing feature-map height to 1.
- 2-layer bidirectional LSTM, hidden size 256.
- Linear head over **53** classes (alphabet of 52 + CTC blank).
- Loss: **CTC** (blank=0). Optimizer: Adam, lr 1e-3, StepLR. Early stopping on val CER (patience 10).
- Total parameters: **8,208,181**.

## 4. Baseline Results

Full run: 50 epochs configured; best model at **epoch 21**, early-stopped at epoch 31 (patience 10).

| Metric | Value |
|--------|-------|
| Best Val CER | **0.1065** |
| Best Val WER | **0.2460** |
| Mean CER (val) | 0.1066 |
| Median CER (val) | 0.0000 |
| Perfect (CER=0) | 17,387 / 23,064 (75.4%) |

Training curves: see Section 7 (loss + CER vs epoch).

## 5. Error Analysis

From the worst-20 predictions (Section 8), the highest-CER cases are almost
entirely **single-character / punctuation targets** (`a`, `.`, `,`, `(`, `?`, `#`).
This is a **CER-metric artifact**, not a real weakness: CER = edit distance ÷
ground-truth length, so predicting a multi-char word for a 1-char target yields a
huge ratio (e.g. `morality` for `a` → CER 7.0). On actual words the model is
strong — 75.4% exact match and median CER 0. Qualitatively (Section 7 sample
predictions) it reads cursive words like `knowing`, `course`, `could` correctly;
the genuine failures are heavily degraded/noisy crops (e.g. `exert`→`ehert`,
`inequity`→`reregity`).

**Takeaways for Milestone 2:** the remaining real errors are driven by geometric
variability and noise (slant, uneven strokes, degraded crops) — exactly what the
STN rectifier and children's-handwriting augmentation target. Punctuation-crop
CER inflation can be reported separately (CER on words ≥2 chars).

## 6. Deliverables Checklist (per project plan)

- [x] Colab/Jupyter notebook showing the data pipeline — `notebooks/milestone1_baseline.ipynb`
- [ ] Trained baseline model — `checkpoints/best_model.pt` (from full run)
- [ ] This report with dataset, distribution, and baseline CER
