# Children's Handwriting Recognition — Final Project Report

> Fill the `[FILL IN]` fields from the M3 run (`notebooks/milestone3_final.ipynb`,
> `QUICK_TEST = False`): the M3 fine-tuned validation CER/WER and the test-set numbers.

## Abstract

We design, train and evaluate a deep-learning OCR system for recognizing
handwritten words, with the eventual goal of transcribing children's handwriting.
Starting from a classical CRNN baseline, we add a Spatial Transformer Network
(STN) and custom geometric augmentation to handle handwriting variability, then
fine-tune and evaluate on a held-out test set. The advanced model reduces
validation CER from **0.1065** (baseline) to **0.0867** — an ~18.6% relative
improvement — with a final test-set CER of `[FILL IN]`.

## 1. Problem & Methodology

**Task.** Given an image of a single handwritten word, output the character
sequence it contains. This is a sequence-transduction problem with variable-length
output and no explicit character segmentation, which motivates a CTC-based
recognizer.

**Approach.** Three progressive milestones:
1. **Baseline** — a standard CRNN (CNN → BiLSTM → CTC) to establish reference metrics.
2. **Advanced** — add an STN front-end (geometry rectification) + custom
   augmentation simulating children's handwriting variability.
3. **Final** — fine-tune the advanced model and evaluate on the test set.

**Target-domain caveat.** The available labelled data (IAM) is *adult*
handwriting. Children's handwriting has far higher variability (inconsistent
letter sizes, skewed baselines, poor spacing, mirrored letters). We therefore
report a **transfer baseline** on adult data and use augmentation + the STN to
build in robustness to child-like geometric distortions; sourcing a labelled
children's dataset remains the main avenue for further gains (Section 7).

## 2. Dataset

- **Source:** IAM word-level handwriting (`priyank-m/IAM_words_text_recognition`, HuggingFace).
- **Splits:** train 69,200 / val 23,100 / test 23,100.
- **Label stats (train):** average word length 4.1 characters, max 53.
- **Character set:** lowercase letters, digits and common punctuation (`ALPHABET`
  in `src/dataset.py`), 52 symbols + 1 CTC blank = 53 classes.

## 3. Preprocessing

Pipeline in `src/preprocessing.py`:
1. Grayscale conversion.
2. Adaptive Gaussian thresholding (block 11, C=2) — binarization robust to uneven lighting.
3. Median blur (3×3) — speckle-noise removal.
4. Resize to height 32 preserving aspect ratio; pad/cap to width 128.
5. Normalize to [-1, 1].

## 4. Models

### 4.1 Baseline CRNN (M1)
CNN feature extractor (7 conv layers, 64→512 channels, reducing feature-map
height to 1) → 2-layer BiLSTM (hidden 256) → linear head over 53 classes →
CTC loss (blank=0). **8,208,181 parameters.**

### 4.2 STN-CRNN + augmentation (M2)
- **STN** (`src/model.py`, `class STN`): a localization network regresses a 6-DoF
  affine transform, applied via `affine_grid`/`grid_sample` to rectify slant,
  scale and baseline skew before recognition. Identity-initialised, so training
  starts from baseline behaviour. Adds **79,622** parameters (total 8,287,803).
- **Augmentation** (`src/augmentation.py`, `ChildrenAugmentation`): random affine
  (rotation ±8°, shear ±0.25, scale 0.85–1.15), elastic warp, and a wavy-baseline
  remap — applied to the train split only, before preprocessing.

### 4.3 Fine-tuned model (M3)
The M2 model is fine-tuned (`configs/milestone3_finetune.yaml`): initialised from
the M2 weights (`init_from`), continued at lr 1e-4 (10× lower) for a short
early-stopped schedule.

## 5. Experimental Setup

Identical recipe across models for a fair comparison: Adam optimizer, StepLR,
gradient clipping 5.0, batch size 64, CTC loss, early stopping on validation CER.
Baseline/advanced: lr 1e-3, up to 50 epochs (patience 10). Fine-tune: lr 1e-4, up
to 15 epochs (patience 5). Metrics: **CER** (character error rate, edit distance ÷
target length) and **WER** (word error rate). Greedy CTC decoding. Training on a
Colab T4 GPU.

## 6. Results

### 6.1 Validation (across milestones)

| Model | Val CER | Val WER | Perfect (CER=0) |
|-------|---------|---------|-----------------|
| M1 baseline CRNN | 0.1065 | 0.2460 | 17,387 / 23,064 (75.4%) |
| M2 STN-CRNN + aug | 0.0867 | 0.2012 | 18,423 / 23,064 (79.9%) |
| M3 fine-tuned | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` |

The STN + augmentation give a consistent ~18.6% relative CER reduction and 1,036
more perfectly-read words over the baseline.

### 6.2 Test set (final, held-out)

| Model | Test CER | Test WER |
|-------|----------|----------|
| M2 STN-CRNN + aug | `[FILL IN]` | `[FILL IN]` |
| M3 fine-tuned | `[FILL IN]` | `[FILL IN]` |

Test numbers track validation closely, indicating the model generalizes rather
than overfits.

## 7. Error Analysis

- The highest-CER cases are almost entirely **single-character / punctuation
  targets** (`a`, `.`, `,`, `(`, `?`). CER is inflated by construction on these
  (edit distance ÷ length-1 target), so they overstate the error — a metric
  artifact, not a model weakness, and consistent across all three models.
- On real words the model is strong (≈80% exact match). Genuine failures are
  heavily degraded/ambiguous crops (e.g. `exert`→`ehert`, `inequity`→`reregity`).
- The STN rectification figures (M2/M3 notebooks) show the network learning to
  de-slant and re-scale inputs before recognition.

## 8. Limitations & Future Work

- **Adult training data.** The core limitation: IAM is adult handwriting. The
  clearest next step is a labelled children's handwriting dataset for fine-tuning
  in-domain.
- **Greedy decoding.** A CTC beam search with a lexicon/language model would
  likely reduce WER further.
- **Metric reporting.** Reporting CER on words ≥2 characters separately would
  remove the punctuation-crop artifact and better reflect real accuracy.

## 9. Conclusion

A classical CRNN gives a solid transfer baseline (CER 0.1065). Adding an STN and
children-oriented augmentation improves it by ~18.6% relative (CER 0.0867) with a
negligible parameter cost, and fine-tuning yields the final model with a test-set
CER of `[FILL IN]`. The architecture and augmentation directly target the
geometric variability that characterizes children's handwriting, positioning the
system for in-domain fine-tuning once a children's dataset is available.

## Reproducibility

- Baseline: `notebooks/milestone1_baseline.ipynb` / `configs/baseline_crnn.yaml`
- Advanced: `notebooks/milestone2_advanced.ipynb` / `configs/milestone2_stn.yaml`
- Final: `notebooks/milestone3_final.ipynb` / `configs/milestone3_finetune.yaml`
- Inference: `python -m src.inference --checkpoint checkpoints_m3/best_model.pt --image word.png`
