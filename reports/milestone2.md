# Milestone 2 — Advanced Architecture (STN-CRNN) & Data Augmentation

> Fill the `[FILL IN]` fields from the notebook run (`notebooks/milestone2_advanced.ipynb`).
> Numbers come from the **full** run (`QUICK_TEST = False`), not the quick test.

## 1. Goal

Upgrade the Milestone 1 baseline to handle the geometric variability of
children's handwriting (project plan, task 2): inconsistent letter sizes,
slanted words, skewed baselines and uneven spacing. Two additions, both toggled
by config so the comparison against the baseline is fair:

1. A **Spatial Transformer Network (STN)** front-end.
2. **Custom data augmentation** that simulates children's distortions.

Everything else (dataset, preprocessing, training loop, optimizer, metrics) is
identical to Milestone 1.

## 2. Advanced Architecture — STN-CRNN

New architecture (`src/model.py`): **STN → CNN → BiLSTM → CTC**.

- **STN** (`class STN`): a small localization network (2 conv blocks →
  adaptive pool → 2 FC layers) regresses 6 affine parameters; the image is
  resampled with `affine_grid` + `grid_sample`. It learns to rectify slant,
  scale and baseline skew *before* the CRNN reads the image.
- Initialised to the **identity transform**, so training starts from baseline
  behaviour and only learns corrections it needs — no risk of degrading the
  strong M1 starting point.
- The CRNN body is unchanged from Milestone 1.
- Total parameters: `[FILL IN]` (STN adds `[FILL IN]`; from Section 4).

## 3. Custom Data Augmentation

Pipeline in `src/augmentation.py` (`class ChildrenAugmentation`), applied to the
**train split only**, before preprocessing. Each transform maps the axes on
which children's handwriting varies:

| Transform | Simulates |
|-----------|-----------|
| Random affine (rotation ±8°, shear ±0.25, scale 0.85–1.15) | slant, skewed baseline, inconsistent letter size |
| Elastic warp (Gaussian displacement field) | wobbly, non-uniform strokes |
| Baseline wave (column-wise vertical sinusoid) | wavy, non-flat baseline |

Each is applied independently with probability `p` (default 0.5). See Section 3
of the notebook for before/after figures.

## 4. Results — M2 vs. Baseline

Full run: `[FILL IN]` epochs (early-stopped at epoch `[FILL IN]`).

| Metric (validation) | Baseline (M1) | STN-CRNN + aug (M2) |
|---------------------|---------------|---------------------|
| Best Val CER | 0.1065 | `[FILL IN]` |
| Best Val WER | 0.2460 | `[FILL IN]` |
| Mean CER | 0.1066 | `[FILL IN]` |
| Median CER | 0.0000 | `[FILL IN]` |
| Perfect (CER=0) | 17,387 / 23,064 | `[FILL IN]` |

Training curves and the STN rectification visualization (input vs. rectified
image): see Section 6 of the notebook.

> **Note on augmentation + adult data:** augmentation makes the *train* task
> harder while the *val* set stays clean adult handwriting, so a small CER
> change on IAM is expected. The point is robustness to child-like distortions,
> which the STN rectification figures and hard-example analysis demonstrate
> directly. True gains require the children's dataset (Milestone 3).

## 5. Error Analysis

Worst-20 predictions (Section 7 of the notebook):

- `[FILL IN — how the worst cases differ from the M1 baseline]`

As in M1, the highest-CER cases are dominated by single-character / punctuation
targets, where CER (edit distance ÷ ground-truth length) is inflated by
construction rather than reflecting a real weakness.

**Hard examples the model handles vs. fails on:** `[FILL IN — cite 2–3 examples
from the sample-predictions and STN-rectification figures]`.

## 6. Deliverables Checklist (per project plan)

- [x] Advanced architecture code — STN front-end in `src/model.py`, augmentation in `src/augmentation.py`
- [x] Colab/Jupyter notebook — `notebooks/milestone2_advanced.ipynb`
- [ ] Updated model weights — `checkpoints_m2/best_model.pt` (from full run)
- [ ] Interim report with baseline comparison + error analysis (this file, once `[FILL IN]` complete)
