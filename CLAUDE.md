# CLAUDE.md — Children's Handwriting OCR (university mini-project)

## What this project is
Deep-learning OCR for children's handwriting. University mini-project —
**minimal scope on purpose**: the goal is to check every deliverable in the work
plan (`docs/תוכנית עבודה - זיהוי כתב יד של ילדים.pdf`), not to maximize quality.
Prefer the smallest change that satisfies the plan; reuse existing code; don't add
datasets or architectures beyond what the plan names.

Runs in **Google Colab** (T4 GPU). ML deps (torch/numpy/cv2) are NOT installed
locally — local checks are syntax-only; real verification is a Colab run.

## Milestones
| # | Scope | Status |
|---|-------|--------|
| 1 | Data, preprocessing, baseline CRNN | DONE — CER 0.1065 / WER 0.2460 (val); submitted as notebook + report |
| 2 | Advanced arch (STN-CRNN) + augmentation | DONE — CER 0.0867 / WER 0.2012 (val, −18.6%); notebook + report |
| 3 | Fine-tune, test-set eval, inference pipeline, report + slides | DONE — final test CER 0.0969 / WER 0.2215 (77.9% perfect); report + slides filled |

## Submission snapshots
The authoritative checklist is `docs/SUBMISSIONS.md`. When the user asks "what do
I submit for M1 / M2 / M3?", read that file and give them the matching bundle.
Each milestone is a self-contained bundle; the pattern is always
**notebook/code + trained weights + report**:

- **M1:** `notebooks/milestone1_baseline.ipynb` (with outputs) + `checkpoints/best_model.pt` + `reports/milestone1.md`
- **M2:** `src/model.py` (STN) + `src/augmentation.py` + `checkpoints_m2/best_model.pt` + `reports/milestone2.md` (+ M2 notebook with outputs)
- **M3:** inference pipeline + `reports/final_report.md` + presentation (built during M3)

## Architecture (M1 → M2)
- **M1 baseline:** CRNN = `CNN → BiLSTM → CTC` (`src/model.py`, `configs/baseline_crnn.yaml`).
  ~8.2M params, 53 classes. Full-run result: mean CER 0.1066, 75% exact-word (val).
- **M2:** adds `STN` front-end (`use_stn` flag, identity-init) → `STN → CNN → BiLSTM → CTC`,
  plus `ChildrenAugmentation` (affine/elastic/baseline-wave, train split only via
  `data.augment`). Reuses `src/train.py` unchanged. `configs/milestone2_stn.yaml`.
- **M3:** fine-tunes M2 via `init_from` in `src/train.py` (`configs/milestone3_finetune.yaml`,
  lr 1e-4). Adds `src/evaluate.py` (test-split scoring) and `src/inference.py`
  (checkpoint→image→text, runnable as `python -m src.inference`). Notebook
  `notebooks/milestone3_final.ipynb` regenerates the M2 base if absent, fine-tunes,
  evaluates on test, demos inference. Report `reports/final_report.md` + slides
  `reports/presentation.md`.
- Each milestone has a `smoke_test*.yaml` for a 2–3 min `QUICK_TEST=True` sanity run.

## Status
All three milestones DONE and pushed to `main`. Submitted as notebook + report
per milestone (weights not submitted, by decision). Final test CER 0.0969.
Tags `milestone-1`/`milestone-2`/`milestone-3` freeze each submission state.

Final numbers of record:
- M1 baseline: val CER 0.1065 / WER 0.2460
- M2 STN-CRNN + aug: val CER 0.0867 / WER 0.2012; test CER 0.0981 / WER 0.2234
- M3 fine-tuned: val CER 0.0903 / WER 0.2099; **test CER 0.0969 / WER 0.2215, 77.9% perfect**

## Conventions
- Configs drive everything; don't hardcode. Add a flag + a `smoke_test*` variant for new features.
- Reports use `[FILL IN]` placeholders until real run numbers exist — never invent numbers.
- Colab notebooks: cell to clone repo + `pip install -r requirements.txt`, then `QUICK_TEST` toggle.
