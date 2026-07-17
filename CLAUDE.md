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
| 1 | Data, preprocessing, baseline CRNN | code-complete; awaiting full run + report numbers |
| 2 | Advanced arch (STN-CRNN) + augmentation | code-complete; awaiting full run + report numbers |
| 3 | Fine-tune, test-set eval, final report + slides | not started |

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
- Each milestone has a `smoke_test*.yaml` for a 2–3 min `QUICK_TEST=True` sanity run.

## What I still need from the user (pending)
These fill the `[FILL IN]` fields in the reports and produce the weights to submit:

1. **M1 cell [9] metrics** — paste `Best model from epoch...`, `Best CER`, `Best WER`
   → fills `reports/milestone1.md` best-CER/WER rows and the M2 comparison table baseline.
2. **M2 full run** — run `notebooks/milestone2_advanced.ipynb` with `QUICK_TEST=False`;
   paste its final metrics (best CER/WER, mean/median/perfect) → fills `reports/milestone2.md`.
3. **Weights** — download `checkpoints/best_model.pt` (M1) and `checkpoints_m2/best_model.pt` (M2) from Colab.

## Conventions
- Configs drive everything; don't hardcode. Add a flag + a `smoke_test*` variant for new features.
- Reports use `[FILL IN]` placeholders until real run numbers exist — never invent numbers.
- Colab notebooks: cell to clone repo + `pip install -r requirements.txt`, then `QUICK_TEST` toggle.
