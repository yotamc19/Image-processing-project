# Submission Checklist

Exactly what to hand in for each milestone, mapped from the work plan
(`docs/תוכנית עבודה - זיהוי כתב יד של ילדים.pdf`) to files in this repo.
Each milestone is self-contained — submit only that milestone's rows.

---

## Milestone 1 — Data, Preprocessing & Baseline (due 17 May 2026)

Plan deliverables → repo artifacts:

| Plan deliverable | Submit | Status |
|------------------|--------|--------|
| Jupyter/Colab notebook showing the data pipeline | `notebooks/milestone1_baseline.ipynb` | ✅ done (code-only; run outputs captured in the report) |
| Trained baseline model | `checkpoints/best_model.pt` | ➖ not submitting (notebook + report only, by decision) |
| Short report: dataset, its distribution, baseline metrics (CER) | `reports/milestone1.md` | ✅ done (CER 0.1065 / WER 0.2460) |

**Before submitting M1:**
1. Run notebook with `QUICK_TEST = False`, keep the cell outputs.
2. Download `checkpoints/best_model.pt` from Colab.
3. Fill every `[FILL IN]` in `reports/milestone1.md` (label stats, param count,
   best/mean/median CER, WER, error-analysis takeaways).
4. Notebook + weights + report = the M1 submission.

---

## Milestone 2 — Advanced Architecture & Augmentation (due 7 Jun 2026)

Plan deliverables → repo artifacts:

| Plan deliverable | Submit | Status |
|------------------|--------|--------|
| Advanced architecture code | `src/model.py` (STN), `src/augmentation.py` | ✅ done |
| Updated model weights | `checkpoints_m2/best_model.pt` | ➖ not submitting (notebook + report only, by decision) |
| Interim report: comparison vs. baseline + error analysis with hard examples | `reports/milestone2.md` + `notebooks/milestone2_advanced.ipynb` (with outputs) | ✅ done (CER 0.0867 / WER 0.2012, −18.6% vs baseline) |

**Before submitting M2:**
1. Run `notebooks/milestone2_advanced.ipynb` with `QUICK_TEST = False`, keep outputs.
2. Download `checkpoints_m2/best_model.pt` from Colab.
3. Fill every `[FILL IN]` in `reports/milestone2.md` (params, epochs, M2 CER/WER,
   comparison table, error-analysis notes on hard cases).
4. Code + weights + interim report (+ notebook) = the M2 submission.

---

## Milestone 3 — Final Eval, Tuning & Presentation (due 28 Jun 2026)

Plan deliverables → repo artifacts (built during M3, listed here for planning):

| Plan deliverable | Submit | Status |
|------------------|--------|--------|
| Final clean, documented, runnable inference pipeline | `src/inference.py` (+ `src/evaluate.py`, `notebooks/milestone3_final.ipynb`) | ✅ done |
| Comprehensive final report (methodology, experiments, results, conclusions) | `reports/final_report.md` | ✅ done (test CER 0.0969 / WER 0.2215) |
| Project summary presentation | `reports/presentation.md` | ✅ done |

**M3 additional work from the plan (done in the notebook):** fine-tuning from the
M2 weights at a lower LR (`configs/milestone3_finetune.yaml`), and evaluation on
the **test set** (`src/evaluate.py`) for the final numbers.

**Before submitting M3:**
1. Run `notebooks/milestone3_final.ipynb` with `QUICK_TEST = True` (sanity), then `False` (~1.5h).
2. Paste the M3 val CER/WER and test-set CER/WER → fill `reports/final_report.md` + `reports/presentation.md`.
3. Submission bundle = inference pipeline code + final report + presentation (+ notebook with outputs).

---

## How the pieces map (quick reference)

```
M1  notebooks/milestone1_baseline.ipynb  +  checkpoints/best_model.pt      +  reports/milestone1.md
M2  src/model.py, src/augmentation.py    +  checkpoints_m2/best_model.pt   +  reports/milestone2.md (+ notebook)
M3  inference pipeline                   +  final report                   +  presentation
```

Each milestone builds on the last but is submitted as its own bundle. Nothing
from a later milestone is needed to submit an earlier one.
