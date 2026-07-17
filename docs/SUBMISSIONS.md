# Submission Checklist

Exactly what to hand in for each milestone, mapped from the work plan
(`docs/תוכנית עבודה - זיהוי כתב יד של ילדים.pdf`) to files in this repo.
Each milestone is self-contained — submit only that milestone's rows.

---

## Milestone 1 — Data, Preprocessing & Baseline (due 17 May 2026)

Plan deliverables → repo artifacts:

| Plan deliverable | Submit | Status |
|------------------|--------|--------|
| Jupyter/Colab notebook showing the data pipeline | `notebooks/milestone1_baseline.ipynb` (with outputs) | ✅ done |
| Trained baseline model | `checkpoints/best_model.pt` (from the full run) | ⬜ export from Colab |
| Short report: dataset, its distribution, baseline metrics (CER) | `reports/milestone1.md` | ⬜ fill `[FILL IN]` from run |

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
| Updated model weights | `checkpoints_m2/best_model.pt` (from the full run) | ⬜ export from Colab |
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
| Final clean, documented, runnable inference pipeline | `src/` + an `inference.py` / inference notebook (TBD) | ⬜ M3 |
| Comprehensive final report (methodology, experiments, results, conclusions) | `reports/final_report.md` (TBD) | ⬜ M3 |
| Project summary presentation | `reports/presentation.*` (TBD) | ⬜ M3 |

**M3 additional work from the plan:** final fine-tuning + hyperparameter search,
and evaluation on the **test set** (not just val) for the final numbers.

---

## How the pieces map (quick reference)

```
M1  notebooks/milestone1_baseline.ipynb  +  checkpoints/best_model.pt      +  reports/milestone1.md
M2  src/model.py, src/augmentation.py    +  checkpoints_m2/best_model.pt   +  reports/milestone2.md (+ notebook)
M3  inference pipeline                   +  final report                   +  presentation
```

Each milestone builds on the last but is submitted as its own bundle. Nothing
from a later milestone is needed to submit an earlier one.
