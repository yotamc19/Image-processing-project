# Children's Handwriting Recognition (OCR)

Deep learning OCR system for recognizing children's handwriting, built as a university image processing project.

## Project Structure

```
├── configs/          # Training configuration files
├── notebooks/        # Colab notebooks (one per milestone)
├── src/              # Python source modules
├── data/             # Datasets (gitignored, downloaded at runtime)
├── checkpoints/      # Model checkpoints (gitignored)
├── reports/          # Milestone reports
└── docs/             # Project documentation
```

## Setup

1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Register for the IAM Handwriting Database at https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
4. Download `words.tgz` and `words.txt`, extract into `data/iam/`

## Running on Google Colab

Open the milestone notebooks in `notebooks/` — they handle setup, data loading, and training end-to-end.

## Milestones

| # | Description | Status |
|---|-------------|--------|
| 1 | Data collection, preprocessing, baseline CRNN | In progress |
| 2 | Advanced architecture (TrOCR/Attention) | Pending |
| 3 | Final evaluation, report, presentation | Pending |
