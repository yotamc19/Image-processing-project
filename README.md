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

## Setup (Google Colab)

1. Open [Google Colab](https://colab.research.google.com)
2. File → Open notebook → GitHub → paste this repo's URL
3. Select the notebook you want to run (e.g. `notebooks/milestone1_baseline.ipynb`)
4. Set runtime to **T4 GPU**: Runtime → Change runtime type → T4 GPU → Save
5. Run cells one by one with **Shift + Enter**

The dataset downloads automatically from HuggingFace — no manual download needed.

## How the Pipeline Works

### 1. Load Dataset
Downloads thousands of images of handwritten text (IAM dataset) from HuggingFace. Each image is a photo of a word paired with what that word actually says. This is what the model learns from.

### 2. Explore Data
Examine the dataset before using it: how many samples, word length distribution, which characters appear most often.

### 3. Preprocessing
Clean up images before feeding them to the model:
- **Binarization** — convert to pure black and white (remove gray areas)
- **Denoising** — remove tiny dots and specs
- **Resize & normalize** — make all images the same size (the model needs fixed dimensions)

### 4. Build Model
Create the CRNN neural network — the "brain" that will learn to read handwriting. Architecture: CNN extracts visual features → BiLSTM reads them as a sequence → CTC loss handles variable-length output.

### 5. Train
Show the model thousands of handwriting images with their correct labels, over and over (50 rounds/epochs). Each round it gets slightly better at reading. Takes ~1-2 hours on a Colab GPU.

### 6. Evaluate
Measure how well the model learned using **CER** (Character Error Rate) — the percentage of characters it gets wrong. Lower is better. Show sample predictions (what it thinks vs what it actually says) and analyze the worst mistakes to understand where it fails.

## Milestones

| # | Description | Status |
|---|-------------|--------|
| 1 | Data collection, preprocessing, baseline CRNN | In progress |
| 2 | Advanced architecture (TrOCR/Attention) + data augmentation | Pending |
| 3 | Final evaluation, fine-tuning, report, presentation | Pending |
