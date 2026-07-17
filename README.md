# Children's Handwriting Recognition (OCR)

Deep learning OCR system for recognizing children's handwriting, built as a university image processing project.

**Run Milestone 1 in Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yotamc19/Image-processing-project/blob/main/notebooks/milestone1_baseline.ipynb)

**Run Milestone 2 in Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yotamc19/Image-processing-project/blob/main/notebooks/milestone2_advanced.ipynb)

**Run Milestone 3 in Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yotamc19/Image-processing-project/blob/main/notebooks/milestone3_final.ipynb)

Set `QUICK_TEST = True` (cell 2) for a ~2-3 min end-to-end sanity run on a small subset before committing to the full training run.

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

## How the Pipeline Works (Milestone 1 Notebook)

Each step matches a section in `notebooks/milestone1_baseline.ipynb`:

### Step 1 — Setup
Clones the project repo into Colab and installs dependencies. Like copying your project folder onto Colab's computer.

### Step 2 — Load Dataset
Downloads thousands of images of handwritten text (IAM dataset) from HuggingFace automatically. Each image is a photo of a handwritten line paired with what it actually says. This is what the model learns from.

### Step 3 — Dataset Exploration
Look at the data before using it: how many samples, word length distribution, which characters appear most often. Includes histograms and charts.

### Step 4 — Sample Images
Display 20 random handwriting images with their labels so you can see what the data looks like.

### Step 5 — Preprocessing Demo
Clean up images before feeding them to the model, shown side-by-side (before vs after):
- **Binarization** — convert to pure black and white (remove gray areas)
- **Denoising** — remove tiny dots and specs
- **Resize & normalize** — make all images the same size (the model needs fixed dimensions)

### Step 6 — Build Model
Create the CRNN neural network — the "brain" that will learn to read handwriting. It hasn't learned anything yet, just the empty architecture. Prints the model structure and parameter count.

Architecture: CNN extracts visual features → BiLSTM reads them as a sequence → CTC loss handles variable-length output.

### Step 7 — Train
The main event. Show the model thousands of handwriting images and tell it what each one says, over and over (up to 50 rounds/epochs). Each round it gets slightly better at reading. Training stops early if the model stops improving. Takes ~30-90 min on Colab GPU.

### Step 8 — Results & Predictions
Load the best model checkpoint, plot training curves (loss and CER over epochs), and show 15 sample predictions comparing what the model thinks vs ground truth.

### Step 9 — Error Analysis
Find the model's 20 worst mistakes and show overall statistics. Understanding where it fails tells you what to improve in milestone 2.

## Milestones

| # | Description | Status |
|---|-------------|--------|
| 1 | Data collection, preprocessing, baseline CRNN | Done — CER 0.1065 / WER 0.2460 |
| 2 | Advanced architecture (STN-CRNN) + data augmentation | Done — CER 0.0867 / WER 0.2012 (−18.6%) |
| 3 | Fine-tuning, test-set eval, inference pipeline, report + slides | Built — awaiting full run |
