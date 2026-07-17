"""Milestone 3 — final evaluation on the held-out test split.

Unlike M1/M2 (which report validation metrics during training), this scores a
trained model on the test set for the final, unbiased performance number
required by the project plan.
"""

import numpy as np
import torch

from src.dataset import get_dataloaders
from src.metrics import compute_cer, compute_wer, evaluate_batch


def evaluate_test_set(model, config, label_encoder, device, split="test"):
    """Run a trained model over a split and return aggregate metrics + per-sample
    predictions. `split` is 'test' or 'val' (uses the matching dataloader)."""
    train_loader, val_loader, test_loader = get_dataloaders(config, label_encoder)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[split]

    model.eval()
    records = []  # (cer, wer, pred, target)
    with torch.no_grad():
        for images, labels, label_lengths, input_lengths in loader:
            images = images.to(device)
            log_probs = model(images)
            out = evaluate_batch(log_probs.cpu(), labels, label_lengths, label_encoder)
            for pred, target in zip(out["predictions"], out["targets"]):
                records.append((compute_cer(pred, target), compute_wer(pred, target), pred, target))

    cers = np.array([r[0] for r in records])
    wers = np.array([r[1] for r in records])
    return {
        "split": split,
        "n": len(records),
        "mean_cer": float(cers.mean()),
        "median_cer": float(np.median(cers)),
        "mean_wer": float(wers.mean()),
        "perfect": int((cers == 0).sum()),
        "records": records,
    }


def print_report(metrics, worst_n=20):
    m = metrics
    print(f"=== Test-set evaluation ({m['n']} samples, split='{m['split']}') ===")
    print(f"Mean CER:   {m['mean_cer']:.4f}")
    print(f"Median CER: {m['median_cer']:.4f}")
    print(f"Mean WER:   {m['mean_wer']:.4f}")
    print(f"Perfect (CER=0): {m['perfect']} / {m['n']} ({100 * m['perfect'] / m['n']:.1f}%)")

    worst = sorted(m["records"], key=lambda r: r[0], reverse=True)[:worst_n]
    print(f"\n=== Worst {worst_n} predictions (highest CER) ===")
    print(f'{"CER":>6} | {"Predicted":>20} | {"Ground Truth":>20}')
    print("-" * 55)
    for cer, _wer, pred, target in worst:
        print(f"{cer:6.2f} | {pred:>20} | {target:>20}")
