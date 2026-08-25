"""Stage-by-stage visualization of the inference pipeline (demo video).

`src/inference.py` returns only the final string. This module exposes the
intermediate state so each step of image -> text can be shown on screen:

    1. preprocessing   raw photo -> binarized -> denoised -> resized + padded
    2. STN             the model's own straightened view of the word
    3. time slices     the CNN squashes width 128 -> 32 steps (~4 px each)
    4. per-step chars  what character each step predicts, and how confident
    5. CTC collapse    raw frame string -> merge repeats -> drop blanks -> word

Usage (notebook):
    from src.inference import load_model
    from src.visualize import explain
    model, encoder, device = load_model('checkpoints_m3/best_model.pt')
    result = explain(model, 'photo.jpg', encoder, device)
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.inference import _to_gray_uint8
from src.preprocessing import build_preprocessing_pipeline

BLANK_SYMBOL = "-"
SPACE_SYMBOL = "_"


def _display_char(char):
    """Make blanks and spaces visible in text panels."""
    if char == "":
        return BLANK_SYMBOL
    if char == " ":
        return SPACE_SYMBOL
    return char


def preprocessing_stages(image, img_height=32, img_width=128):
    """Return [(title, image)] for every preprocessing step, plus the model input.

    Mirrors `build_preprocessing_pipeline`; the final tensor comes from the real
    pipeline so what is shown is what the model is fed.
    """
    gray = _to_gray_uint8(image)
    stages = [("1. Original (grayscale)", gray)]

    binarized = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    stages.append(("2. Binarized", binarized))

    denoised = cv2.medianBlur(binarized, 3)
    stages.append(("3. Denoised", denoised))

    h, w = denoised.shape
    content_width = min(int(w * img_height / h), img_width)
    resized = cv2.resize(denoised, (content_width, img_height))
    if content_width < img_width:
        pad = np.full((img_height, img_width - content_width), 255, dtype=np.uint8)
        resized = np.concatenate([resized, pad], axis=1)
    stages.append((f"4. Resized + padded ({img_height}x{img_width})", resized))

    tensor = build_preprocessing_pipeline(img_height, img_width)(gray)
    return stages, tensor, content_width


def run_with_intermediates(model, image, encoder, device, img_height=32, img_width=128):
    """Transcribe an image and keep every intermediate the video needs."""
    stages, tensor, content_width = preprocessing_stages(image, img_height, img_width)
    batch = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        stn_out = model.stn(batch) if model.stn is not None else batch
        log_probs = model(batch)

    probs = log_probs[:, 0, :].exp().cpu().numpy()  # (T, C)
    top_idx = probs.argmax(axis=1)
    top_prob = probs.max(axis=1)

    frame_chars = [_display_char(encoder.idx_to_char.get(int(i), "")) for i in top_idx]
    text = encoder.decode(top_idx.tolist())

    return {
        "stages": stages,
        "idx_to_char": dict(encoder.idx_to_char),
        "model_input": tensor.squeeze(0).cpu().numpy(),
        "stn_output": stn_out[0, 0].cpu().numpy(),
        "content_width": content_width,
        "probs": probs,
        "top_idx": top_idx,
        "top_prob": top_prob,
        "frame_chars": frame_chars,
        "text": text,
        "num_steps": probs.shape[0],
        "img_width": img_width,
    }


def plot_preprocessing(result, figsize_per_panel=4):
    """Stage 1 — what the raw photo turns into before the network sees it."""
    stages = result["stages"]
    fig, axes = plt.subplots(1, len(stages), figsize=(figsize_per_panel * len(stages), 3))
    for ax, (title, img) in zip(axes, stages):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle("Stage 1 — Preprocessing", fontsize=15, y=1.06)
    plt.tight_layout()
    plt.show()


def plot_stn(result):
    """Stage 2 — the model's learned straightening of the word."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 4))
    axes[0].imshow(result["model_input"], cmap="gray")
    axes[0].set_title("Model input (after preprocessing)", fontsize=12)
    axes[1].imshow(result["stn_output"], cmap="gray")
    axes[1].set_title("After the STN — the network's own rectified view", fontsize=12)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Stage 2 — Spatial Transformer", fontsize=15, y=1.04)
    plt.tight_layout()
    plt.show()


def plot_time_slices(result, label_every=4):
    """Stage 3 — the width-to-time split: each step reads one vertical strip."""
    img = result["stn_output"]
    num_steps = result["num_steps"]
    px_per_step = result["img_width"] / num_steps

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.imshow(img, cmap="gray", aspect="auto", extent=[0, num_steps, img.shape[0], 0])
    for step in range(num_steps + 1):
        ax.axvline(step, color="tab:red", linewidth=0.6, alpha=0.7)
    ax.set_xticks(np.arange(0, num_steps, label_every) + 0.5)
    ax.set_xticklabels(range(0, num_steps, label_every), fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel(
        f"time step (each one reads ~{px_per_step:.0f} px of width)", fontsize=11
    )
    ax.set_title(
        f"Stage 3 — the CNN turns width {result['img_width']} into "
        f"{num_steps} time steps",
        fontsize=15,
    )
    plt.tight_layout()
    plt.show()


def plot_frame_predictions(result, min_prob=0.02, max_classes=12):
    """Stage 4 — per-step character, its confidence, and the full class heatmap."""
    probs = result["probs"]
    num_steps = result["num_steps"]
    img = result["stn_output"]

    fig, axes = plt.subplots(
        3, 1, figsize=(15, 8), gridspec_kw={"height_ratios": [1, 1.4, 2.2]}
    )

    axes[0].imshow(img, cmap="gray", aspect="auto", extent=[0, num_steps, img.shape[0], 0])
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[0].set_title("Stage 4 — what each time step reads", fontsize=15)

    axes[1].bar(np.arange(num_steps) + 0.5, result["top_prob"], color="tab:blue", width=0.8)
    for step, (char, prob) in enumerate(zip(result["frame_chars"], result["top_prob"])):
        axes[1].text(
            step + 0.5, min(prob + 0.04, 1.02), char,
            ha="center", va="bottom", fontsize=13, family="monospace",
            color="tab:red" if char == BLANK_SYMBOL else "black",
        )
    axes[1].set_xlim(0, num_steps)
    axes[1].set_ylim(0, 1.18)
    axes[1].set_ylabel("confidence", fontsize=11)
    axes[1].set_xticks([])
    axes[1].spines[["top", "right"]].set_visible(False)

    active = [c for c in range(probs.shape[1]) if probs[:, c].max() >= min_prob]
    active.sort(key=lambda c: probs[:, c].max(), reverse=True)
    active = sorted(active[:max_classes])
    heat = probs[:, active].T

    axes[2].imshow(
        heat, aspect="auto", cmap="viridis", vmin=0, vmax=1,
        extent=[0, num_steps, len(active) - 0.5, -0.5],
    )
    labels = []
    for c in active:
        char = _display_char(result_char(result, c))
        labels.append(f"blank {BLANK_SYMBOL}" if c == 0 else char)
    axes[2].set_yticks(range(len(active)))
    axes[2].set_yticklabels(labels, fontsize=11, family="monospace")
    axes[2].set_xlabel("time step", fontsize=11)
    axes[2].set_title("probability per character, per time step", fontsize=12)

    plt.tight_layout()
    plt.show()


def result_char(result, class_idx):
    """Character for a class index, using the encoder mapping captured at run time."""
    return result.get("idx_to_char", {}).get(class_idx, "")


def ctc_collapse_steps(result):
    """Return the three CTC decoding stages as display strings."""
    raw = "".join(result["frame_chars"])

    merged_chars = []
    prev = None
    for char in result["frame_chars"]:
        if char != prev:
            merged_chars.append(char)
        prev = char
    merged = "".join(merged_chars)

    final = "".join(
        c if c != SPACE_SYMBOL else " " for c in merged_chars if c != BLANK_SYMBOL
    )
    return raw, merged, final


def plot_ctc_collapse(result):
    """Stage 5 — raw frame string collapsing into the predicted word."""
    raw, merged, final = ctc_collapse_steps(result)
    lines = [
        ("raw per-step output", raw),
        ("merge repeated neighbours", merged),
        ("drop blanks -> prediction", final),
    ]

    fig, ax = plt.subplots(figsize=(15, 3.6))
    ax.axis("off")
    ax.set_title("Stage 5 — CTC collapse", fontsize=15, y=1.0)
    for i, (label, value) in enumerate(lines):
        y = 0.72 - i * 0.26
        ax.text(0.0, y, label, fontsize=12, color="gray", transform=ax.transAxes)
        ax.text(
            0.34, y, value, fontsize=17, family="monospace",
            transform=ax.transAxes, color="black" if i < 2 else "tab:red",
        )
    plt.tight_layout()
    plt.show()

    print(f"raw    : {raw}")
    print(f"merged : {merged}")
    print(f"final  : {final!r}")


def explain(model, image, encoder, device, img_height=32, img_width=128, show=True):
    """Run every stage and (by default) plot all five figures in order."""
    result = run_with_intermediates(model, image, encoder, device, img_height, img_width)
    if show:
        plot_preprocessing(result)
        if model.stn is not None:
            plot_stn(result)
        plot_time_slices(result)
        plot_frame_predictions(result)
        plot_ctc_collapse(result)
    return result
