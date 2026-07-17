"""Milestone 3 — inference pipeline.

Load a trained checkpoint and transcribe a handwritten-word image to text.
This is the clean, runnable entry point required by the project plan.

Usage (CLI):
    python -m src.inference --checkpoint checkpoints_m3/best_model.pt --image word.png

Usage (Python):
    from src.inference import load_model, predict
    model, encoder, device = load_model("checkpoints_m3/best_model.pt")
    text = predict(model, "word.png", encoder, device)
"""

import argparse

import numpy as np
import torch

from src.dataset import LabelEncoder
from src.model import CRNN
from src.preprocessing import build_preprocessing_pipeline


def load_model(checkpoint_path, use_stn=True, device=None):
    """Load a CRNN/STN-CRNN checkpoint. Returns (model, encoder, device)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = LabelEncoder()
    model = CRNN(num_classes=len(encoder), use_stn=use_stn).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, encoder, device


def _to_gray_uint8(image):
    """Accept a path, a PIL image, or a numpy array; return uint8 grayscale."""
    if isinstance(image, str):
        from PIL import Image
        image = Image.open(image)
    if hasattr(image, "convert"):  # PIL
        if image.mode != "L":
            image = image.convert("L")
        return np.array(image, dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 3:
        import cv2
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr.astype(np.uint8)


def predict(model, image, encoder, device, img_height=32, img_width=128):
    """Transcribe a single image (path / PIL / numpy) to text."""
    transform = build_preprocessing_pipeline(img_height, img_width)
    tensor = transform(_to_gray_uint8(image)).unsqueeze(0).to(device)  # (1,1,H,W)
    with torch.no_grad():
        log_probs = model(tensor)  # (T, 1, C)
    return encoder.decode_greedy(log_probs[:, 0, :])


def main():
    parser = argparse.ArgumentParser(description="Transcribe a handwritten word image.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--image", required=True, help="Path to the word image")
    parser.add_argument("--no-stn", action="store_true", help="Load a plain CRNN (M1 baseline)")
    args = parser.parse_args()

    model, encoder, device = load_model(args.checkpoint, use_stn=not args.no_stn)
    print(predict(model, args.image, encoder, device))


if __name__ == "__main__":
    main()
