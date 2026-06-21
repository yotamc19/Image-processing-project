import cv2
import numpy as np
import torch


def build_preprocessing_pipeline(img_height=32, img_width=128):
    def preprocess(img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        img = cv2.medianBlur(img, 3)

        h, w = img.shape
        new_w = int(w * img_height / h)
        if new_w > img_width:
            new_w = img_width
        img = cv2.resize(img, (new_w, img_height))

        if new_w < img_width:
            pad = np.full((img_height, img_width - new_w), 255, dtype=np.uint8)
            img = np.concatenate([img, pad], axis=1)

        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5

        tensor = torch.FloatTensor(img).unsqueeze(0)
        return tensor

    return preprocess


def visualize_preprocessing(img_path):
    import matplotlib.pyplot as plt

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    steps = [("Original", img.copy())]

    binarized = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    steps.append(("Binarized", binarized.copy()))

    denoised = cv2.medianBlur(binarized, 3)
    steps.append(("Denoised", denoised.copy()))

    h, w = denoised.shape
    new_w = int(w * 32 / h)
    if new_w > 128:
        new_w = 128
    resized = cv2.resize(denoised, (new_w, 32))
    if new_w < 128:
        pad = np.full((32, 128 - new_w), 255, dtype=np.uint8)
        resized = np.concatenate([resized, pad], axis=1)
    steps.append(("Resized + Padded", resized))

    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - 0.5) / 0.5
    steps.append(("Normalized", normalized))

    fig, axes = plt.subplots(1, len(steps), figsize=(4 * len(steps), 3))
    for ax, (title, step_img) in zip(axes, steps):
        ax.imshow(step_img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
