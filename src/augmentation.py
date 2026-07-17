"""Milestone 2 — custom data augmentation simulating children's handwriting.

Children's handwriting differs from the adult IAM data mainly in *geometry*:
inconsistent letter sizes, slanted/rotated words, skewed baselines and uneven
spacing (project plan, task 2). These transforms perturb the clean adult words
along exactly those axes so the model learns to be robust to them.

Each transform takes a uint8 grayscale image (H, W) with white background
(255) and returns a uint8 image of the same dtype. Augmentation runs *before*
the preprocessing pipeline (binarization/resize), and only on the train split.
"""

import random

import cv2
import numpy as np

# White background so warped/border regions match the paper, not black ink.
_BORDER = 255


def _random_affine(img, rng):
    """Rotation + shear + anisotropic scale — slant and size variability."""
    h, w = img.shape
    angle = rng.uniform(-8, 8)          # skewed baseline
    scale = rng.uniform(0.85, 1.15)     # inconsistent letter size
    shear = rng.uniform(-0.25, 0.25)    # slant

    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    m[0, 1] += shear                    # inject horizontal shear
    return cv2.warpAffine(
        img, m, (w, h), borderValue=_BORDER,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )


def _elastic(img, rng, alpha=6.0, sigma=4.0):
    """Smooth local warping — wobbly, non-uniform strokes."""
    h, w = img.shape
    dx = cv2.GaussianBlur((rng.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur((rng.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma)
    dx *= alpha
    dy *= alpha
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xx + dx).astype(np.float32)
    map_y = (yy + dy).astype(np.float32)
    return cv2.remap(
        img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=_BORDER,
    )


def _baseline_wave(img, rng):
    """Column-wise vertical shift — wavy, non-flat baseline."""
    h, w = img.shape
    amp = rng.uniform(1.0, 3.0)
    phase = rng.uniform(0, 2 * np.pi)
    freq = rng.uniform(0.5, 1.5) * np.pi / w
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    shift = amp * np.sin(freq * xx + phase)
    map_x = xx.astype(np.float32)
    map_y = (yy + shift).astype(np.float32)
    return cv2.remap(
        img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=_BORDER,
    )


class ChildrenAugmentation:
    """Randomly applies the children's-handwriting distortions.

    Callable: uint8 (H, W) -> uint8 (H, W). Composable in front of the
    preprocessing pipeline for the train split only.
    """

    def __init__(self, p=0.5, seed=None):
        self.p = p
        self._np_rng = np.random.RandomState(seed)
        self._py_rng = random.Random(seed)

    def __call__(self, img):
        if self._py_rng.random() < self.p:
            img = _random_affine(img, self._py_rng)
        if self._py_rng.random() < self.p:
            img = _elastic(img, self._np_rng)
        if self._py_rng.random() < self.p:
            img = _baseline_wave(img, self._np_rng)
        return img


def build_augmentation(config):
    """Build the train-time augmentation from a config's `data` block.

    Returns None when augmentation is disabled so the caller can skip it.
    """
    aug_cfg = config.get("data", {}).get("augment")
    if not aug_cfg:
        return None
    p = 0.5 if aug_cfg is True else float(aug_cfg.get("p", 0.5))
    return ChildrenAugmentation(p=p)
