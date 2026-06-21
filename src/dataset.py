import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 '.,-/:;!?\"()+&#"


class LabelEncoder:
    def __init__(self, alphabet=ALPHABET):
        self.blank = 0
        self.char_to_idx = {c: i + 1 for i, c in enumerate(alphabet)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(alphabet)}
        self.idx_to_char[0] = ""

    def __len__(self):
        return len(self.char_to_idx) + 1

    def encode(self, text):
        text = text.lower()
        return [self.char_to_idx.get(c, self.char_to_idx[" "]) for c in text]

    def decode(self, indices):
        chars = []
        prev = None
        for idx in indices:
            if idx != self.blank and idx != prev:
                c = self.idx_to_char.get(idx, "")
                if c:
                    chars.append(c)
            prev = idx
        return "".join(chars)

    def decode_greedy(self, log_probs):
        indices = log_probs.argmax(dim=-1)
        return self.decode(indices.tolist())


class IAMDataset(Dataset):
    """Wraps HuggingFace IAM dataset for use with CRNN pipeline."""

    def __init__(self, hf_dataset, transform=None):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = sample["image"]

        if img.mode != "L":
            img = img.convert("L")
        img = np.array(img, dtype=np.uint8)

        label = sample["text"]

        if self.transform:
            img = self.transform(img)

        return img, label


def load_iam_splits():
    """Load IAM word-level dataset from HuggingFace. Returns train, val, test splits."""
    from datasets import load_dataset

    ds = load_dataset("priyank-m/IAM_words_text_recognition")
    return ds["train"], ds["val"], ds["test"]


def collate_fn(batch, label_encoder=None):
    if label_encoder is None:
        label_encoder = LabelEncoder()

    images, labels_raw = zip(*batch)

    max_w = max(img.shape[-1] for img in images)

    padded_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            pad_w = max_w - img.shape[-1]
            if pad_w > 0:
                img = torch.nn.functional.pad(img, (0, pad_w), value=0)
            padded_images.append(img)
        else:
            h, w = img.shape[:2]
            padded = np.zeros((h, max_w), dtype=img.dtype)
            padded[:, :w] = img
            padded_images.append(
                torch.FloatTensor(padded).unsqueeze(0) / 255.0
            )

    images_tensor = torch.stack(padded_images)

    encoded_labels = []
    label_lengths = []
    for label in labels_raw:
        enc = label_encoder.encode(label)
        encoded_labels.extend(enc)
        label_lengths.append(len(enc))

    labels_tensor = torch.IntTensor(encoded_labels)
    label_lengths_tensor = torch.IntTensor(label_lengths)

    input_lengths = torch.IntTensor(
        [images_tensor.shape[-1] // 4] * len(padded_images)
    )

    return images_tensor, labels_tensor, label_lengths_tensor, input_lengths


def get_dataloaders(config, label_encoder=None):
    from functools import partial
    from src.preprocessing import build_preprocessing_pipeline

    if label_encoder is None:
        label_encoder = LabelEncoder()

    transform = build_preprocessing_pipeline(
        config["data"]["img_height"],
        config["data"]["img_width"],
    )

    train_hf, val_hf, test_hf = load_iam_splits()

    train_ds = IAMDataset(train_hf, transform=transform)
    val_ds = IAMDataset(val_hf, transform=transform)
    test_ds = IAMDataset(test_hf, transform=transform)

    collate = partial(collate_fn, label_encoder=label_encoder)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
