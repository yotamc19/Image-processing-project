import os
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src.dataset import LabelEncoder, get_dataloaders
from src.model import CRNN
from src.metrics import evaluate_batch
from src.utils import save_checkpoint, plot_training_curves


def train_one_epoch(model, dataloader, optimizer, criterion, device, clip_grad=5.0):
    model.train()
    total_loss = 0
    num_batches = 0

    for images, labels, label_lengths, input_lengths in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        input_lengths = input_lengths.to(device)

        log_probs = model(images)

        loss = criterion(log_probs, labels, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(model, dataloader, criterion, label_encoder, device):
    model.eval()
    total_loss = 0
    total_cer = 0
    total_wer = 0
    num_batches = 0

    with torch.no_grad():
        for images, labels, label_lengths, input_lengths in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            input_lengths = input_lengths.to(device)

            log_probs = model(images)

            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            total_loss += loss.item()

            results = evaluate_batch(
                log_probs.cpu(), labels.cpu(), label_lengths.cpu(), label_encoder
            )
            total_cer += results["cer"]
            total_wer += results["wer"]
            num_batches += 1

    n = max(num_batches, 1)
    return {
        "val_loss": total_loss / n,
        "cer": total_cer / n,
        "wer": total_wer / n,
    }


def train(config_path, checkpoint_dir="checkpoints"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    label_encoder = LabelEncoder()
    train_loader, val_loader, _ = get_dataloaders(config, label_encoder)

    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")

    model = CRNN(
        num_classes=len(label_encoder),
        rnn_hidden=config["model"]["rnn_hidden"],
        rnn_layers=config["model"]["rnn_layers"],
        dropout=config["model"]["dropout"],
        use_stn=config["model"].get("use_stn", False),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"]["step_size"],
        gamma=config["training"]["gamma"],
    )
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    best_cer = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stop_patience"]

    train_losses = []
    val_losses = []
    val_cers = []

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(config["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            clip_grad=config["training"]["clip_grad"],
        )

        val_metrics = validate(model, val_loader, criterion, label_encoder, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_metrics["val_loss"])
        val_cers.append(val_metrics["cer"])

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"CER: {val_metrics['cer']:.4f} | "
            f"WER: {val_metrics['wer']:.4f}"
        )

        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(checkpoint_dir, "best_model.pt"),
            )
            print(f"  -> New best CER: {best_cer:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"\nTraining complete. Best CER: {best_cer:.4f}")
    plot_training_curves(train_losses, val_losses, val_cers)

    return {
        "best_cer": best_cer,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_cers": val_cers,
    }
