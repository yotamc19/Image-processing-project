import torch
import matplotlib.pyplot as plt


def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def plot_training_curves(train_losses, val_losses, val_cers):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(val_cers, label="Val CER", color="red")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("CER")
    ax2.set_title("Validation Character Error Rate")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def show_predictions(model, dataloader, label_encoder, device, n=10):
    model.eval()
    shown = 0

    fig, axes = plt.subplots(n, 1, figsize=(12, 2 * n))
    if n == 1:
        axes = [axes]

    with torch.no_grad():
        for images, labels, label_lengths, input_lengths in dataloader:
            images = images.to(device)
            log_probs = model(images)

            offset = 0
            for i in range(images.size(0)):
                if shown >= n:
                    break

                length = label_lengths[i].item()
                target_indices = labels[offset:offset + length].tolist()
                target_text = "".join(
                    label_encoder.idx_to_char.get(idx, "")
                    for idx in target_indices
                )
                offset += length

                pred_text = label_encoder.decode_greedy(log_probs[:, i, :])

                img = images[i].cpu().squeeze().numpy()
                img = (img * 0.5 + 0.5)

                axes[shown].imshow(img, cmap="gray")
                axes[shown].set_title(
                    f"GT: '{target_text}' | Pred: '{pred_text}'"
                )
                axes[shown].axis("off")
                shown += 1

            if shown >= n:
                break

    plt.tight_layout()
    plt.show()
