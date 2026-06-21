import editdistance


def compute_cer(predicted, target):
    if len(target) == 0:
        return 0.0 if len(predicted) == 0 else 1.0
    return editdistance.eval(predicted, target) / len(target)


def compute_wer(predicted, target):
    pred_words = predicted.split()
    target_words = target.split()
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return editdistance.eval(pred_words, target_words) / len(target_words)


def evaluate_batch(log_probs, labels, label_lengths, label_encoder):
    batch_size = log_probs.size(1)
    predictions = []
    targets = []

    offset = 0
    for i in range(batch_size):
        length = label_lengths[i].item()
        target_indices = labels[offset:offset + length].tolist()
        target_text = "".join(
            label_encoder.idx_to_char.get(idx, "") for idx in target_indices
        )
        targets.append(target_text)
        offset += length

        pred_text = label_encoder.decode_greedy(log_probs[:, i, :])
        predictions.append(pred_text)

    total_cer = sum(compute_cer(p, t) for p, t in zip(predictions, targets))
    total_wer = sum(compute_wer(p, t) for p, t in zip(predictions, targets))

    return {
        "cer": total_cer / batch_size,
        "wer": total_wer / batch_size,
        "predictions": predictions,
        "targets": targets,
    }
