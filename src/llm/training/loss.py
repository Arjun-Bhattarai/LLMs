import torch
import torch.nn.functional as F


def cross_entropy_loss(logits, targets):
    """
    Calculate next-token prediction cross-entropy loss.

    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)
        targets: Tensor of shape (batch_size, seq_len)

    Returns:
        Scalar cross-entropy loss.
    """

    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()

    return F.cross_entropy(logits_flat, targets_flat)


def perplexity(loss):
    """
    Calculate perplexity from cross-entropy loss.
    """

    return torch.exp(loss)


def calc_loss_batch(
    input_batch,
    target_batch,
    model,
    device,
):
    """
    Calculate loss for a single batch.
    """

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    loss = cross_entropy_loss(
        logits,
        target_batch,
    )

    return loss


def calc_loss_loader(
    data_loader,
    model,
    device,
    num_batches=None,
):
    """
    Calculate average loss over a DataLoader.
    """

    total_loss = 0.0

    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(
            num_batches,
            len(data_loader),
        )

    for i, (input_batch, target_batch) in enumerate(data_loader):

        if i >= num_batches:
            break

        loss = calc_loss_batch(
            input_batch,
            target_batch,
            model,
            device,
        )

        total_loss += loss.item()

    return total_loss / num_batches