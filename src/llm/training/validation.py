import torch

from llm.training.loss import cross_entropy_loss


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate cross-entropy loss for a single batch.
    """

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    loss = cross_entropy_loss(logits, target_batch)

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate average loss over a DataLoader.

    Args:
        data_loader: PyTorch DataLoader.
        model: GPT model.
        device: CPU or CUDA device.
        num_batches: Number of batches to evaluate.
                     If None, evaluates all batches.

    Returns:
        Average loss.
    """

    total_loss = 0.0

    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

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