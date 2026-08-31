import os
import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    global_step,
    train_loss,
    val_loss,
    config,
    path,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(checkpoint, path)


def load_checkpoint(
    model,
    optimizer,
    path,
    device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    return checkpoint