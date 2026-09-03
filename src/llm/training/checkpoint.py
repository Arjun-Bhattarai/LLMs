
import os

import torch


def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    global_step,
    train_loss,
    val_loss,
    config,
    path,
):
    """
    Save the complete training state.

    Includes:
    - Model parameters
    - Optimizer state
    - AMP GradScaler state
    - Epoch
    - Global training step
    - Training loss
    - Validation loss
    - Model configuration
    """

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
    }

    checkpoint_dir = os.path.dirname(path)

    if checkpoint_dir:
        os.makedirs(
            checkpoint_dir,
            exist_ok=True,
        )

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    model,
    optimizer,
    scaler,
    path,
    device,
):
    """
    Load a training checkpoint.

    Supports both:
    - New checkpoints containing AMP scaler state
    - Older checkpoints without scaler state
    """

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    # Restore model
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Restore optimizer
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # Restore AMP scaler if available.
    #
    # Your existing 10K checkpoint was created
    # before scaler state was added, so it will
    # simply start with a fresh scaler.
    if (
        scaler is not None
        and "scaler_state_dict" in checkpoint
    ):
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

        print(
            "AMP GradScaler state restored."
        )

    elif scaler is not None:
        print(
            "No AMP GradScaler state found "
            "in checkpoint. Using a fresh scaler."
        )

    return checkpoint
