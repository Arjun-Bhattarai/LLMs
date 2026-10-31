"""Atomic, backward-compatible training checkpoints."""

import os
import random
import tempfile

import numpy as np
import torch


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    model, optimizer, scaler=None, epoch=0, global_step=0,
    train_loss=None, val_loss=None, config=None, path=None,
    scheduler=None, batch_in_epoch=0, tokens_seen=0,
    best_val_loss=float("inf"), sampler_state=None, extra_state=None,
):
    """Save full train state via same-directory atomic replacement.

    The scaler positional argument remains optional for legacy callers.
    """
    if path is None:
        raise ValueError("path is required")
    checkpoint = {
        "model_state_dict": _unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": int(epoch), "global_step": int(global_step),
        "batch_in_epoch": int(batch_in_epoch),
        "tokens_seen": int(tokens_seen),
        "train_loss": train_loss, "val_loss": val_loss,
        "best_val_loss": float(best_val_loss), "config": config,
        "rng_state": capture_rng_state(), "sampler_state": sampler_state,
        "extra_state": extra_state or {},
    }
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_checkpoint(
    model, optimizer=None, scaler=None, path=None, device="cpu", scheduler=None,
    restore_rng=True,
):
    """Restore a new or legacy checkpoint; missing legacy metadata is tolerated."""
    if path is None:
        raise ValueError("path is required")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    _unwrap(model).load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if restore_rng:
        restore_rng_state(checkpoint.get("rng_state"))
    checkpoint.setdefault("batch_in_epoch", 0)
    checkpoint.setdefault("tokens_seen", 0)
    checkpoint.setdefault("best_val_loss", checkpoint.get("val_loss", float("inf")))
    checkpoint.setdefault("scheduler_state_dict", None)
    checkpoint.setdefault("sampler_state", None)
    return checkpoint
