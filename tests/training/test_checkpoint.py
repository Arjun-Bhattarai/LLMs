import torch

from llm.gpt.config import GPT_CONFIG_124M
from llm.gpt.model import GPTModel
from llm.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


def test_save_checkpoint(tmp_path):
    """Test that a checkpoint is saved correctly."""

    config = GPT_CONFIG_124M.copy()

    # Small model for fast testing
    config["context_length"] = 8
    config["emb_dim"] = 64
    config["n_heads"] = 4
    config["n_layers"] = 2

    model = GPTModel(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=2,
        global_step=100,
        train_loss=2.5,
        val_loss=2.8,
        config=config,
        path=checkpoint_path,
    )

    assert checkpoint_path.exists()


def test_load_checkpoint(tmp_path):
    """Test that checkpoint data is restored correctly."""

    config = GPT_CONFIG_124M.copy()

    config["context_length"] = 8
    config["emb_dim"] = 64
    config["n_heads"] = 4
    config["n_layers"] = 2

    model = GPTModel(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=2,
        global_step=100,
        train_loss=2.5,
        val_loss=2.8,
        config=config,
        path=checkpoint_path,
    )

    new_model = GPTModel(config)

    new_optimizer = torch.optim.AdamW(
        new_model.parameters(),
        lr=0.001,
    )

    checkpoint = load_checkpoint(
        model=new_model,
        optimizer=new_optimizer,
        path=checkpoint_path,
        device=torch.device("cpu"),
    )

    assert checkpoint["epoch"] == 2
    assert checkpoint["global_step"] == 100
    assert checkpoint["train_loss"] == 2.5
    assert checkpoint["val_loss"] == 2.8
    assert checkpoint["config"] == config


def test_model_weights_are_restored(tmp_path):
    """Test that model weights are restored correctly."""

    config = GPT_CONFIG_124M.copy()

    config["context_length"] = 8
    config["emb_dim"] = 64
    config["n_heads"] = 4
    config["n_layers"] = 2

    model = GPTModel(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    checkpoint_path = tmp_path / "checkpoint.pt"

    original_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
    }

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=1,
        global_step=10,
        train_loss=3.0,
        val_loss=3.2,
        config=config,
        path=checkpoint_path,
    )

    new_model = GPTModel(config)

    new_optimizer = torch.optim.AdamW(
        new_model.parameters(),
        lr=0.001,
    )

    load_checkpoint(
        model=new_model,
        optimizer=new_optimizer,
        path=checkpoint_path,
        device=torch.device("cpu"),
    )

    for key, value in new_model.state_dict().items():
        assert torch.equal(
            value,
            original_state[key],
        )