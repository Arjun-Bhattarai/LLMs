import torch
import tiktoken

from llm.gpt.config import GPT_CONFIG_124M
from llm.gpt.model import GPTModel
from llm.training.trainer import (
    evaluate_model,
    train_model_simple,
)


def create_dummy_loader(
    batch_size=2,
    seq_len=8,
    num_batches=2,
):
    """Create small dummy input/target batches for testing."""

    batches = []

    vocab_size = GPT_CONFIG_124M["vocab_size"]

    for _ in range(num_batches):
        inputs = torch.randint(
            0,
            vocab_size,
            (batch_size, seq_len),
        )

        targets = torch.randint(
            0,
            vocab_size,
            (batch_size, seq_len),
        )

        batches.append((inputs, targets))

    return batches


def test_evaluate_model():
    """Test training and validation loss calculation."""

    config = GPT_CONFIG_124M.copy()

    # Small model for faster testing
    config["context_length"] = 8
    config["emb_dim"] = 64
    config["n_heads"] = 4
    config["n_layers"] = 2

    model = GPTModel(config)

    train_loader = create_dummy_loader()
    val_loader = create_dummy_loader()

    device = torch.device("cpu")

    train_loss, val_loss = evaluate_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        eval_iter=2,
    )

    assert isinstance(train_loss, float)
    assert isinstance(val_loss, float)

    assert train_loss >= 0
    assert val_loss >= 0


def test_train_model_simple():
    """Test complete training loop."""

    config = GPT_CONFIG_124M.copy()

    # Small model for fast testing
    config["context_length"] = 8
    config["emb_dim"] = 64
    config["n_heads"] = 4
    config["n_layers"] = 2

    model = GPTModel(config)

    train_loader = create_dummy_loader(
        batch_size=2,
        seq_len=8,
        num_batches=2,
    )

    val_loader = create_dummy_loader(
        batch_size=2,
        seq_len=8,
        num_batches=2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    device = torch.device("cpu")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=1,
        eval_freq=1,
        eval_iter=1,
        start_context="test",
        tokenizer=tokenizer,
    )

    assert len(train_losses) > 0
    assert len(val_losses) > 0
    assert len(tokens_seen) > 0

    assert len(train_losses) == len(val_losses)
    assert len(train_losses) == len(tokens_seen)

    assert all(loss >= 0 for loss in train_losses)
    assert all(loss >= 0 for loss in val_losses)

    assert tokens_seen[-1] > 0