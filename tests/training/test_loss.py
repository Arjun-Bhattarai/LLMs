import torch

from llm.training.loss import (
    cross_entropy_loss,
    perplexity,
)


def test_cross_entropy_loss():

    logits = torch.tensor(
        [
            [
                [2.0, 1.0, 0.1],
                [0.1, 2.0, 1.0],
            ]
        ]
    )

    targets = torch.tensor(
        [
            [0, 1],
        ]
    )

    loss = cross_entropy_loss(logits, targets)

    assert loss.ndim == 0
    assert loss.item() > 0


def test_perplexity():

    loss = torch.tensor(1.0)

    result = perplexity(loss)

    assert torch.allclose(
        result,
        torch.exp(loss),
    )


def test_loss_with_gpt_shape():

    batch_size = 2
    seq_len = 3
    vocab_size = 50257

    logits = torch.randn(
        batch_size,
        seq_len,
        vocab_size,
    )

    targets = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
    )

    loss = cross_entropy_loss(
        logits,
        targets,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_perplexity_is_positive():

    loss = torch.tensor(2.0)

    result = perplexity(loss)

    assert result > 0