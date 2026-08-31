import torch
import torch.nn as nn

from llm.training.validation import (
    calc_loss_batch,
    calc_loss_loader,
)


class DummyModel(nn.Module):
    def __init__(self, vocab_size=10):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x):
        batch_size, seq_len = x.shape

        return torch.zeros(
            batch_size,
            seq_len,
            self.vocab_size,
        )


def test_calc_loss_batch():
    model = DummyModel(vocab_size=10)

    input_batch = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
    ])

    target_batch = torch.tensor([
        [2, 3, 4],
        [5, 6, 7],
    ])

    loss = calc_loss_batch(
        input_batch,
        target_batch,
        model,
        torch.device("cpu"),
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_calc_loss_loader():
    model = DummyModel(vocab_size=10)

    input_batch = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
    ])

    target_batch = torch.tensor([
        [2, 3, 4],
        [5, 6, 7],
    ])

    data_loader = [
        (input_batch, target_batch),
        (input_batch, target_batch),
    ]

    loss = calc_loss_loader(
        data_loader,
        model,
        torch.device("cpu"),
    )

    assert isinstance(loss, float)
    assert loss > 0
    assert torch.isfinite(torch.tensor(loss))


def test_calc_loss_loader_num_batches():
    model = DummyModel(vocab_size=10)

    input_batch = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
    ])

    target_batch = torch.tensor([
        [2, 3, 4],
        [5, 6, 7],
    ])

    data_loader = [
        (input_batch, target_batch),
        (input_batch, target_batch),
        (input_batch, target_batch),
    ]

    loss = calc_loss_loader(
        data_loader,
        model,
        torch.device("cpu"),
        num_batches=2,
    )

    assert isinstance(loss, float)
    assert loss > 0
    assert torch.isfinite(torch.tensor(loss))