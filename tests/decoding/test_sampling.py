import torch
import tiktoken

from llm.decoding.sampling import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
)


def test_text_to_token_ids():
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = text_to_token_ids(
        "Hello world",
        tokenizer,
    )

    assert token_ids.ndim == 2
    assert token_ids.shape[0] == 1
    assert token_ids.dtype == torch.long


def test_token_ids_to_text():
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = text_to_token_ids(
        "Hello world",
        tokenizer,
    )

    text = token_ids_to_text(
        token_ids,
        tokenizer,
    )

    assert text == "Hello world"


def test_generate_output_shape():
    class DummyModel(torch.nn.Module):
        def forward(self, idx):
            batch_size, seq_len = idx.shape

            return torch.zeros(
                batch_size,
                seq_len,
                100,
            )

    model = DummyModel()

    idx = torch.tensor([[1, 2, 3]])

    output = generate(
        model=model,
        idx=idx,
        max_new_tokens=5,
        context_size=10,
    )

    assert output.shape == (1, 8)


def test_generate_with_temperature_and_top_k():
    class DummyModel(torch.nn.Module):
        def forward(self, idx):
            batch_size, seq_len = idx.shape

            logits = torch.zeros(
                batch_size,
                seq_len,
                100,
            )

            logits[:, :, 5] = 10.0
            logits[:, :, 10] = 9.0
            logits[:, :, 20] = 8.0

            return logits

    model = DummyModel()

    idx = torch.tensor([[1, 2, 3]])

    torch.manual_seed(123)

    output = generate(
        model=model,
        idx=idx,
        max_new_tokens=5,
        context_size=10,
        temperature=1.0,
        top_k=3,
    )

    assert output.shape == (1, 8)
    assert torch.all(
        torch.isin(
            output[:, 3:],
            torch.tensor([5, 10, 20]),
        )
    )