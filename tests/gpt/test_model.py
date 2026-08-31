import torch

from llm.gpt.config import GPT_CONFIG_124M
from llm.gpt.model import GPTModel


def test_gpt_model_output_shape():

    torch.manual_seed(123)

    model = GPTModel(GPT_CONFIG_124M)

    batch = torch.randint(
        0,
        GPT_CONFIG_124M["vocab_size"],
        (2, 4),
    )

    logits = model(batch)

    assert logits.shape == (
        2,
        4,
        GPT_CONFIG_124M["vocab_size"],
    )


def test_gpt_model_output_is_finite():

    model = GPTModel(GPT_CONFIG_124M)

    batch = torch.randint(
        0,
        GPT_CONFIG_124M["vocab_size"],
        (2, 4),
    )

    logits = model(batch)

    assert torch.isfinite(logits).all()