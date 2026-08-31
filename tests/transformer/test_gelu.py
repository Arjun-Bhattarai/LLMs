import torch

from llm.transformer.gelu import GELU


def test_gelu_output_shape():
    gelu = GELU()

    x = torch.randn(2, 3, 768)
    output = gelu(x)

    assert output.shape == x.shape


def test_gelu_output_is_finite():
    gelu = GELU()

    x = torch.randn(2, 3, 768)
    output = gelu(x)

    assert torch.isfinite(output).all()