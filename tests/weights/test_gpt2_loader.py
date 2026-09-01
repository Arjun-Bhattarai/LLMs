import numpy as np
import pytest
import torch

from llm.weights.gpt2_loader import (
    MODEL_CONFIGS,
    assign,
    load_weights_into_gpt,
    load_pretrained_gpt2,
)


def test_model_configs():
    """All supported GPT-2 model sizes should be configured."""

    assert "124M" in MODEL_CONFIGS
    assert "355M" in MODEL_CONFIGS
    assert "774M" in MODEL_CONFIGS
    assert "1558M" in MODEL_CONFIGS


def test_assign_valid_shape():
    """assign() should convert a NumPy array into a Parameter."""

    left = torch.zeros(2, 3)
    right = np.zeros((2, 3))

    result = assign(left, right)

    assert isinstance(result, torch.nn.Parameter)
    assert result.shape == left.shape


def test_assign_shape_mismatch():
    """assign() should reject incompatible tensor shapes."""

    left = torch.zeros(2, 3)
    right = np.zeros((4, 5))

    with pytest.raises(ValueError):
        assign(left, right)


def test_load_pretrained_gpt2_124m():
    """
    GPT-2 124M should load successfully from
    the locally downloaded weights.
    """

    model = load_pretrained_gpt2(
        model_size="124M",
        models_dir="gpt2",
        device=torch.device("cpu"),
    )

    assert model is not None
    assert model.training is False

    # GPT-2 124M architecture
    assert model.tok_emb.weight.shape == (50257, 768)
    assert model.pos_emb.weight.shape == (1024, 768)

    # 12 transformer blocks
    assert len(model.trf_blocks) == 12


def test_pretrained_model_forward():
    """Loaded GPT-2 model should produce valid logits."""

    model = load_pretrained_gpt2(
        model_size="124M",
        models_dir="gpt2",
        device=torch.device("cpu"),
    )

    input_ids = torch.randint(
        0,
        50257,
        (1, 8),
    )

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (1, 8, 50257)
    assert torch.isfinite(logits).all()


def test_invalid_model_size():
    """Unsupported GPT-2 model sizes should raise an error."""

    with pytest.raises(ValueError):
        load_pretrained_gpt2(
            model_size="invalid",
            models_dir="gpt2",
            device=torch.device("cpu"),
        )
