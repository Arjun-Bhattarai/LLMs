import torch

from llm.transformer.transformer_block import TransformerBlock


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


def test_transformer_block():
    torch.manual_seed(123)

    x = torch.rand(2, 4, 768)

    block = TransformerBlock(GPT_CONFIG_124M)

    output = block(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Transformer Block test passed!")


if __name__ == "__main__":
    test_transformer_block()