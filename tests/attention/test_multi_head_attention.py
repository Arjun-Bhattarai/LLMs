import torch

from llm.attention.multi_head_attention import MultiHeadAttention


def main():
    torch.manual_seed(123)

    # Example input:
    # batch_size = 2
    # context_length = 6
    # embedding dimension = 3

    inputs = torch.tensor(
        [
            [
                [0.43, 0.15, 0.89],
                [0.55, 0.87, 0.66],
                [0.57, 0.85, 0.64],
                [0.22, 0.58, 0.33],
                [0.77, 0.25, 0.10],
                [0.05, 0.80, 0.55],
            ],
            [
                [0.43, 0.15, 0.89],
                [0.55, 0.87, 0.66],
                [0.57, 0.85, 0.64],
                [0.22, 0.58, 0.33],
                [0.77, 0.25, 0.10],
                [0.05, 0.80, 0.55],
            ],
        ]
    )

    d_in = 3
    d_out = 6
    context_length = 6
    num_heads = 2

    attention = MultiHeadAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=num_heads,
    )

    output = attention(inputs)

    print("Input shape:", inputs.shape)
    print("Output shape:", output.shape)
    print("\nOutput:")
    print(output)

    # Basic checks
    assert output.shape == (2, 6, 6)

    print("\n✓ Multi-Head Attention test passed!")


if __name__ == "__main__":
    main()