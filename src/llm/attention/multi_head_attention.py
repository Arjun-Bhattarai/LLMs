import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.

    Uses learnable Query, Key, and Value projections.
    Prevents tokens from attending to future tokens.
    """

    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout=0.0,
        num_heads=2,
        qkv_bias=False,
    ):
        super().__init__()

        assert d_out % num_heads == 0, (
            "d_out must be divisible by num_heads"
        )

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Learnable Query, Key, and Value projections
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Final projection after combining all heads
        self.out_proj = nn.Linear(d_out, d_out)

        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)

        # Causal mask: prevents looking at future tokens
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1,
            ),
        )

    def forward(self, x):
        """
        Input:
            x: (batch_size, num_tokens, d_in)

        Output:
            context_vec: (batch_size, num_tokens, d_out)
        """

        batch_size, num_tokens, _ = x.shape

        # Create Query, Key, and Value
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Split d_out into multiple heads
        # (B, T, d_out)
        # -> (B, T, H, head_dim)

        queries = queries.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )

        keys = keys.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )

        values = values.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )

        # Move heads before token dimension
        # (B, T, H, D)
        # -> (B, H, T, D)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Attention scores
        # (B, H, T, D) @ (B, H, D, T)
        # -> (B, H, T, T)

        attn_scores = queries @ keys.transpose(2, 3)

        # Apply causal mask
        mask = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores = attn_scores.masked_fill(
            mask,
            -torch.inf,
        )

        # Scaled softmax
        attn_weights = torch.softmax(
            attn_scores / (self.head_dim ** 0.5),
            dim=-1,
        )

        # Dropout
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        # (B, H, T, T) @ (B, H, T, D)
        # -> (B, H, T, D)

        context_vec = attn_weights @ values

        # Move token dimension back
        # (B, H, T, D)
        # -> (B, T, H, D)

        context_vec = context_vec.transpose(1, 2)

        # Combine all heads
        # (B, T, H, D)
        # -> (B, T, d_out)

        context_vec = context_vec.contiguous().view(
            batch_size,
            num_tokens,
            self.d_out,
        )

        # Final output projection
        context_vec = self.out_proj(context_vec)

        return context_vec