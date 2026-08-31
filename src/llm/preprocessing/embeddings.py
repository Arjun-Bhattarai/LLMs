import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Maps token IDs to dense embedding vectors."""

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight