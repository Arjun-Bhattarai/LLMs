import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings."""

    def __init__(
        self,
        context_length: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            context_length,
            embedding_dim,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[-1]

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        return self.embedding(positions)