from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointTransformerLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_neighbors: int = 16):
        super().__init__()
        self.num_neighbors = num_neighbors

        # Linear layers for query, key, value
        self.to_q = nn.Linear(in_dim, out_dim)
        self.to_k = nn.Linear(in_dim, out_dim)
        self.to_v = nn.Linear(in_dim, out_dim)

        # Position encoding
        self.positional_encoding = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        # Gamma function for attention score transformation
        self.gamma = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """
        x: point features (B, N, C)
        position: point positions (B, N, 3)
        """
        B, N, C = x.shape

        # Get k-nearest neighbors
        dist = torch.cdist(position, position)  # (B, N, N)
        _, idx = torch.topk(
            dist, k=self.num_neighbors, dim=-1, largest=False
        )  # (B, N, K)

        # Gather neighbors
        neighbors = torch.gather(
            x.unsqueeze(2).expand(-1, -1, self.num_neighbors, -1),
            dim=1,
            index=idx.unsqueeze(-1).expand(-1, -1, -1, C),
        )  # (B, N, K, C)

        # Compute query, key, value
        q = self.to_q(x)  # (B, N, C)
        k = self.to_k(neighbors)  # (B, N, K, C)
        v = self.to_v(neighbors)  # (B, N, K, C)

        # Position encoding
        positional_encoding = self.positional_encoding(
            position.unsqueeze(2) - position.unsqueeze(1)
        )  # (B, N, N, C)
        positional_encoding = torch.gather(
            positional_encoding, dim=2, index=idx.unsqueeze(-1).expand(-1, -1, -1, C)
        )  # (B, N, K, C)

        # Compute attention scores using subtraction as in original paper
        attention = (q.unsqueeze(2) - k + positional_encoding) / torch.sqrt(
            torch.tensor(C, dtype=torch.float32)
        )

        # Apply gamma function to attention scores
        attention = self.gamma(attention)

        attention = F.softmax(attention, dim=2)  # (B, N, K, C)

        # Apply attention
        out = torch.sum(attention * (v + positional_encoding), dim=2)  # (B, N, C)

        return out


class PointTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_neighbors: int = 16):
        super().__init__()
        self.attention = PointTransformerLayer(dim, dim, num_neighbors)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """
        x: point features (B, N, C)
        position: point positions (B, N, 3)
        """
        # Self-attention
        x = x + self.attention(self.norm1(x), position)

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class PointTransformer(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, num_blocks: int = 4, num_neighbors: int = 16
    ):
        super().__init__()

        # Initial feature embedding
        self.embed = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [PointTransformerBlock(out_dim, num_neighbors) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """
        x: point features (B, N, C)
        position: point positions (B, N, 3)
        """
        # Initial embedding
        x = self.embed(x.transpose(1, 2)).transpose(1, 2)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, position)

        return x
