from __future__ import annotations

import torch
from torch import nn


class ReadEncoder(nn.Module):
    def __init__(self, vocab_size: int, read_dim: int = 128, positions: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, read_dim)
        self.pos_embed = nn.Embedding(positions, read_dim)

    def forward(self, reads: torch.Tensor) -> torch.Tensor:
        batch, read_count, segments = reads.shape
        flattened = reads.reshape(batch * read_count, segments)
        encoded = self.embedding(flattened)
        positions = self.pos_embed(torch.arange(segments, device=reads.device))
        encoded = (encoded + positions.unsqueeze(0)).mean(dim=1)
        return encoded.reshape(batch, read_count, -1)


def _read_convolutions(
    read_dim: int,
    channels: tuple[int, ...] = (256, 512, 512, 512),
    kernel_size: int = 5,
    dropout: float = 0.2,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = read_dim
    for out_channels in channels:
        layers.extend(
            [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            ]
        )
        in_channels = out_channels
    return nn.Sequential(*layers)


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, read_dim: int = 128):
        super().__init__()
        self.encoder = ReadEncoder(vocab_size, read_dim)
        self.read_conv = _read_convolutions(read_dim)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, reads: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(reads).transpose(1, 2)
        hidden = self.read_conv(hidden).mean(dim=2)
        return self.classifier(hidden).squeeze(1)


class GeneFeatureEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.out_dim = hidden

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class HybridCNNClassifier(nn.Module):
    """CNN over 2bRAD reads fused with ten weighted Prokka gene features."""

    def __init__(self, vocab_size: int, read_dim: int = 128, feature_dim: int = 10):
        super().__init__()
        self.encoder = ReadEncoder(vocab_size, read_dim)
        self.read_conv = _read_convolutions(read_dim)
        self.read_pool = nn.AdaptiveAvgPool1d(1)
        self.tabular = GeneFeatureEncoder(feature_dim, hidden=16)
        fusion_dim = 512 + self.tabular.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 2, 1),
        )

    def forward(self, reads: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(reads).transpose(1, 2)
        hidden = self.read_pool(self.read_conv(hidden)).squeeze(-1)
        tabular = self.tabular(features)
        return self.classifier(torch.cat([hidden, tabular], dim=1)).squeeze(1)


def build_model(
    vocab_size: int,
    read_dim: int = 128,
    feature_dim: int | None = None,
) -> nn.Module:
    if feature_dim is None:
        return CNNClassifier(vocab_size, read_dim)
    return HybridCNNClassifier(vocab_size, read_dim, feature_dim)

