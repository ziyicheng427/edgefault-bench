"""Standard, compact, and lightweight domain-generalization 1D models."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )


class StandardCNN1D(nn.Module):
    """Conventional full-convolution reference with a 128-dimensional embedding."""

    def __init__(self, num_classes: int = 4, in_channels: int = 1):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, 15, stride=2),
            nn.MaxPool1d(2),
            ConvBlock(32, 64, 9, stride=2),
            nn.MaxPool1d(2),
            ConvBlock(64, 128, 7, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, signals: Tensor, *, return_embedding: bool = False):
        embedding = self.features(signals).squeeze(-1)
        logits = self.classifier(embedding)
        return (logits, embedding) if return_embedding else logits


class DepthwiseSeparableBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=9,
                stride=stride,
                padding=4,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )


class CompactDepthwiseCNN1D(nn.Module):
    """Edge-oriented depthwise-separable network with a 64-dimensional embedding."""

    def __init__(self, num_classes: int = 4, in_channels: int = 1):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        self.features = nn.Sequential(
            ConvBlock(in_channels, 16, 9, stride=2),
            nn.MaxPool1d(2),
            DepthwiseSeparableBlock(16, 24, stride=2),
            DepthwiseSeparableBlock(24, 40, stride=2),
            DepthwiseSeparableBlock(40, 64, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, signals: Tensor, *, return_embedding: bool = False):
        embedding = self.features(signals).squeeze(-1)
        logits = self.classifier(embedding)
        return (logits, embedding) if return_embedding else logits


def coral_loss(embeddings: Tensor, domains: Tensor) -> Tensor:
    """Average pairwise covariance discrepancy among source subdomains."""

    unique_domains = torch.unique(domains)
    covariances: list[Tensor] = []
    for domain in unique_domains:
        values = embeddings[domains == domain]
        if len(values) < 2:
            continue
        centered = values - values.mean(dim=0, keepdim=True)
        covariances.append(centered.T @ centered / (len(values) - 1))
    if len(covariances) < 2:
        return embeddings.sum() * 0.0
    losses: list[Tensor] = []
    dimension = embeddings.shape[1]
    for left_index, left in enumerate(covariances):
        for right in covariances[left_index + 1 :]:
            losses.append(torch.sum((left - right) ** 2) / (4.0 * dimension**2))
    return torch.stack(losses).mean()


def build_model(
    model_id: str, *, num_classes: int = 4, in_channels: int = 1
) -> nn.Module:
    if model_id == "standard_cnn_1d":
        return StandardCNN1D(num_classes=num_classes, in_channels=in_channels)
    if model_id in {"compact_depthwise_cnn_1d", "compact_coral_cnn_1d"}:
        return CompactDepthwiseCNN1D(
            num_classes=num_classes, in_channels=in_channels
        )
    raise ValueError(f"unknown model: {model_id!r}")


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
