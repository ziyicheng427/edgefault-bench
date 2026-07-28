"""Deterministic post-split sampling and signal-noise transformations."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


def stratified_fraction_indices(
    indices: Tensor,
    targets: Tensor,
    domains: Tensor,
    *,
    fraction: float,
    seed: int,
    target_class: int | None = None,
) -> Tensor:
    """Subsample within class-domain strata after partition assignment.

    If ``target_class`` is set, only that class is reduced and every other index is retained.
    At least one example is kept in each affected non-empty stratum.
    """

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    source = indices.detach().cpu().numpy().astype(np.int64, copy=False)
    labels = targets[indices].detach().cpu().numpy()
    groups = domains[indices].detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in sorted(np.unique(labels)):
        for domain in sorted(np.unique(groups[labels == label])):
            mask = (labels == label) & (groups == domain)
            stratum = source[mask]
            if target_class is not None and label != target_class:
                selected.extend(stratum.tolist())
                continue
            count = max(1, math.floor(len(stratum) * fraction))
            selected.extend(rng.choice(stratum, size=count, replace=False).tolist())
    return torch.tensor(sorted(selected), dtype=torch.long)


def add_awgn(signals: Tensor, *, snr_db: float, seed: int) -> Tensor:
    """Add deterministic Gaussian noise at a per-window SNR, then re-standardize."""

    if signals.ndim != 3:
        raise ValueError("signals must have shape [batch, channel, samples]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(signals.shape, generator=generator, dtype=signals.dtype)
    signal_power = signals.square().mean(dim=(-2, -1), keepdim=True)
    noise_scale = torch.sqrt(signal_power / (10.0 ** (snr_db / 10.0)))
    noisy = signals + noise * noise_scale
    centered = noisy - noisy.mean(dim=-1, keepdim=True)
    scale = centered.std(dim=-1, keepdim=True, correction=0).clamp_min(1e-8)
    return centered / scale
