"""Small, documented signal-feature baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import kurtosis, skew

FEATURE_NAMES = (
    "mean",
    "standard_deviation",
    "root_mean_square",
    "peak_to_peak",
    "crest_factor",
    "skewness",
    "kurtosis",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_entropy",
)


def multichannel_feature_names(channel_names: tuple[str, ...]) -> tuple[str, ...]:
    """Return stable channel-qualified names for concatenated signal features."""

    if not channel_names or len(set(channel_names)) != len(channel_names):
        raise ValueError("channel_names must contain unique names")
    return tuple(
        f"{channel}_{feature}"
        for channel in channel_names
        for feature in FEATURE_NAMES
    )


def extract_features(windows: ArrayLike, *, sampling_rate: float) -> NDArray[np.float64]:
    """Extract time- and frequency-domain features from ``(samples, time)`` windows."""

    signals = np.asarray(windows, dtype=np.float64)
    if signals.ndim != 2 or signals.shape[1] < 4:
        raise ValueError("windows must have shape (samples, time) with at least four time points")
    if not np.isfinite(signals).all():
        raise ValueError("windows contain non-finite values")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")

    epsilon = np.finfo(np.float64).eps
    mean = signals.mean(axis=1)
    standard_deviation = signals.std(axis=1)
    root_mean_square = np.sqrt(np.mean(np.square(signals), axis=1))
    peak_to_peak = np.ptp(signals, axis=1)
    crest_factor = np.max(np.abs(signals), axis=1) / np.maximum(root_mean_square, epsilon)
    skewness = np.nan_to_num(skew(signals, axis=1, bias=False), nan=0.0)
    excess_kurtosis = np.nan_to_num(kurtosis(signals, axis=1, bias=False), nan=0.0)

    spectrum = np.abs(np.fft.rfft(signals, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(signals.shape[1], d=1.0 / sampling_rate)
    spectral_mass = np.maximum(spectrum.sum(axis=1), epsilon)
    probabilities = spectrum / spectral_mass[:, None]
    spectral_centroid = np.sum(probabilities * frequencies[None, :], axis=1)
    spectral_bandwidth = np.sqrt(
        np.sum(probabilities * (frequencies[None, :] - spectral_centroid[:, None]) ** 2, axis=1)
    )
    spectral_entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, epsilon)), axis=1
    ) / np.log(probabilities.shape[1])

    return np.column_stack(
        (
            mean,
            standard_deviation,
            root_mean_square,
            peak_to_peak,
            crest_factor,
            skewness,
            excess_kurtosis,
            spectral_centroid,
            spectral_bandwidth,
            spectral_entropy,
        )
    )


def extract_multichannel_features(
    windows: ArrayLike, *, sampling_rate: float
) -> NDArray[np.float64]:
    """Concatenate the documented features for ``(samples, channels, time)``."""

    signals = np.asarray(windows, dtype=np.float64)
    if signals.ndim != 3 or signals.shape[1] == 0:
        raise ValueError("windows must have shape (samples, channels, time)")
    return np.hstack(
        [
            extract_features(signals[:, channel], sampling_rate=sampling_rate)
            for channel in range(signals.shape[1])
        ]
    )
