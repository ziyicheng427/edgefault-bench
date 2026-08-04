import numpy as np
import pytest

from edgefault_bench.features import (
    FEATURE_NAMES,
    extract_features,
    extract_multichannel_features,
    multichannel_feature_names,
)


def test_feature_shape_and_finiteness() -> None:
    time = np.arange(256) / 1_024.0
    windows = np.vstack(
        [np.sin(2 * np.pi * 40 * time), np.sin(2 * np.pi * 80 * time)]
    )
    features = extract_features(windows, sampling_rate=1_024.0)
    assert features.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert features[1, FEATURE_NAMES.index("spectral_centroid")] > features[
        0, FEATURE_NAMES.index("spectral_centroid")
    ]


def test_features_reject_invalid_sampling_rate() -> None:
    with pytest.raises(ValueError, match="positive"):
        extract_features(np.ones((2, 16)), sampling_rate=0)


def test_multichannel_features_are_channel_qualified_and_concatenated() -> None:
    base = np.linspace(-1.0, 1.0, 64)
    windows = np.stack(
        (
            np.stack((base, base * 2, base * -1)),
            np.stack((base + 1, base * 3, base * -2)),
        )
    )
    features = extract_multichannel_features(windows, sampling_rate=10_000.0)
    names = multichannel_feature_names(("x", "y", "z"))

    assert features.shape == (2, 30)
    assert len(names) == 30
    assert names[0] == "x_mean"
    assert names[10] == "y_mean"
    assert names[20] == "z_mean"
