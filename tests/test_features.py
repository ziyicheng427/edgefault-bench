import numpy as np
import pytest

from edgefault_bench.features import FEATURE_NAMES, extract_features


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

