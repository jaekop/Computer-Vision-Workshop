import numpy as np
import pytest

from src.features import build_features, center_ink


def test_pixels_and_hog_shapes():
    X = np.zeros((3, 784), dtype=np.uint8)
    X[:, 10:20] = 255
    assert build_features(X, "pixels").shape == (3, 784)
    H = build_features(X, "hog")
    assert H.shape[0] == 3
    assert np.isfinite(H).all()


def test_center_ink_no_wrap():
    X = np.zeros((1, 28, 28), dtype=np.uint8)
    X[0, 0, 0] = 255
    centered = center_ink(X.reshape(1, 784)).reshape(28, 28)
    assert centered.sum() == 255
    assert centered[13:15, 13:15].max() == 255


def test_bad_representation():
    with pytest.raises(ValueError):
        build_features(np.zeros((1, 784), dtype=np.uint8), "magic")

