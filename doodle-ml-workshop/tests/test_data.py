import io
from unittest.mock import patch

import numpy as np

from src.data import build_dataset, fetch_category_prefix
from src.config import CLASSES


class Stream(io.BytesIO):
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()


def test_partial_npy_streaming():
    X = np.arange(30 * 784, dtype=np.uint32).reshape(30, 784).astype(np.uint8)
    payload = io.BytesIO()
    np.save(payload, X)
    with patch("src.data.urlopen", side_effect=lambda *args, **kwargs: Stream(payload.getvalue())):
        result = fetch_category_prefix("cat", n=7)
    assert result.shape == (7, 784)
    np.testing.assert_array_equal(result, X[:7])


def test_reproducible_stratified_splits():
    fake = np.arange(900 * 784, dtype=np.uint32).reshape(900, 784).astype(np.uint8)
    with patch("src.data.fetch_category_prefix", return_value=fake):
        a = build_dataset()
        b = build_dataset()
    assert a.X_train.shape == (len(CLASSES) * 600, 784)
    assert a.X_val.shape == (len(CLASSES) * 150, 784)
    assert a.X_test.shape == (len(CLASSES) * 150, 784)
    np.testing.assert_array_equal(a.y_train, b.y_train)
    for labels, count in [(a.y_train, 600), (a.y_val, 150), (a.y_test, 150)]:
        np.testing.assert_array_equal(np.bincount(labels), [count] * len(CLASSES))
