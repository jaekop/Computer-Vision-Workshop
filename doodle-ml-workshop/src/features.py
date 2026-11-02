"""Pixel and HOG feature pipelines used identically for training and drawing."""
import numpy as np
from skimage.feature import hog

from .config import IMAGE_SIZE


def center_ink(X: np.ndarray, threshold: int = 12) -> np.ndarray:
    """Translate each doodle's non-background bounding box toward the center.

    Zero-fill rather than wrap around. This is not rotation-invariant.
    """
    arr = np.asarray(X, dtype=np.uint8).reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    output = np.zeros_like(arr)
    for i, img in enumerate(arr):
        rows, cols = np.where(img > threshold)
        if not len(rows):
            continue
        dy = int(round((IMAGE_SIZE - 1 - (int(rows.min()) + int(rows.max()))) / 2))
        dx = int(round((IMAGE_SIZE - 1 - (int(cols.min()) + int(cols.max()))) / 2))
        ys0, ys1 = max(0, -dy), min(IMAGE_SIZE, IMAGE_SIZE - dy)
        xs0, xs1 = max(0, -dx), min(IMAGE_SIZE, IMAGE_SIZE - dx)
        output[i, ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx] = img[ys0:ys1, xs0:xs1]
    return output.reshape(-1, IMAGE_SIZE * IMAGE_SIZE)


def build_features(
    X: np.ndarray, representation: str = "pixels", center: bool = False
) -> np.ndarray:
    """Transform 784 uint8 pixels into normalized raw-pixel or HOG features."""
    raw = np.asarray(X, dtype=np.uint8).reshape(-1, IMAGE_SIZE ** 2)
    if center:
        raw = center_ink(raw)
    images = raw.astype(np.float32).reshape(-1, IMAGE_SIZE, IMAGE_SIZE) / 255.0
    if representation == "pixels":
        return images.reshape(len(images), -1)
    if representation == "hog":
        return np.stack([
            hog(img, orientations=9, pixels_per_cell=(4, 4),
                cells_per_block=(2, 2), block_norm="L2-Hys",
                feature_vector=True)
            for img in images
        ]).astype(np.float32)
    raise ValueError("representation must be 'pixels' or 'hog'")

