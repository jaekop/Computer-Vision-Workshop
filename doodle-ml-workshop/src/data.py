"""Small, deterministic Quick, Draw! bitmap subset; downloads only needed bytes.

The official numpy_bitmap files are uncompressed .npy arrays. We stream the
header and the first SAMPLES_PER_CLASS rows rather than downloading each
entire category. This avoids hundreds of megabytes of workshop downloads.
"""
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np

from .config import (
    CLASSES, DEFAULT_CACHE, IMAGE_SIZE, SEED, SAMPLES_PER_CLASS,
    TRAIN_PER_CLASS, VAL_PER_CLASS, TEST_PER_CLASS,
)

SOURCE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"


@dataclass
class DoodleData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: tuple[str, ...]


def fetch_category_prefix(category: str, n: int = SAMPLES_PER_CLASS) -> np.ndarray:
    """Read just n 28x28 bitmaps from Google's official Quick, Draw! .npy."""
    if category not in CLASSES:
        raise ValueError(f"Unknown category: {category!r}")
    url = SOURCE.format(quote(category, safe=""))
    # Explicit range helps some proxies; read() still limits transfer if
    # the origin ignores Range.
    req = Request(url, headers={"Range": f"bytes=0-{n * 784 + 4095}"})
    with urlopen(req, timeout=90) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported .npy version {version} for {category}")
        if len(shape) != 2 or shape[1] != IMAGE_SIZE ** 2:
            raise ValueError(f"Unexpected Quick Draw shape: {shape}")
        if fortran or np.dtype(dtype) != np.dtype("uint8"):
            raise ValueError(f"Unexpected Quick Draw encoding: {dtype}, fortran={fortran}")
        if shape[0] < n:
            raise ValueError(f"Not enough samples for {category}")
        count = n * IMAGE_SIZE * IMAGE_SIZE
        payload = stream.read(count)
    if len(payload) != count:
        raise IOError(f"Incomplete download for {category}: {len(payload)} / {count}")
    return np.frombuffer(payload, dtype=np.uint8).copy().reshape(n, -1)


def build_dataset() -> DoodleData:
    """Build stratified splits from a deterministic, per-category prefix.

    We sample from the first 900 rows of each category, not a random sample of
    Google's full dataset. Results are useful for teaching, not a published
    benchmark. Never use the test split for hyperparameter selection.
    """
    rng = np.random.default_rng(SEED)
    buckets = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for label, category in enumerate(CLASSES):
        images = fetch_category_prefix(category)
        indices = rng.permutation(len(images))
        starts = (0, TRAIN_PER_CLASS, TRAIN_PER_CLASS + VAL_PER_CLASS)
        sizes = (TRAIN_PER_CLASS, VAL_PER_CLASS, TEST_PER_CLASS)
        for split, start, size in zip(("train", "val", "test"), starts, sizes):
            xs, ys = buckets[split]
            xs.append(images[indices[start:start + size]])
            ys.append(np.full(size, label, dtype=np.int64))
    assembled = {}
    for split, (xs, ys) in buckets.items():
        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        order = rng.permutation(len(y))
        assembled[split] = (X[order], y[order])
    return DoodleData(
        *assembled["train"], *assembled["val"], *assembled["test"], CLASSES
    )


def load_data(cache_dir: str | Path | None = None, force_download: bool = False) -> DoodleData:
    """Load a local organizer-prepared .npz, or stream a tiny official subset."""
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
    path = root / "quickdraw_subset_v1.npz"
    if path.exists() and not force_download:
        with np.load(path, allow_pickle=False) as archive:
            names = tuple(str(s) for s in archive["class_names"])
            if names != CLASSES:
                raise ValueError("Cached class names differ from src.config.CLASSES")
            return DoodleData(
                archive["X_train"], archive["y_train"],
                archive["X_val"], archive["y_val"],
                archive["X_test"], archive["y_test"], names
            )
    root.mkdir(parents=True, exist_ok=True)
    data = build_dataset()
    np.savez_compressed(
        path, X_train=data.X_train, y_train=data.y_train,
        X_val=data.X_val, y_val=data.y_val,
        X_test=data.X_test, y_test=data.y_test,
        class_names=np.asarray(data.class_names),
    )
    return data

