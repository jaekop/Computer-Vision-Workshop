"""Organizer: prepare and cache the small official Quick, Draw! subset."""
import argparse
from pathlib import Path

from src.data import load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    data = load_data(cache_dir=Path(args.cache_dir), force_download=args.force)
    print("Prepared:", data.class_names)
    print("Train:", data.X_train.shape, "Val:", data.X_val.shape, "Test:", data.X_test.shape)
    print("Cache:", Path(args.cache_dir) / "quickdraw_subset_v1.npz")


if __name__ == "__main__":
    main()

