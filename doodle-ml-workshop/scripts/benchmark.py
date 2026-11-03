"""Reproducible VALIDATION benchmark; never tune on the held-out test split."""
import json
from pathlib import Path

from src.data import load_data
from src.models import fit_model
from src.evaluation import validation_accuracy


def main():
    data = load_data(cache_dir="data")
    variants = [
        ("knn_pixels", "knn", "pixels", False, {}),
        ("tree_pixels", "tree", "pixels", False, {}),
        ("forest_pixels", "forest", "pixels", False, {}),
        ("forest_centered", "forest", "pixels", True, {}),
        ("svm_hog", "svm", "hog", False, {}),
    ]
    results = {}
    for label, name, representation, center, params in variants:
        model = fit_model(data.X_train, data.y_train, name, representation, center, **params)
        score = validation_accuracy(model, data.X_val, data.y_val)
        results[label] = {"validation_accuracy": score,
                          "model": name, "representation": representation,
                          "center": center}
        print(f"{label:20s} validation accuracy = {score:.3f}")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/validation_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print("Saved reports/validation_results.json; TEST split remains untouched.")


if __name__ == "__main__":
    main()

