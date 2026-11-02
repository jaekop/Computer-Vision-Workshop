"""Organizer only: train one demo model and serialize for Hugging Face Spaces."""
from pathlib import Path
import joblib

from src.data import load_data
from src.models import fit_model
from src.evaluation import validation_accuracy


def main():
    data = load_data(cache_dir="data")
    model = fit_model(data.X_train, data.y_train, "svm", "hog")
    print("Validation accuracy:", round(validation_accuracy(model, data.X_val, data.y_val), 4))
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/demo_model.joblib")
    print("Saved models/demo_model.joblib. Only load models you trust.")


if __name__ == "__main__":
    main()

