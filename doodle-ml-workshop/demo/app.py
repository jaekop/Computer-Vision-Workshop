"""Hugging Face Space entry point. Deploy with a pre-trained trusted model."""
from pathlib import Path
import sys

import joblib

HERE = Path(__file__).resolve()
# Works both as demo/app.py in this repo and as app.py at Space root.
ROOT = HERE.parent if (HERE.parent / "src").is_dir() else HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.playground import launch_playground

MODEL_PATH = ROOT / "models" / "demo_model.joblib"
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Demo model missing. Run 'python -m scripts.train_demo' from repo root "
        "and deploy models/demo_model.joblib with this app."
    )

model = joblib.load(MODEL_PATH)  # Never load untrusted joblib files.
launch_playground({"Demo — HOG + SVM": model}, share=False)
