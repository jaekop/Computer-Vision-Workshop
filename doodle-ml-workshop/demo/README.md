# Hosted opening demo (optional)

This folder contains the Gradio Space entry point. A separate hosted demo is
optional and cannot access models trained in a student's Colab session.

Organizer workflow:
1. From the repository root: python -m scripts.prepare_dataset
2. From the repository root: python -m scripts.train_demo
3. Upload demo/app.py as app.py, src/, requirements.txt, and the generated
   models/demo_model.joblib to a Hugging Face Gradio Space.
4. Ensure the Space app imports src modules at the expected paths.
5. Test the hosted canvas with black strokes on a white background.

The hosted model artifact is intentionally not committed: it must be generated
and tested by the organizer. Joblib files are executable during loading; only
load artifacts created by a trusted source.

