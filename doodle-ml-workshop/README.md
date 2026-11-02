# Build Your Own Doodle AI

**Classical machine learning: from pixels to predictions.**
A beginner-friendly, 60-minute UCSB Data Science Club workshop built around
one doodle recognizer and three fully worked teaching notebooks.

## Open in Google Colab

| Notebook | Topic | Colab |
| --- | --- | --- |
| Method 1 | KNN and Decision Trees | [Open in Colab](https://colab.research.google.com/github/jaekop/Computer-Vision-Workshop/blob/doodle-ai-workshop/doodle-ml-workshop/notebooks/method_1_similarity_rules.ipynb) |
| Method 2 | Random Forests and Features | [Open in Colab](https://colab.research.google.com/github/jaekop/Computer-Vision-Workshop/blob/doodle-ai-workshop/doodle-ml-workshop/notebooks/method_2_forests_features.ipynb) |
| Method 3 | SVM and Evaluation | [Open in Colab](https://colab.research.google.com/github/jaekop/Computer-Vision-Workshop/blob/doodle-ai-workshop/doodle-ml-workshop/notebooks/method_3_svm_evaluation.ipynb) |
| Final Challenge | Build your classifier | [Open in Colab](https://colab.research.google.com/github/jaekop/Computer-Vision-Workshop/blob/doodle-ai-workshop/doodle-ml-workshop/notebooks/final_challenge.ipynb) |

## How it works

GitHub holds the notebooks and shared code. Colab runs Python and scikit-learn.
Gradio provides a drawing canvas for the model trained in that SAME runtime.
An optional separately hosted demo is for the opening hook only.

The official Google Quick, Draw! numpy bitmap source is streamed by reading
only a small prefix of each of six categories: airplane, bicycle, cat, fish,
house and tree. No massive dataset download is necessary. A cached .npz file
is created on first run. Training uses 600 examples/category, validation 150,
and sealed test 150. All notebooks share identical splits (seed 42).

**Data caveat:** this is a stratified random split of a deterministic *prefix*
of each class file, not a random sample of the entire Quick, Draw! collection.
Use results as workshop demonstrations, not published performance benchmarks.

## Presenter quick start

1. Open each Method notebook and run all cells ahead of the event.
2. Run the final notebook in a fresh Colab runtime and confirm its Gradio app.
3. Locally, from this folder: python -m pip install -r requirements.txt
4. Optionally cache data: python -m scripts.prepare_dataset
5. Run measured validation baselines: python -m scripts.benchmark
6. Optional hosted demo: python -m scripts.train_demo and follow demo/README.md
7. Replace any unmeasured accuracy milestone with a calibrated one.

## Workshop schedule

0–5 demo · 5–15 foundations · 15–20 Become the Classifier ·
20–28 Method 1 · 28–36 Method 2 · 36–41 Fool the Classifier ·
41–48 Method 3 · 48–53 advanced ideas · 53–60 final challenge.

## Student workflow

- Run each premade Method notebook with Shift+Enter.
- Try the suggested parameter changes and inspect mistakes.
- In Final Challenge, use Guided Mode: copy working code from the Methods.
- Or use Challenge Mode: change models, features and parameters yourself.
- Tune using validation data only. Reveal the test score once at the end.
- Launch Gradio from YOUR Colab to test your own model on drawings.

## Local use

From the workshop folder:

    python -m pip install -r requirements.txt
    python -m scripts.prepare_dataset
    python -m pytest -q
    jupyter notebook

The notebooks auto-detect a local checkout or clone this branch into Colab.

## Slides

[Editable workshop slides](https://docs.google.com/presentation/d/1eR-EpH2LtQqkN51yrW1VAq0jIObyY4FToDJKQyYFf5w/edit)

## Important before the event

The code is provided as a complete teaching implementation but has **not yet
been executed in a live Colab runtime by the assistant**. Presenter must run
the checklist in PLAN.md, verify Gradio and Google data access, and measure
validation scores. The hosted demo model is not bundled.

## Credits

See ATTRIBUTION.md for Quick, Draw! licensing and official docs.

