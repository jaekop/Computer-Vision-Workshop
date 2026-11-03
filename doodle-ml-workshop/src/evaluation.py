"""Visual evaluation helpers. Tune on validation, not test."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


def validation_accuracy(model, X_val, y_val) -> float:
    return float(accuracy_score(y_val, model.predict(X_val)))


def show_confusion(model, X, y, title: str = "Validation confusion matrix"):
    predictions = model.predict(X)
    matrix = confusion_matrix(y, predictions, labels=np.arange(len(model.class_names)))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ConfusionMatrixDisplay(matrix, display_labels=model.class_names).plot(
        ax=ax, cmap="Purples", colorbar=False, xticks_rotation=40
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def show_examples(X, y, class_names, count: int = 12, seed: int = 42):
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(X), min(count, len(X)), replace=False)
    cols = 4
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.6 * rows))
    for ax in np.asarray(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.asarray(axes).ravel(), picks):
        ax.imshow(X[i].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        ax.set_title(class_names[int(y[i])])
    fig.tight_layout()
    return fig


def show_mistakes(model, X, y, count: int = 12):
    pred = model.predict(X)
    bad = np.flatnonzero(pred != y)[:count]
    if len(bad) == 0:
        print("No mistakes in this slice.")
        return None
    cols = 4
    rows = int(np.ceil(len(bad) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 2.6))
    for ax in np.asarray(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.asarray(axes).ravel(), bad):
        ax.imshow(X[i].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"True: {model.class_names[int(y[i])]}\n"
            f"Pred: {model.class_names[int(pred[i])]}", fontsize=9
        )
    fig.tight_layout()
    return fig

