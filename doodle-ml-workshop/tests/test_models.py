import numpy as np

from src.models import fit_model
from src.evaluation import validation_accuracy


def test_knn_model_end_to_end():
    X = np.zeros((20, 784), dtype=np.uint8)
    X[:10, :20] = 255
    X[10:, 20:40] = 255
    y = np.array([0] * 10 + [1] * 10)
    model = fit_model(X, y, "knn", n_neighbors=3)
    assert model.predict(X[:1]).shape == (1,)
    assert validation_accuracy(model, X, y) > 0.9
    scores = model.ranked_scores(X[:1])
    assert len(scores) == 2
    assert scores[0][0] in ("airplane", "bicycle")
