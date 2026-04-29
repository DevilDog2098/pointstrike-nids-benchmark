from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from src.metrics import compute_classification_metrics


def evaluate_on_test_set(best_result: dict[str, Any], X_test, y_test) -> dict[str, Any]:
    """Evaluate the selected pipeline on the held-out test set."""
    pipeline = best_result["pipeline"]

    start_inference = perf_counter()
    y_test_pred = pipeline.predict(X_test)
    inference_time_seconds = perf_counter() - start_inference

    y_test_score = _get_score_vector(pipeline, X_test)
    test_metrics = compute_classification_metrics(
        np.asarray(y_test),
        np.asarray(y_test_pred),
        None if y_test_score is None else np.asarray(y_test_score),
    )
    test_metrics["inference_time_seconds"] = float(inference_time_seconds)
    return test_metrics


def _get_score_vector(pipeline, X):
    """Return a score vector when supported by the estimator."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        probabilities = pipeline.predict_proba(X)
        return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        return pipeline.decision_function(X)
    return None
