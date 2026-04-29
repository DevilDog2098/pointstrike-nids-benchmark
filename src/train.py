from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from src.metrics import compute_classification_metrics, select_best_result
from src.model_registry import build_model


def fit_and_select_model(
    model_key: str,
    candidate_configs: list[dict[str, Any]],
    preprocessor: Any,
    X_train,
    y_train,
    X_val,
    y_val,
) -> dict[str, Any]:
    """Train all candidate configurations and select the best validation result."""
    candidate_results: list[dict[str, Any]] = []

    for candidate in candidate_configs:
        model = build_model(model_key, candidate)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        start_train = perf_counter()
        pipeline.fit(X_train, y_train)
        training_time_seconds = perf_counter() - start_train

        start_inference = perf_counter()
        y_val_pred = pipeline.predict(X_val)
        inference_time_seconds = perf_counter() - start_inference

        y_val_score = _get_score_vector(pipeline, X_val)
        validation_metrics = compute_classification_metrics(
            np.asarray(y_val),
            np.asarray(y_val_pred),
            None if y_val_score is None else np.asarray(y_val_score),
        )
        validation_metrics["training_time_seconds"] = float(training_time_seconds)
        validation_metrics["inference_time_seconds"] = float(inference_time_seconds)

        candidate_results.append(
            {
                "model_key": model_key,
                "params": candidate,
                "pipeline": pipeline,
                "validation_metrics": validation_metrics,
            }
        )

    return select_best_result(candidate_results)


def _get_score_vector(pipeline: Pipeline, X):
    """Return a score vector when supported by the estimator."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        probabilities = pipeline.predict_proba(X)
        return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        return pipeline.decision_function(X)
    return None