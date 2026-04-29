from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute the benchmark metrics used in the paper."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

    metrics: dict[str, Any] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(false_positive_rate),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    if y_score is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))

    return metrics


def select_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the best configuration using the paper's metric hierarchy."""
    if not results:
        raise ValueError("No candidate results were provided for selection.")

    return sorted(
        results,
        key=lambda item: (
            -item["validation_metrics"]["recall"],
            item["validation_metrics"]["false_positive_rate"],
            -item["validation_metrics"]["precision"],
            -item["validation_metrics"]["f1"],
        ),
    )[0]
