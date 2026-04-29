from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.utils import ensure_dir, save_json


def save_experiment_outputs(
    experiment_id: str,
    best_result: dict[str, Any],
    test_metrics: dict[str, Any],
    paths: dict[str, str],
) -> None:
    """Save model artifacts and metric outputs for an experiment."""
    metrics_dir = ensure_dir(paths["metrics_dir"])
    model_dir = ensure_dir(paths["model_dir"])
    log_dir = ensure_dir(paths["log_dir"])

    metrics_payload = {
        "experiment_id": experiment_id,
        "selected_model_key": best_result["model_key"],
        "selected_params": best_result["params"],
        "validation_metrics": best_result["validation_metrics"],
        "test_metrics": test_metrics,
    }

    save_json(metrics_payload, metrics_dir / f"{experiment_id.lower()}_metrics.json")
    save_json(metrics_payload, log_dir / f"{experiment_id.lower()}_summary.json")

    model_path = Path(model_dir) / f"{experiment_id.lower()}_pipeline.joblib"
    joblib.dump(best_result["pipeline"], model_path)
