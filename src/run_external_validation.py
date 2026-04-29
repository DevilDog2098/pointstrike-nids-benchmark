from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.metrics import compute_classification_metrics
from src.utils import save_json


class ExternalValidationError(RuntimeError):
    """Raised when external validation cannot be completed."""


MODEL_PATHS = {
    "random_forest": "outputs/models/shared_schema/ext-unsw-001_pipeline.joblib",
    "xgboost": "outputs/models/shared_schema/ext-unsw-002_pipeline.joblib",
    "mlp": "outputs/models/shared_schema/ext-unsw-003_pipeline.joblib",
}


def load_shared_cic_dataset(project_root: Path) -> pd.DataFrame:
    """Load the harmonized CIC-IDS2017 shared-schema dataset."""
    dataset_path = project_root / "data" / "shared_schema" / "cic_ids2017_shared_schema.csv"
    if not dataset_path.exists():
        raise ExternalValidationError(f"Shared CIC dataset not found: {dataset_path}")

    dataset = pd.read_csv(dataset_path, low_memory=False)
    required_columns = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "smean", "dmean", "rate", "Label"]
    missing = [column for column in required_columns if column not in dataset.columns]
    if missing:
        raise ExternalValidationError(f"CIC shared dataset missing required columns: {missing}")

    numeric_columns = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "smean", "dmean", "rate", "Label"]
    for column in numeric_columns:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    before_rows = len(dataset)
    dataset = dataset.dropna(subset=numeric_columns).copy()
    after_rows = len(dataset)

    print(f"CIC shared dataset rows before cleaning: {before_rows}")
    print(f"CIC shared dataset rows after cleaning: {after_rows}")

    dataset["Label"] = dataset["Label"].astype(int)
    return dataset


def load_trained_pipeline(project_root: Path, model_key: str):
    """Load a saved shared-schema trained pipeline."""
    if model_key not in MODEL_PATHS:
        raise ExternalValidationError(f"Unsupported model key: {model_key}")

    model_path = project_root / MODEL_PATHS[model_key]
    if not model_path.exists():
        raise ExternalValidationError(f"Saved model not found: {model_path}")

    return joblib.load(model_path), model_path


def evaluate_external_model(project_root: Path, model_key: str) -> dict[str, Any]:
    """Evaluate one saved shared-schema model on the CIC shared-schema dataset."""
    dataset = load_shared_cic_dataset(project_root)
    pipeline, model_path = load_trained_pipeline(project_root, model_key)

    feature_columns = [column for column in dataset.columns if column != "Label"]
    X = dataset[feature_columns]
    y = dataset["Label"]

    start_inference = perf_counter()
    y_pred = pipeline.predict(X)
    inference_time_seconds = perf_counter() - start_inference

    y_score = _get_score_vector(pipeline, X)
    metrics = compute_classification_metrics(
        np.asarray(y),
        np.asarray(y_pred),
        None if y_score is None else np.asarray(y_score),
    )
    metrics["inference_time_seconds"] = float(inference_time_seconds)

    return {
        "model_key": model_key,
        "model_path": str(model_path),
        "feature_columns": feature_columns,
        "row_count": int(len(dataset)),
        "metrics": metrics,
    }


def _get_score_vector(pipeline, X):
    """Return a score vector when supported by the estimator."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        probabilities = pipeline.predict_proba(X)
        return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        return pipeline.decision_function(X)
    return None


def main() -> None:
    project_root = PROJECT_ROOT
    output_dir = project_root / "outputs" / "logs" / "external_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_key in ["random_forest", "xgboost", "mlp"]:
        result = evaluate_external_model(project_root, model_key)
        save_json(result, output_dir / f"{model_key}_cic_ids2017_external_validation.json")

        print("=" * 80)
        print(f"EXTERNAL VALIDATION COMPLETE: {model_key}")
        print("=" * 80)
        print("Model path:", result["model_path"])
        print("Rows scored:", result["row_count"])
        print("Metrics:", result["metrics"])
        print()


if __name__ == "__main__":
    main()