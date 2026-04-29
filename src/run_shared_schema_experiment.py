from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluate import evaluate_on_test_set
from src.experiment_logger import save_experiment_outputs
from src.preprocessing import build_preprocessor
from src.split_data import create_train_val_test_split
from src.train import fit_and_select_model
from src.utils import load_yaml, save_json


def run_shared_schema_experiment(
    experiment_id: str,
    model_key: str,
    candidate_configs: list[dict[str, Any]],
) -> None:
    """Run a shared-schema benchmark experiment on harmonized UNSW data."""
    project_root = Path(__file__).resolve().parents[1]

    data_config = load_yaml(project_root / "configs" / "data_config.yaml")
    experiment_config = load_yaml(project_root / "configs" / "experiment_config.yaml")

    shared_path = project_root / "data" / "shared_schema" / "unsw_shared_schema.csv"
    dataset = pd.read_csv(shared_path, low_memory=False)

    target_column = "Label"
    feature_columns = [column for column in dataset.columns if column != target_column]

    feature_config = _build_shared_feature_config(data_config)

    train_df, val_df, test_df = create_train_val_test_split(
        dataset=dataset,
        target_column=target_column,
        splits_config=data_config["splits"],
        random_seed=int(data_config["random_seed"]),
    )

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_val = val_df[feature_columns]
    y_val = val_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    preprocessor, numeric_features, categorical_features, binary_features = build_preprocessor(
        feature_columns=feature_columns,
        feature_config=feature_config,
        model_key=model_key,
    )

    best_result = fit_and_select_model(
        model_key=model_key,
        candidate_configs=candidate_configs,
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    test_metrics = evaluate_on_test_set(best_result, X_test, y_test)

    defaults = experiment_config["experiment_defaults"]
    paths = {
        "metrics_dir": str(project_root / defaults["metrics_dir"] / "shared_schema"),
        "model_dir": str(project_root / defaults["model_dir"] / "shared_schema"),
        "log_dir": str(project_root / defaults["log_dir"] / "shared_schema"),
    }
    save_experiment_outputs(experiment_id, best_result, test_metrics, paths)

    manifest = {
        "experiment_id": experiment_id,
        "model_key": model_key,
        "dataset_path": str(shared_path),
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "binary_features": binary_features,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "selected_params": best_result["params"],
    }
    save_json(
        manifest,
        project_root / "outputs" / "logs" / "shared_schema" / f"{experiment_id.lower()}_manifest.json",
    )

    print(f"{experiment_id} complete.")
    print("Selected parameters:", best_result["params"])
    print("Validation metrics:", best_result["validation_metrics"])
    print("Test metrics:", test_metrics)


def _build_shared_feature_config(data_config: dict[str, Any]):
    """Build a minimal feature config object for the shared-schema datasets."""
    from src.feature_config import FeatureConfig

    return FeatureConfig(
        excluded_features=["Label"],
        categorical_features=[],
        binary_passthrough_features=[],
        numeric_imputer_strategy=data_config["feature_policy"]["numeric_imputer_strategy"],
        categorical_imputer_strategy=data_config["feature_policy"]["categorical_imputer_strategy"],
        scaler_for_models=list(data_config["feature_policy"]["scaler_for_models"]),
        unscaled_models=list(data_config["feature_policy"]["unscaled_models"]),
    )