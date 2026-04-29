from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data_loader import load_unsw_nb15
from src.evaluate import evaluate_on_test_set
from src.experiment_logger import save_experiment_outputs
from src.feature_config import build_feature_config
from src.preprocessing import build_preprocessor
from src.split_data import create_train_val_test_split
from src.train import fit_and_select_model
from src.utils import load_yaml, save_json


def run_experiment(
    experiment_id: str,
    model_key: str,
    candidate_configs: list[dict[str, Any]],
) -> None:
    """Run a full benchmark experiment from data load through evaluation."""
    project_root = Path(__file__).resolve().parents[1]

    data_config = load_yaml(project_root / "configs" / "data_config.yaml")
    experiment_config = load_yaml(project_root / "configs" / "experiment_config.yaml")

    dataset = load_unsw_nb15(data_config)
    print("Loaded dataset shape:", dataset.shape)
    print("Columns:", list(dataset.columns))
    print("Label distribution:")
    print(dataset["Label"].value_counts(dropna=False))
    feature_config = build_feature_config(data_config)

    dataset = dataset.drop_duplicates().reset_index(drop=True)

    target_column = data_config["dataset"]["target_column"]
    train_df, val_df, test_df = create_train_val_test_split(
        dataset=dataset,
        target_column=target_column,
        splits_config=data_config["splits"],
        random_seed=int(data_config["random_seed"]),
    )

    feature_columns = [
        column for column in dataset.columns if column not in feature_config.excluded_features
    ]
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
        "metrics_dir": str(project_root / defaults["metrics_dir"]),
        "model_dir": str(project_root / defaults["model_dir"]),
        "log_dir": str(project_root / defaults["log_dir"]),
    }
    save_experiment_outputs(experiment_id, best_result, test_metrics, paths)

    manifest = {
        "experiment_id": experiment_id,
        "model_key": model_key,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "binary_features": binary_features,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "selected_params": best_result["params"],
    }
    save_json(manifest, project_root / "outputs" / "logs" / f"{experiment_id.lower()}_manifest.json")

    print(f"{experiment_id} complete.")
    print("Selected parameters:", best_result["params"])
    print("Validation metrics:", best_result["validation_metrics"])
    print("Test metrics:", test_metrics)