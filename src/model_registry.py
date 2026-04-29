from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


class ModelRegistryError(RuntimeError):
    """Raised when a requested model cannot be created."""


def build_model(model_key: str, params: dict[str, Any]) -> Any:
    """Construct a classifier instance for the requested model family."""
    if model_key == "logistic_regression":
        return LogisticRegression(**params)

    if model_key == "random_forest":
        return RandomForestClassifier(**params)

    if model_key == "gradient_boosting":
        return GradientBoostingClassifier(**params)

    if model_key == "xgboost":
        if XGBClassifier is None:
            raise ModelRegistryError(
                "xgboost is not installed. Install it or switch to gradient_boosting."
            )
        safe_params = {
            **params,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        return XGBClassifier(**safe_params)

    if model_key == "svm":
        return SVC(**params)

    if model_key == "mlp":
        hidden_layer_sizes = params.get("hidden_layer_sizes")
        if isinstance(hidden_layer_sizes, list):
            params = {**params, "hidden_layer_sizes": tuple(hidden_layer_sizes)}
        return MLPClassifier(**params)

    raise ModelRegistryError(f"Unsupported model key: {model_key}")