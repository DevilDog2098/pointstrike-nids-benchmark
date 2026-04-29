from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureConfig:
    excluded_features: list[str]
    categorical_features: list[str]
    binary_passthrough_features: list[str]
    numeric_imputer_strategy: str
    categorical_imputer_strategy: str
    scaler_for_models: list[str]
    unscaled_models: list[str]


def build_feature_config(data_config: dict[str, Any]) -> FeatureConfig:
    """Construct the feature configuration object from YAML settings."""
    feature_policy = data_config["feature_policy"]
    return FeatureConfig(
        excluded_features=list(feature_policy["excluded_features"]),
        categorical_features=list(feature_policy["categorical_features"]),
        binary_passthrough_features=list(feature_policy["binary_passthrough_features"]),
        numeric_imputer_strategy=str(feature_policy["numeric_imputer_strategy"]),
        categorical_imputer_strategy=str(feature_policy["categorical_imputer_strategy"]),
        scaler_for_models=list(feature_policy["scaler_for_models"]),
        unscaled_models=list(feature_policy["unscaled_models"]),
    )


def infer_numeric_features(columns: list[str], config: FeatureConfig) -> list[str]:
    """Infer numeric feature columns after applying exclusions and known categorical lists."""
    excluded = set(config.excluded_features)
    categorical = set(config.categorical_features)
    binary = set(config.binary_passthrough_features)
    return [
        column
        for column in columns
        if column not in excluded and column not in categorical and column not in binary
    ]
