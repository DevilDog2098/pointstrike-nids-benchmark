from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_config import FeatureConfig, infer_numeric_features


def build_preprocessor(
    feature_columns: list[str],
    feature_config: FeatureConfig,
    model_key: str,
) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    """Build a leakage-safe preprocessing transformer for a specific model family."""
    categorical_features = [
        column for column in feature_config.categorical_features if column in feature_columns
    ]
    binary_features = [
        column for column in feature_config.binary_passthrough_features if column in feature_columns
    ]
    numeric_features = infer_numeric_features(feature_columns, feature_config)

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=feature_config.numeric_imputer_strategy))
    ]
    if model_key in feature_config.scaler_for_models:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=feature_config.categorical_imputer_strategy)),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    binary_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
            ("binary", binary_pipeline, binary_features),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_features, categorical_features, binary_features
