from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


class SplitValidationError(ValueError):
    """Raised when split configuration values are invalid."""


def create_train_val_test_split(
    dataset: pd.DataFrame,
    target_column: str,
    splits_config: dict[str, Any],
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible stratified train/validation/test splits."""
    train_size = float(splits_config["train_size"])
    val_size = float(splits_config["val_size"])
    test_size = float(splits_config["test_size"])

    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise SplitValidationError(
            f"Split fractions must sum to 1.0, but summed to {total}."
        )

    stratify_labels = dataset[target_column] if splits_config.get("stratify", True) else None

    train_df, temp_df = train_test_split(
        dataset,
        test_size=(1.0 - train_size),
        random_state=random_seed,
        stratify=stratify_labels,
    )

    temp_target = temp_df[target_column] if splits_config.get("stratify", True) else None
    val_fraction_of_temp = val_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=random_seed,
        stratify=temp_target,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
