from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class DatasetLoadError(RuntimeError):
    """Raised when the dataset cannot be loaded as expected."""


EXPECTED_UNSW_COLUMNS = [
    "id",
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
    "attack_cat",
    "Label",
]


def load_unsw_nb15(data_config: dict[str, Any]) -> pd.DataFrame:
    """Load the configured UNSW-NB15 benchmark CSV files."""
    dataset_cfg = data_config["dataset"]
    raw_dir = Path(dataset_cfg["raw_data_dir"])

    csv_paths = sorted(raw_dir.glob("*.csv"))
    if not csv_paths:
        raise DatasetLoadError(f"No CSV files found in '{raw_dir}'.")

    allowed_names = {
        "UNSW_NB15_training-set.csv",
        "UNSW_NB15_testing-set.csv",
    }

    selected_paths = [p for p in csv_paths if p.name in allowed_names]
    if not selected_paths:
        raise DatasetLoadError(
            "Could not find UNSW_NB15_training-set.csv or UNSW_NB15_testing-set.csv in data/raw."
        )

    frames: list[pd.DataFrame] = []
    for csv_path in selected_paths:
        frame = pd.read_csv(csv_path, low_memory=False)
        frame.columns = [str(col).strip() for col in frame.columns]

        unnamed_columns = [col for col in frame.columns if col.startswith("Unnamed:")]
        if unnamed_columns:
            frame = frame.drop(columns=unnamed_columns)

        frame = _normalize_label_column(frame)

        if list(frame.columns) != EXPECTED_UNSW_COLUMNS:
            raise DatasetLoadError(
                f"Unexpected schema in {csv_path.name}. "
                f"Expected {len(EXPECTED_UNSW_COLUMNS)} columns, found {len(frame.columns)}.\n"
                f"Columns found: {list(frame.columns)}"
            )

        frames.append(frame)

    dataset = pd.concat(frames, axis=0, ignore_index=True)
    dataset = _clean_label_values(dataset)
    return dataset


def _normalize_label_column(dataset: pd.DataFrame) -> pd.DataFrame:
    """Normalize common label column spelling differences."""
    column_map = {str(col).strip().lower(): col for col in dataset.columns}

    if "label" not in column_map:
        raise DatasetLoadError("Expected target column 'Label' or 'label' was not found.")

    original_label_col = column_map["label"]
    if original_label_col != "Label":
        dataset = dataset.rename(columns={original_label_col: "Label"})

    return dataset


def _clean_label_values(dataset: pd.DataFrame) -> pd.DataFrame:
    """Coerce Label to binary numeric form and drop invalid rows."""
    dataset["Label"] = pd.to_numeric(dataset["Label"], errors="coerce")
    dataset = dataset.dropna(subset=["Label"]).copy()
    dataset["Label"] = dataset["Label"].astype(int)
    dataset = dataset[dataset["Label"].isin([0, 1])].copy()

    if dataset.empty:
        raise DatasetLoadError("No valid rows remained after cleaning the Label column.")

    return dataset