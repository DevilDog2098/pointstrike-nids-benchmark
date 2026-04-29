from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class SharedSchemaError(RuntimeError):
    """Raised when shared-schema datasets cannot be built."""


UNSW_TO_SHARED = {
    "dur": "dur",
    "spkts": "spkts",
    "dpkts": "dpkts",
    "sbytes": "sbytes",
    "dbytes": "dbytes",
    "smean": "smean",
    "dmean": "dmean",
    "rate": "rate",
    "Label": "Label",
}

CIC_TO_SHARED = {
    "Flow Duration": "dur",
    "Total Fwd Packets": "spkts",
    "Total Backward Packets": "dpkts",
    "Total Length of Fwd Packets": "sbytes",
    "Total Length of Bwd Packets": "dbytes",
    "Fwd Packet Length Mean": "smean",
    "Bwd Packet Length Mean": "dmean",
    "Flow Packets/s": "rate",
    "BinaryLabel": "Label",
}


def load_unsw_processed(project_root: Path) -> pd.DataFrame:
    """Load the processed UNSW benchmark partition files and combine them."""
    raw_dir = project_root / "data" / "raw"
    file_names = [
        "UNSW_NB15_training-set.csv",
        "UNSW_NB15_testing-set.csv",
    ]

    frames: list[pd.DataFrame] = []
    for file_name in file_names:
        path = raw_dir / file_name
        if not path.exists():
            raise SharedSchemaError(f"Missing UNSW file: {path}")

        frame = pd.read_csv(path, low_memory=False)
        frame.columns = [str(col).strip() for col in frame.columns]
        if "label" in frame.columns and "Label" not in frame.columns:
            frame = frame.rename(columns={"label": "Label"})
        frames.append(frame)

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined["Label"] = pd.to_numeric(combined["Label"], errors="coerce")
    combined = combined.dropna(subset=["Label"]).copy()
    combined["Label"] = combined["Label"].astype(int)
    combined = combined[combined["Label"].isin([0, 1])].copy()

    return combined


def load_cic_combined(project_root: Path) -> pd.DataFrame:
    """Load and combine all CIC-IDS2017 machine-learning CSV files."""
    raw_dir = project_root / "data" / "cic_ids2017" / "raw"
    csv_paths = sorted(raw_dir.glob("*.csv"))

    if not csv_paths:
        raise SharedSchemaError(f"No CIC CSV files found in: {raw_dir}")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, low_memory=False)
        frame.columns = [str(col).strip() for col in frame.columns]

        if "Label" not in frame.columns:
            raise SharedSchemaError(f"Missing Label column in CIC file: {csv_path.name}")

        frame["Label"] = frame["Label"].astype(str).str.strip()
        frame["BinaryLabel"] = frame["Label"].apply(
            lambda value: 0 if value.upper() == "BENIGN" else 1
        )
        frames.append(frame)

    combined = pd.concat(frames, axis=0, ignore_index=True)
    return combined


def build_shared_unsw(dataset: pd.DataFrame) -> pd.DataFrame:
    """Project UNSW data into the shared feature schema."""
    required_columns = list(UNSW_TO_SHARED.keys())
    missing = [column for column in required_columns if column not in dataset.columns]
    if missing:
        raise SharedSchemaError(f"UNSW dataset is missing required columns: {missing}")

    shared = dataset[required_columns].copy()
    shared = shared.rename(columns=UNSW_TO_SHARED)
    return shared


def build_shared_cic(dataset: pd.DataFrame) -> pd.DataFrame:
    """Project CIC data into the shared feature schema."""
    required_columns = list(CIC_TO_SHARED.keys())
    missing = [column for column in required_columns if column not in dataset.columns]
    if missing:
        raise SharedSchemaError(f"CIC dataset is missing required columns: {missing}")

    shared = dataset[required_columns].copy()
    shared = shared.rename(columns=CIC_TO_SHARED)

    for column in ["dur", "spkts", "dpkts", "sbytes", "dbytes", "smean", "dmean", "rate"]:
        shared[column] = pd.to_numeric(shared[column], errors="coerce")

    shared["Label"] = pd.to_numeric(shared["Label"], errors="coerce")
    shared = shared.dropna().copy()
    shared["Label"] = shared["Label"].astype(int)

    return shared


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "shared_schema"
    output_dir.mkdir(parents=True, exist_ok=True)

    unsw = load_unsw_processed(project_root)
    cic = load_cic_combined(project_root)

    unsw_shared = build_shared_unsw(unsw)
    cic_shared = build_shared_cic(cic)

    unsw_path = output_dir / "unsw_shared_schema.csv"
    cic_path = output_dir / "cic_ids2017_shared_schema.csv"

    unsw_shared.to_csv(unsw_path, index=False)
    cic_shared.to_csv(cic_path, index=False)

    print("=" * 80)
    print("SHARED SCHEMA DATASETS CREATED")
    print("=" * 80)
    print(f"UNSW shared dataset path: {unsw_path}")
    print(f"UNSW shared dataset shape: {unsw_shared.shape}")
    print("UNSW label distribution:")
    print(unsw_shared["Label"].value_counts(dropna=False).sort_index())
    print()

    print(f"CIC shared dataset path: {cic_path}")
    print(f"CIC shared dataset shape: {cic_shared.shape}")
    print("CIC label distribution:")
    print(cic_shared["Label"].value_counts(dropna=False).sort_index())
    print()

    print("Shared columns:")
    print(list(unsw_shared.columns))


if __name__ == "__main__":
    main()