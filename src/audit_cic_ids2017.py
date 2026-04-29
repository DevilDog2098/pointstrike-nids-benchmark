from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class CICAuditError(RuntimeError):
    """Raised when the CIC-IDS2017 audit cannot be completed."""


def load_cic_ids2017_files(data_dir: str | Path) -> list[tuple[str, pd.DataFrame]]:
    """Load all CIC-IDS2017 CSV files from a directory."""
    data_path = Path(data_dir)
    csv_paths = sorted(data_path.glob("*.csv"))

    if not csv_paths:
        raise CICAuditError(f"No CSV files found in: {data_path}")

    loaded_files: list[tuple[str, pd.DataFrame]] = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, low_memory=False)
        frame.columns = [str(col).strip() for col in frame.columns]
        loaded_files.append((csv_path.name, frame))

    return loaded_files


def normalize_binary_labels(frame: pd.DataFrame, label_column: str = "Label") -> pd.DataFrame:
    """Map CIC labels to binary labels: BENIGN -> 0, everything else -> 1."""
    if label_column not in frame.columns:
        raise CICAuditError(f"Expected label column '{label_column}' was not found.")

    normalized = frame.copy()
    normalized[label_column] = normalized[label_column].astype(str).str.strip()
    normalized["BinaryLabel"] = normalized[label_column].apply(
        lambda value: 0 if value.upper() == "BENIGN" else 1
    )
    return normalized


def summarize_file(file_name: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Create a summary dictionary for a single CIC file."""
    normalized = normalize_binary_labels(frame)

    label_counts = normalized["Label"].value_counts(dropna=False).to_dict()
    binary_counts = normalized["BinaryLabel"].value_counts(dropna=False).to_dict()

    return {
        "file_name": file_name,
        "row_count": int(len(normalized)),
        "column_count": int(len(normalized.columns)),
        "columns": list(normalized.columns),
        "label_counts": label_counts,
        "binary_label_counts": binary_counts,
    }


def build_combined_dataset(loaded_files: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Combine all CIC files into one cleaned dataframe."""
    normalized_frames: list[pd.DataFrame] = []

    for file_name, frame in loaded_files:
        normalized = normalize_binary_labels(frame)
        normalized["_source_file"] = file_name
        normalized_frames.append(normalized)

    combined = pd.concat(normalized_frames, axis=0, ignore_index=True)
    return combined


def compare_column_sets(loaded_files: list[tuple[str, pd.DataFrame]]) -> dict[str, Any]:
    """Compare schemas across all CIC files."""
    file_columns = {file_name: set(frame.columns) for file_name, frame in loaded_files}

    common_columns = set.intersection(*file_columns.values())
    all_columns = set.union(*file_columns.values())

    mismatches = {
        file_name: sorted(all_columns - columns)
        for file_name, columns in file_columns.items()
    }

    return {
        "common_columns": sorted(common_columns),
        "all_columns": sorted(all_columns),
        "mismatches": mismatches,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cic_data_dir = project_root / "data" / "cic_ids2017" / "raw"

    loaded_files = load_cic_ids2017_files(cic_data_dir)

    print("=" * 80)
    print("CIC-IDS2017 DATASET AUDIT")
    print("=" * 80)
    print(f"Source directory: {cic_data_dir}")
    print(f"CSV files found: {len(loaded_files)}")
    print()

    schema_report = compare_column_sets(loaded_files)
    print("Common column count across all files:", len(schema_report["common_columns"]))
    print("Total unique column count across all files:", len(schema_report["all_columns"]))
    print()

    for file_name, frame in loaded_files:
        summary = summarize_file(file_name, frame)
        print("-" * 80)
        print(f"File: {summary['file_name']}")
        print(f"Rows: {summary['row_count']}")
        print(f"Columns: {summary['column_count']}")
        print("Label distribution:")
        for label, count in summary["label_counts"].items():
            print(f"  {label}: {count}")
        print("Binary label distribution:")
        for label, count in summary["binary_label_counts"].items():
            print(f"  {label}: {count}")
        print()

    combined = build_combined_dataset(loaded_files)
    print("=" * 80)
    print("COMBINED DATASET SUMMARY")
    print("=" * 80)
    print(f"Combined rows: {len(combined)}")
    print(f"Combined columns: {len(combined.columns)}")
    print("Combined binary label distribution:")
    print(combined["BinaryLabel"].value_counts(dropna=False).sort_index())
    print()

    print("Common columns:")
    for column in schema_report["common_columns"]:
        print(f"  {column}")


if __name__ == "__main__":
    main()