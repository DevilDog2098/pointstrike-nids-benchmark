from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Persist a dictionary as formatted JSON."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
