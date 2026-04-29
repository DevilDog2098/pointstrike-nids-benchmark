from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_shared_schema_experiment import run_shared_schema_experiment
from src.utils import load_yaml


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    model_config = load_yaml(project_root / "configs" / "model_config.yaml")

    run_shared_schema_experiment(
        experiment_id="EXT-UNSW-001",
        model_key="random_forest",
        candidate_configs=model_config["models"]["random_forest"]["candidate_configs"],
    )


if __name__ == "__main__":
    main()