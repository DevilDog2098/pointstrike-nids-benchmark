from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiment import run_experiment
from src.utils import load_yaml


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    model_config = load_yaml(project_root / "configs" / "model_config.yaml")

    run_experiment(
        experiment_id="EXP-005",
        model_key="mlp",
        candidate_configs=model_config["models"]["mlp"]["candidate_configs"],
    )


if __name__ == "__main__":
    main()