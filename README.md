# PointStrike NIDS Benchmark

Reproducible benchmark project for binary network intrusion detection on tabular network-flow data using UNSW-NB15.

## Initial scope

* Primary task: binary classification (`Label`: 0 = benign, 1 = attack)
* Primary dataset: UNSW-NB15
* Planned external validation dataset: CIC-IDS2017
* Model families:

  * Logistic Regression
  * Random Forest
  * XGBoost or Gradient Boosting
  * Support Vector Machine
  * Multi-Layer Perceptron

## Project structure

* `data/raw/` for original dataset files
* `data/interim/` for split files and cached intermediate artifacts
* `data/processed/` for model-ready datasets if needed
* `configs/` for YAML configuration files
* `src/` for reusable source code
* `scripts/` for experiment entry points
* `outputs/` for models, metrics, figures, and logs

## Core benchmark exclusions

The core benchmark excludes the following input fields from model training:

* `srcip`
* `dstip`
* `attack_cat`
* `Label` (used as target only)
* `Stime`
* `Ltime`
* `stcpb`
* `dtcpb`

## First experiment

`EXP-001` runs Logistic Regression on the documented core preprocessing pipeline.

## Suggested setup

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Place UNSW-NB15 CSV files in `data/raw/`.
4. Review YAML settings in `configs/`.
5. Run `scripts/run_exp_001.py`.
