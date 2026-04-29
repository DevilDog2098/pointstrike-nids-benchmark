# PointStrike NIDS Benchmark

Reproducible benchmark project for binary network intrusion detection using tabular network-flow data, with a baseline benchmark on UNSW-NB15 and external validation on CIC-IDS2017.

## Project overview

This repository contains the code, configuration files, and experiment workflow used for a reproducible evaluation of common machine learning models for network intrusion detection. The study compares baseline models on the UNSW-NB15 dataset and then evaluates cross-dataset generalization on CIC-IDS2017 using a harmonized shared-feature design.

## Study scope

- **Primary task:** binary classification (`Label`: 0 = benign, 1 = attack)
- **Primary benchmark dataset:** UNSW-NB15
- **External validation dataset:** CIC-IDS2017
- **Model families evaluated:**
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Support Vector Machine
  - Multi-Layer Perceptron

## Repository structure

- `configs/` – YAML configuration files
- `src/` – reusable source code
- `scripts/` – experiment entry points
- `outputs/` – generated models, metrics, figures, and logs (excluded from public repo by default)
- `data/` – local dataset directory structure (excluded from public repo)

## Core baseline benchmark

The primary benchmark was conducted on the processed UNSW-NB15 training and testing partitions.

### Core excluded fields

The final baseline benchmark excludes the following fields from model input:

- `id`
- `attack_cat`
- `Label` (used only as the target)
- `stcpb`
- `dtcpb`

### Baseline experiments

- `EXP-001` – Logistic Regression
- `EXP-002` – Random Forest
- `EXP-003` – XGBoost
- `EXP-004` – Support Vector Machine
- `EXP-005` – Multi-Layer Perceptron

## Cross-dataset validation

The external-validation phase uses a harmonized shared-feature subset between UNSW-NB15 and CIC-IDS2017.

### Shared-schema UNSW retraining experiments

- `EXT-UNSW-001` – Random Forest
- `EXT-UNSW-002` – XGBoost
- `EXT-UNSW-003` – Multi-Layer Perceptron

### CIC-IDS2017 external validation

The saved shared-schema models are evaluated on CIC-IDS2017 without retraining on the external dataset.

## Data access

Datasets are **not** redistributed in this repository.

To run the project locally, obtain the original datasets separately and place them into your local `data/` directory structure as needed.

## Suggested setup

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Obtain the required datasets separately and place them into the local `data/` directory.
4. Review YAML settings in `configs/`.
5. Run the benchmark or external-validation scripts from `scripts/`.

## Reproducibility note

This repository is intended to accompany the manuscript:

**Reproducible Evaluation of Machine Learning Models for Network Intrusion Detection: False-Positive Burden and Cross-Dataset Generalization**

It provides the code and configuration workflow used for the reported experiments, while excluding raw datasets and large generated artifacts from version control.
