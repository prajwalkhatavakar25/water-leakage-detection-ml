# Water Leakage Detection Using Machine Learning

## Overview

This project develops a machine learning-based water leakage
detection system using pressure and refill measurements from
a simulated network of 100 connected water tanks.

Four classification algorithms are implemented and evaluated
to identify potential leakage conditions.

## Objectives

- Analyse water tank sensor measurements.
- Perform exploratory data analysis.
- Develop meaningful sensor-based features.
- Implement four classification algorithms.
- Evaluate and compare predictive performance.
- Investigate model errors and feature importance.
- Document a proposed deployment architecture.
- Establish model monitoring procedures.

## Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Environment | Google Colab |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Storage | Joblib |
| Version Control | Git and GitHub |

## Dataset

**Source:** https://github.com/IndustrialNetwork/GraphDataset.git

**Repository commit:**

`862da8b419e65b240c107a343b72dad44915c811`

The repository contains 14 CSV files. This experiment uses
the 12 measurement CSV files.

Original dataset dimensions:

- 100,000 timestamps
- 301 columns
- 100 tank pressure measurements
- 100 refill indicators
- 100 leakage indicators

The dataset contains 10 million individual tank-time observations.

## Exploratory Data Analysis

Important findings:

- Missing values: 0
- Duplicate rows: 0
- Leakage observations: 26.80%
- Normal observations: 73.20%
- Average pressure: 54.26
- Pressure change was examined over sequential timestamps.

## Feature Engineering

Three features are used:

1. Pressure
2. Pressure Change
3. Refill

The target variable is Leak:

- 0 = No Leak
- 1 = Leak

## Data Splitting

Chronological splitting was used.

| Dataset | Time Range | Sample Size |
|---|---|---:|
| Training | 0–69,999 | 30,000 |
| Validation | 70,000–84,999 | 10,000 |
| Testing | 85,000–99,999 | 10,000 |

Stratified sampling was performed independently within each
chronological period.

## Machine Learning Algorithms

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine

## Final Test Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7666 | 0.5345 | 0.5967 | 0.5639 | 0.749 | 0.6472 |
| Decision Tree | 0.7888 | 0.5664 | 0.7034 | 0.6275 | 0.7853 | 0.5622 |
| Random Forest | 0.8655 | 0.7541 | 0.6947 | 0.7232 | 0.8798 | 0.7928 |
| SVM | 0.8593 | 0.7044 | 0.7643 | 0.7332 | 0.8873 | 0.7565 |

## Model Selection

SVM is the preferred model when recall and F1-score are
the primary priorities.

Random Forest is a strong alternative when precision,
accuracy, PR-AUC and prediction efficiency are important.

## Explainability

Permutation importance was used to investigate the
contribution of Pressure, Pressure Change and Refill.

Pressure showed the greatest importance for SVM,
Random Forest and Decision Tree.

## Deployment

A proposed architecture includes:

Sensors → Data Collection → Preprocessing →
ML Prediction → Alert Generation → Monitoring.

A simulated inference function was implemented.

**Physical sensor deployment has not been implemented.**

## Monitoring

The proposed monitoring procedures cover:

- Prediction quality
- False alarms
- Missed leakage events
- Data quality
- Data drift
- Prediction latency

## Reproducibility

Dataset version, file checksums, software versions,
model configurations and dataset splitting details
are maintained in the documentation directory.

## Limitations

- Only three input features are used.
- Training uses a sampled subset of the observations.
- Performance is evaluated on a sampled future time period.
- Physical sensor deployment is not implemented.
- Hyperparameter optimization has not been exhaustive.
- Real-world performance requires validation using actual sensor data.

## Project Status

Machine learning implementation and experimental evaluation
completed. Deployment architecture and monitoring are documented
as proposed functionality.

## License

Specify the dataset and project license before redistribution.