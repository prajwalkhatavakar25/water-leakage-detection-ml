# Model Card — Support Vector Machine

## Model Information

Model: Support Vector Machine (SVM)

Algorithm: Support Vector Classification

Kernel: RBF

C: 1.0

Gamma: scale

Class Weight: balanced

Random State: Not explicitly set for SVC.

## Intended Purpose

Identify potential water leakage conditions from
water tank pressure, pressure change and refill measurements.

## Input Features

- Pressure
- Pressure Change
- Refill

## Output

0 — No Leak

1 — Leak

## Final Test Performance

| Metric | Result |
|---|---:|
| Accuracy | 0.8593 |
| Precision | 0.7044 |
| Recall | 0.7643 |
| F1-Score | 0.7332 |
| ROC-AUC | 0.8873 |
| PR-AUC | 0.7565 |

## Explainability

Permutation importance was used to investigate
the contribution of each input feature.

| Feature | Importance |
| --- | --- |
| Pressure | 0.33952 |
| Pressure_Change | 0.01437 |
| Refill | 0.0988 |

## Limitations

- The model uses only three input features.
- Predictions are based on the selected dataset.
- Real-world sensor performance remains unverified.
- Prediction scores are decision-function values,
  not calibrated probabilities.
- The current implementation has not been deployed
  on physical water tank hardware.

## Deployment Status

A simulated inference implementation exists.
Production deployment has not been performed.