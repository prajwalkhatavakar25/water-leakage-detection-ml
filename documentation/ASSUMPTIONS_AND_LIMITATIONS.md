# Assumptions and Limitations

## Assumptions

1. Dataset labels are treated as the ground truth.
2. Pressure and refill measurements contain useful
   information for identifying water leakage.
3. Historical pressure observations are available
   for calculating pressure change.
4. The dataset is suitable for an initial experimental
   machine learning investigation.

## Limitations

1. Only three model input features are used.
2. Models are trained using a sampled subset.
3. No actual physical sensor network is connected.
4. The deployment workflow is proposed.
5. The inference demonstration uses simulated measurements.
6. Hyperparameter tuning has not been exhaustive.
7. Inter-tank relationships are not explicitly included
   in the current feature set.
8. Performance on real-world water infrastructure
   has not been established.
9. Permutation importance does not establish causality.
10. The prediction threshold has not been optimized
    for operational costs.

## Future Improvements

- Include tank connectivity information.
- Investigate additional time-series features.
- Perform systematic hyperparameter optimization.
- Evaluate alternative classification algorithms.
- Test on actual water distribution sensor data.
- Implement a real-time sensor integration system.
- Introduce automated drift detection.