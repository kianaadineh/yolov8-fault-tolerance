# Contextual masked-bit feasibility

This is bit completion, not fault detection, repair or AP recovery.
All methods are fixed before test evaluation. Context is clean in this probe.
C/D use the same learner and hyperparameters; D has no neighbor features.
Calibration targets were evaluated but not used to fit or select methods.

| Bit | A test log loss | B test log loss | D test log loss | C test log loss | C minus D |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6941209 | 0.7257987 | 0.6927345 | 0.6942086 | +0.0014741 |
| 1 | 0.6943639 | 0.7251241 | 0.6903940 | 0.6907367 | +0.0003427 |
| 2 | 0.6934241 | 0.7100933 | 0.6761180 | 0.6717887 | -0.0043292 |
| 3 | 0.6932385 | 0.6725779 | 0.6413702 | 0.6310085 | -0.0103617 |
| 4 | 0.6936821 | 0.5904491 | 0.5572888 | 0.5351026 | -0.0221862 |
| 5 | 0.6940437 | 0.4349845 | 0.3923654 | 0.3650960 | -0.0272694 |
| 6 | 0.6927547 | 0.2554709 | 0.1540050 | 0.1417981 | -0.0122069 |
| 7 | 0.6932496 | 0.2261678 | 0.1153481 | 0.1102379 | -0.0051102 |

## Paired exploratory channel-bootstrap comparisons

- C_minus_B: mean log-loss difference -0.06258615; exploratory interval [-0.06555683940775589, -0.059643395368806265].
- C_minus_D: mean log-loss difference -0.00995584; exploratory interval [-0.011044272597539833, -0.008895462085323819].

No novelty, repair-success or deployment-safety claim follows from this report.

## Limitations

- Bit completion is not fault detection or successful repair.
- Known masked-bit position is not available from an unknown fault.
- Clean context does not establish robustness to corrupted neighbors.
- Predictive accuracy does not establish recovery of detection AP.
- Predictor memory, trusted state and runtime costs remain to be measured.
- No COCO images or held-out detection images are used in this probe.
- Known masked-bit position; no fault-localization or correction experiment.
- Model fitting uses training targets only; calibration labels are not used to tune methods.
- Bootstrap conditions on one checkpoint, sampled targets and fitted predictors.
- Intervals are exploratory, not simultaneous; inspect all bits and constant-label groups.
- Per-layer results are descriptive; this run does not measure detection AP recovery.
- Joblib files are executable serialization: load only trusted locally produced files.
