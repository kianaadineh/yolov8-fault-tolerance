# Masked-bit contextual feasibility: implementation notes

## Scope

This module implements protocol v2 (SHA256 d851b7a91569ffa155d3e85ffd7afa1fb80e0df851857979cde7a51e392bceec)
using the frozen dataset and manifest. It requires NumPy 2.4.6 and scikit-learn
1.6.1 for the research run. It does not load YOLO, use COCO images, install
packages, request a network connection, access held-out detection images, alter
screening code, run fault correction or perform Git operations.

The task is completion of a KNOWN masked bit with CLEAN context. Its performance
is not a probability of a real hardware fault, unknown-fault localization,
repair success, or detection-AP recovery. Predictable bits may be unimportant;
important rare weights may be incorrectly corrected by a statistical rule.

## Four fixed methods

A: Laplace-smoothed training bit frequency for each layer/bit.
B: Laplace-smoothed frequency conditional on layer and the seven other bits.
   Alpha=1. An unseen conditional pattern has probability 0.5; the unseen-context
   rate is reported rather than silently changing smoothing or backoff rules.
D: HistGradientBoosting on the other seven bits and a 63-layer one-hot encoding.
C: The same learner/settings as D, plus the 13 frozen neighbor statistics.

No channel ID, weight address, flat index, or center-inclusive statistic is
allowed in predictor input. Source metadata is retained for audit/bootstrap
only. Training, calibration and test output channels are disjoint per layer.
The module validates the declared grouping and NPZ identities. It does not
independently reconstruct all dataset context values from the original model;
the preparation cell performed center-exclusion checks, and fixed hashes bind
that prepared dataset. Statistical usefulness is assessed by comparing C with
B AND the matched-learner control D.

## Implementation choices declared before research fitting

- One C and one D classifier for each of eight target bits.
- Fixed 100 boosting iterations, max_leaf_nodes=15, learning_rate=.1,
  min_samples_leaf=20 and l2_regularization=1, per the protocol.
- Disable early stopping so the sklearn default cannot introduce an implicit
  randomly split validation set or select iteration counts from test results.
- random_state=protocol seed + bit; C and D have the same seed.
- A/B/C/D fit only on the training partition. Calibration-partition predictions
  are reported separately; there is NO fitted probability calibration, threshold
  tuning, hyperparameter search or selection of a winner from these labels.
- Raw probabilities are evaluated. Clip to [1e-6,1-1e-6] for logarithmic loss
  only. Brier score uses native probabilities. Accuracy threshold is >=0.5.
- Retain constant-label per-layer/bit groups. Balanced accuracy is unavailable
  (null) there; log loss, Brier score, prevalence and accuracy remain reported.
- Global means weight sampled target rows equally; the aggregate over bits
  weights eight bits equally. Per-layer/bit metrics are also exported.
- C-minus-B and C-minus-D log-loss differences are paired on identical targets.
  Negative differences favor C. Use 2,000 exploratory percentile bootstrap draws
  (PCG64 seed 20261117), resampling output channels WITHIN each layer and keeping
  their target rows together. The eight bits use the same bootstrap draws.
- These intervals condition on one checkpoint, sampled targets and fitted
  predictors. They do not capture training/architecture uncertainty, are not
  adjusted for multiple bit comparisons and provide no universal coverage,
  detection-safety, fault-rate or repair guarantee. Per-layer metrics are
  descriptive, without claimed per-layer confidence intervals.
- Freeze these implementation details before the research run. Do not tune
  methods on the locked test set after seeing results; a change needs a new
  protocol and genuinely separate confirmation.

## Execution and artifacts

    from contextual06a_runner_v1 import run
    result = run('/content/drive/MyDrive/YOLOv8_Fault_Tolerance')

The module limits numerical thread pools to one. A Linux file lock prevents
concurrent writers. It trains/saves one target bit at a time. An existing complete
bit is verified and reused without refitting. A partial write or unfinished
multi-file bit checkpoint blocks automatic continuation for review; evidence is
not deleted. Resume requires the exact input/source/environment fingerprint.

Detailed artifacts: results/notebook06a/masked_bit_comparison_v1/.
Each bit retains probabilities, fitted A/B frequency tables and C/D predictors,
metrics and measured fit/storage information. Joblib files use executable pickle
serialization: load only trusted files produced by this module. Resume verifies
their hashes but does not deserialize them. Compressed model storage is reported;
it is not a complete deployment RAM, trusted-state or latency-overhead measure.

Publication artifacts: docs/research_reports/notebook06a_feasibility_v1/:
- comparison_summary_v1.json: global metrics, per-bit results and paired intervals.
- per_layer_bit_metrics_v1.csv: calibration/test metrics for all layer/bit/methods.
- feasibility_report_v1.md: compact table and interpretation limits.

There is no automatic positive/negative scientific verdict. Even a lower log
loss does not by itself establish useful correction of rare faults or preserve
clean-model AP. A later unknown-fault experiment must include no-fault controls,
realistic/disclosed priors, false alarms, harmful corrections, trusted metadata,
independent fault locations, repair abstention and deployment cost.

## Tests

Synthetic tests check target-bit invariance, rejection of identity leakage,
matched feature sets, smoothing, proper probability scores, constant labels,
cluster resampling, exclusive/checksummed storage and actual fitting and
serialization on synthetic data. Local tests were run with NumPy 2.3.5 and
scikit-learn 1.8.0, not the research environment. Repeat them in Colab 1.6.1
before using the frozen research dataset. Passing tests is not a research result.
