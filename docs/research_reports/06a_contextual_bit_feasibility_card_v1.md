# Contextual masked-bit feasibility: result card

Status: initial feasibility completed; not fault detection or repair.
Source report SHA256: 95a748398cda073cb13353760647dd82a9e4de71c2fea02daacbbe07c5c851ba

C and D use the same learner and settings. C adds local context.
Output channels are disjoint between training, calibration and test.

| Bit | D test log loss | C test log loss | C minus D |
| ---: | ---: | ---: | ---: |
| 0 | 0.692734452 | 0.694208602 | +0.001474149 |
| 1 | 0.690394019 | 0.690736742 | +0.000342723 |
| 2 | 0.676117955 | 0.671788717 | -0.004329238 |
| 3 | 0.641370223 | 0.631008481 | -0.010361741 |
| 4 | 0.557288819 | 0.535102638 | -0.022186181 |
| 5 | 0.392365425 | 0.365096018 | -0.027269406 |
| 6 | 0.154004962 | 0.141798099 | -0.012206863 |
| 7 | 0.115348118 | 0.110237945 | -0.005110173 |

Mean C-minus-D log-loss difference: -0.0099558413454231
Exploratory interval: [-0.011044272597539833, -0.008895462085323819]

## Interpretation

- Context reduced observed test log loss for bits 2 through 7.
- Bit 0 worsened; bit 1's exploratory interval includes zero.
- Retain all bit results; do not select successful bits and report them as a preregistered subset.
- Intervals are conditional, exploratory and unadjusted for multiple comparisons.
- No unknown-fault detection, correction, false-alarm or AP-recovery result exists yet.
- Clean context and a known masked-bit position were assumed.
- Compressed predictor storage is not total deployment overhead.
- The test set must not be reused for tuning this comparison.

Next: checkpoint this evidence, then specify a separate fault-detection
and repair feasibility protocol with independent faults and no-fault controls.
