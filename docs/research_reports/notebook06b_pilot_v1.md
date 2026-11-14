# Notebook 06b: exploratory detection and repair pilot

## Scope
- Clean test targets: 16061.
- Single-bit fault scenarios per method: 128488.
- Fixed checkpoint, saved C/D predictors, frozen calibration thresholds.
- The target weight under inspection is given; the injected bit is hidden.
- Neighbors are clean; at most one bit is changed by the method.

## Test results

| Metric | C | D |
| --- | ---: | ---: |
| Exact weight recovery (%) | 17.424195 | 12.257954 |
| Clean false intervention (%) | 0.958844 | 0.765830 |
| Wrong-bit intervention / all fault cases (%) | 2.044549 | 1.983065 |
| Abstention / all fault cases (%) | 80.531256 | 85.758981 |

## Exact recovery by injected bit

| Bit | C (%) | D (%) |
| ---: | ---: | ---: |
| 0 | 0.000000 | 0.000000 |
| 1 | 0.000000 | 0.000000 |
| 2 | 0.000000 | 0.000000 |
| 3 | 0.211693 | 0.000000 |
| 4 | 4.495361 | 0.000000 |
| 5 | 15.490941 | 1.351099 |
| 6 | 48.602204 | 23.852811 |
| 7 | 70.593363 | 72.859722 |

## Paired exploratory uncertainty

- Exact recovery, C minus D: 5.166241 percentage points; exploratory 95% interval [4.877572, 5.453158].
- Clean false intervention, C minus D: 0.193014 percentage points; exploratory 95% interval [0.030743, 0.355289].

## Interpretation and limits

- C improves average exact recovery relative to D but increases clean false interventions.
- Both methods have zero exact recovery for injected bits 0, 1 and 2.
- D has higher exact recovery for injected bit 7.
- The 1% budget was enforced on calibration samples; it is not a test guarantee.
- Bootstrap resamples observed output channels within layers with paired methods and bits.
- Intervals condition on fixed models and thresholds; they are exploratory.
- Prior 06a test results were inspected; this is not independent confirmation.
- No network-wide localization, deployment benefit, novelty, or AP recovery is established.
- Per-layer outcome analysis and end-to-end evaluation remain pending.

## Provenance

- Protocol SHA256: c996c54c5d8c14a0c57267617f4967ce48b2343a7c43e03669aff48d52ba1361
- Calibration SHA256: 87bfe4072cc4f8b1eb90fe961d4c348552f674223d6d49a85f5e7e0c7e1931e0
- Outcome archive SHA256: d6739923adbd38e76e41183678f9780bb6b1bfba044525bf4937c155c5a2fae2
- Detailed records: results/notebook06b/evaluation_v1/
- No changes to the 05b campaign.
