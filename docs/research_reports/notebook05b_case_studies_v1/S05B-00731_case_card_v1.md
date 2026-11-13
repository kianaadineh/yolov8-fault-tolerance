# Exploratory fault case: S05B-00731

Status: verified case study, not a layer-sensitivity ranking.
Source report SHA256: b7647e686a7aa6aaf82cc48eb65ca64c1ca72458b88441bafece5ba2b52ac35b

One persistent logical INT8 sign-bit fault in model.0.conv:
q = -69 -> 59. Evaluated on the same 200 development images.

| Metric | Clean | Faulty | Faulty minus clean |
| --- | ---: | ---: | ---: |
| AP50:95 | 0.484276959 | 0.398353270 | -0.085923689 |
| AP50 | 0.623520069 | 0.533275504 | -0.090244564 |
| AR100 | 0.625552656 | 0.536576405 | -0.088976251 |

## Object-level diagnostic

IoU=0.5; confidence=0.001; COCOeval maxDets=100;
original annotations with official crowd/ignore handling.

| Count | Clean | Faulty | Difference |
| --- | ---: | ---: | ---: |
| TP | 1107 | 1016 | -91 |
| FP | 16972 | 19345 | +2373 |
| FN | 204 | 295 | +91 |

Previously matched ground-truth objects lost: 118.
Previously unmatched ground-truth objects gained: 27.

## Evidence and limits

- Clean and faulty prediction archive/payload checksums were verified.
- All twelve COCOeval metric types were reproduced for both conditions.
- This case was selected for its largest observed AP drop in an incomplete campaign.
- It demonstrates vulnerability of this sampled location, not a definitive layer ranking.
- Counts at confidence 0.001 are not deployment-threshold performance estimates.
- Observed output changes do not establish that NMS caused the degradation.
- No new model inference was performed.
