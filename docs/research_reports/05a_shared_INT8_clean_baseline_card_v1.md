# Shared INT8 clean baseline — development validation

Status: PASS
Images: 1000 frozen development images
Session fingerprint: 5fab9260071987f89edf01099cd63925ca4743fb47f925063e9fa17aaac59c39
Evaluation SHA256: 8bb5cca90038987cf54457b7197036439d136167b71f65c49423454f7c37290b

| Metric | FP32 | Logical INT8 per channel | INT8 minus FP32 |
| --- | ---: | ---: | ---: |
| AP50:95 | 0.461404452 | 0.460765586 | -0.000638866 |
| AP50 | 0.620235046 | 0.619671433 | -0.000563612 |
| AP75 | 0.501526305 | 0.502494446 | +0.000968141 |
| AR100 | 0.615880698 | 0.615300794 | -0.000579904 |

## Interpretation

The recorded provisional engineering gate allows at most 0.01 AP50:95
loss and 0.01 AR100 loss. Both conditions are required.
This is not a statistical equivalence test or final held-out evaluation.

Weights use logical INT8 with per-output-channel scales; computation
and activations remain FP32. No faults were injected.
No native INT8 hardware performance claim is made.

Detailed results and paired prediction chunks:
results/notebook05a/shared_clean_cpu_v1/

No commit or push is performed by this execution.
