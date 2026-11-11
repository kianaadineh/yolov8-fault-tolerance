# Notebook 05 interim research checkpoint

## Status

The initial 12-layer, 240-scenario campaign is paused after 60 completed
scenarios on three layers, each evaluated on 1000 development images.
This is an exploratory checkpoint, not completed thesis validation.

The research notebook snapshot contains 99 reviewed code cells.
Publication-management cells appended afterward are excluded.
The full Colab working notebook is preserved in Drive.
Snapshot SHA256: c739ab5c7a631e2b20ba3bb2233f39d2d3f7270d5ec36ce2d5aca5af67a78db9

Do not Run all. The notebook spans multiple runtimes and includes
installations, GPU operations and exclusive file writes.
It is a session record, not a validated end-to-end reproduction pipeline.

## Evidence

This directory includes 64 main-study evaluations, three layer summaries
and an artifact manifest. Large prediction archives remain in Drive.
Hashes identify those artifacts but do not provide access to them.
Original results and experiment plans remain unchanged.

## Clean candidate pilot

All values below concern the same 50 development images.

| Model | AP50:95 |
| --- | ---: |
| FP32 | 0.6069269272910565 |
| Logical INT4 per tensor | 0.005479964579888446 |
| Logical INT4 per output channel | 0.2453185782831067 |
| Logical INT8 per output channel | 0.609048366736769 |

Candidate exploration was sequential. The provisional quality gate was
defined after observing the pilots. These are not held-out test results.
Weights were dequantized for FP32 computation; no native integer
hardware performance was measured.

## Resume point

1. Evaluate shared clean INT8 on 1000 development images.
2. Apply the recorded provisional AP and AR loss budgets.
3. Define broader layer screening and follow-up fault sampling.
4. Record the revised protocol before new fault inference.

The INT8 1000-image evaluation is pending.
The remaining 180 initial fault scenarios have not been executed.
The 4000 held-out images were not evaluated in the reported runs.

## Known limitations

One historical cell contains an insufficient chained-inequality
uniqueness guard. Other recorded split checks independently tested
uniqueness. Correct the guard in a versioned reusable runner before
future execution; preserve the historical record.

Five sampled weights per bit do not establish precision guarantees.
Bootstrap intervals condition on fixed development images.
The reproduced 12-layer selection does not rank all 63 eligible layers.
Target-layer-only INT4 baselines differ between layers.
