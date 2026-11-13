# Automated logical INT8 weight-fault screening v1

## Scope and scientific status

This is a fixed-budget EXPLORATORY campaign, not a power-validated final study.
It covers all 63 eligible Conv2d weight tensors, all 8 logical INT8 bits and ten
uniformly sampled distinct weights per layer/bit: 5,040 persistent single-weight
faults, each evaluated on the same frozen 200 development images. Weight reuse
across different bit strata is allowed. No sign/zero balancing is applied.
It does not cover multiple simultaneous faults, activation faults, bias faults,
scale corruption, native INT8 kernels or physical hardware fault probabilities.
The fixed DFL projection is excluded. Computation and activations are FP32.

The 200 images cover 76 classes; four classes have no valid non-crowd boxes.
All images, including the four without valid non-crowd boxes, are retained.
Evaluation uses original COCO annotations, CPU NMS and COCOeval bbox. The clean
baseline is reconstructed from the checkpoint and must exactly match 05a's
full state hash. Clean and faulty outputs use identical images and settings.
The previously observed engineering sign-bit scenario is not added to screening.
Its location remains in the sampling population and may be sampled by chance.

## Automatic behavior

- Separate `plan` operation freezes source identities, locations, seeds and order
  before screening inference. It performs synthetic CPU initialization only.
- No downloads, network requests, package installations, push or Git mutations.
- Build one common clean model, cache JPEG bytes once and evaluate the clean
  model once for this campaign. No repetition of a clean baseline per layer.
- Inject exactly one logical bit; verify the full other state by temporarily
  masking the one scalar. The faulty model must remain unchanged during inference.
- Restore and verify the exact shared clean state in `finally`.
- Save compressed predictions and an evaluation record for every completed
  scenario. Original artifacts are never overwritten. A prediction saved before
  interruption can be reused after identity validation.
- Exclusive Linux file lock prevents concurrent writers. Existing `.part` or
  failure records block execution for review; they are never deleted automatically.
- Resume checks exact source, sampling, images, package versions, GPU name,
  torch thread count and environment identity. Changes fail closed.
- Unexpected exceptions and invalid outputs stop the campaign and record failure;
  they are not silently skipped, assigned zero harm or turned into a full result.
  A user interrupt caught by Python is recorded as a failure for review. A hard
  runtime loss can resume when no partial write or failure record exists.
- Default wall budget is four hours per invocation, checked between scenarios.
  Results completed before this boundary are retained. There is no automatic
  background service or promise that a Colab runtime survives disconnection.
- Progress snapshots every 50 completed scenarios and at a clean invocation end.
  Partial results do not produce a layer ranking.
- On all 5,040 completions, emit JSON, Markdown and CSV with the exploratory
  63-layer ranking, per-bit means, observed maximum, signed changes and descriptive
  loss counts at .001 and .01 absolute AP (0.1 and 1 AP percentage points).

## Uncertainty and interpretation

Primary: equally weighted mean across bits of nonnegative AP50:95 loss,
max(0, clean AP - faulty AP). Signed changes are retained. Thresholds are
provisional engineering flags, not safety standards or acceptance tests.
An exploratory stratified percentile bootstrap resamples faults WITHIN each bit
stratum (2,000 draws). Its intervals are conditional on the fixed images, use a
small sample, are not simultaneous across layers, and do not establish rank
stability, image uncertainty or rare-tail discovery. Zero observed loss does not
establish immunity. The maximum observed fault is not the population worst case.
Independent confirmation with fresh faults, expanded development images and
apparently low-risk controls requires a separate versioned protocol. Screening
alone must not authorize final protection choices or generalization claims.
No held-out image is used. The full campaign is not extended or stopped for a
preferred ranking. Any resource-driven truncation must be explicitly reported.

## Run

The project root must contain the existing `code/`, `models/`, and `datasets/`
folders. After installation of the exact research dependencies and this package:

    from screen05b_campaign_v1 import Campaign
    campaign = Campaign('/content/drive/MyDrive/YOLOv8_Fault_Tolerance')
    plan = campaign.create_plan()  # once; review output before execution
    result = campaign.run(max_new_scenarios=2, max_wall_seconds=900)
    # After checking the two-scenario gate, run all remaining scenarios:
    result = campaign.run(max_new_scenarios=None, max_wall_seconds=14400)

The two-scenario gate is an implementation check, not outcome-based stopping.
The final execution uses the same frozen plan. It can be resumed by the same call.
The helper intentionally does not bypass a failed gate.

CLI on a Linux computer (or compatible WSL setup), after all assets are local:

    python src/screen05b_campaign_v1.py plan --project-root /path/to/project
    python src/screen05b_campaign_v1.py run --project-root /path/to/project --max-new 2 --max-seconds 900
    python src/screen05b_campaign_v1.py run --project-root /path/to/project --max-seconds 14400

A CUDA GPU is required by this version; no silent CPU fallback. Internet is not
required by this module once dependencies, checkpoint and COCO files are local.
The computer must remain on. Colab execution is on a remote runtime, not on the
user's laptop. Four-hour invocation bounds are not runtime-availability guarantees.

Predictions and detailed results remain under ignored
`results/notebook05b/int8_screening_v1/`. Small progress and final reports are
written to `docs/research_reports/notebook05b_screening_v1/`; no files are staged.
Thousands of prediction files can require several GiB. Inspect actual free
storage and the first two scenarios before a long run. Timing from one clean
model or one convenience fault is not a guaranteed campaign budget.

## Verification limits

Local tests cover bit arithmetic, balanced sampling, checksum/provenance guards,
exclusive writes, retained empty images, prediction validation and partial-report
behavior. Local runtime has no PyTorch/CUDA; these tests do not establish CUDA,
model restoration, quantization reconstruction or actual COCOeval correctness.
The two-scenario Colab gate checks those paths before long execution.
