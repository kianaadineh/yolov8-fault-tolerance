# Notebook 05a CPU clean-validation runner

Purpose: paired FP32 and logical INT8 per-output-channel weight validation
on the 1000 frozen development images. No fault injection, held-out
evaluation, native integer execution or GPU benchmark is implemented.

## Files

- src/baseline05a_runner_v1.py
- tests/test_baseline05a_runner_v1.py
- this note

These are new files; do not overwrite historical notebooks or runners.
Importing the runner performs no model loading or filesystem writes.

## API

1. prepare(project_root): validates input hashes and builds two CPU models.
   Preparation includes two synthetic zero-input warm-up forwards.
   Records package versions, model state hashes and quantization scales.
2. session.smoke(): paired inference on two development images.
3. session.run(max_new_chunks=1): one paired chunk of 25 development images.
4. session.run(max_new_chunks=None): explicitly completes remaining chunks.
5. session.evaluate(): joint COCOeval over all 1000 development IDs and the
   recorded provisional AP/AR engineering quality gate.

Output: code/results/notebook05a/shared_clean_cpu_v1/ in the project Drive.
Outputs remain ignored by the existing repository rules. A small reviewable
report should be copied to the research report area for publication later.

## Recovery behavior

- Completed compressed chunks contain internal payload hashes and provenance.
- Resume rechecks source image hashes and completion records.
- A changed runner, environment, thread count or model state rejects resume.
- Existing artifacts are never intentionally overwritten.
- A .part file requires inspection; it is not automatically deleted or reused.
- Use only one writer for this output directory.
- A runtime disconnect can lose the current unsaved chunk, not previously
  published verified chunks. Stored chunks are still checked before reuse.

## Numerical and evaluation definition

- Fixed checkpoint and data/gate SHA256 values match the published checkpoint.
- All 63 eligible Conv2d weight tensors use symmetric per-output-channel
  quantization with FP32 max-absolute/127 scales and torch.round.
- Clean integer range [-127,127]; this module injects no bits.
- Bias, DFL projection and all non-target state are preserved.
- Both candidates execute with FP32 computation on CPU, identical RGB
  preprocessing and square 640x640 LetterBox, conf=0.001 and NMS IoU=0.7.
- COCOeval uses original annotations, all development IDs and maxDets 1/10/100.
- Empty detections retain their images in evaluation.
- No mean of single-image AP is used.
- Quality gate is a development point-estimate engineering budget, not
  statistical equivalence or an independent final-test result.

## Verification completed before delivery

Twelve standard-library tests passed in the assistant workspace. They cover
exclusive writes, partial-file preservation, compressed round trips,
payload tampering, archive truncation, NaN rejection, split duplicates and
overlap, missing model/image execution records, per-image count mismatch
and retention of an image with zero predictions.

PyTorch and Ultralytics inference were not available in that workspace.
Actual model loading, quantization and inference must be validated by the
Colab smoke test before the 1000-image execution. These limitations are
intentional and must not be replaced with a claim of full reproduction.
