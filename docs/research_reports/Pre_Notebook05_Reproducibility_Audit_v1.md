# Pre-Notebook 05 Reproducibility Audit — v1

Recorded at: 2026-11-07T16:42:19.418714+00:00
Repository HEAD at report creation: `113d42103bcaa360ee8fe34bb98e01cde48d2119`

## Evidence scope

This report summarizes checks and code review performed in this session.
Earlier check results below were observed in Colab and supplied in the
session; this reporting cell does not independently rerun those checks.
No historical inference or COCO AP evaluation was reproduced.

## Dataset checks observed in this session

Persistent dataset root:
`/content/drive/MyDrive/YOLOv8_Fault_Tolerance/datasets/coco2017`

- Validation ZIP size: 815585330 bytes; archive CRC passed.
- Validation ZIP: 5000 JPG files and no duplicate file names.
- Extracted images: 5000.
- Annotation ZIP size: 252907541 bytes; archive CRC passed.
- Extracted instances_val2017.json size: 19987840 bytes.
- JSON: 5000 images, 36781 annotations, 80 categories.
- Image, category and annotation IDs passed uniqueness checks.
- Annotation image and category references passed consistency checks.
- No missing or extra JPG filenames relative to the JSON.
- All 5000 images decoded; dimensions matched the JSON.
- Original ZIP archives were preserved.
- CRC and consistency checks do not establish an independently
  authenticated dataset checksum.
- Dataset YAML and YOLO-format labels have not been prepared or verified
  at this new persistent location.

## Model identity

Checkpoint: `/content/drive/MyDrive/YOLOv8_Fault_Tolerance/models/yolov8s.pt`
Size: 22588772 bytes
SHA256: `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`

Size and hash matched the recorded research checkpoint.
This reporting cell rechecked checkpoint identity.

## Repository and environment

The earlier repository check reported branch main, HEAD
113d42103bcaa360ee8fe34bb98e01cde48d2119, and a clean working tree.

Initial audit runtime reported:
- Python: 3.13.16
- PyTorch: 2.14.1+cu130
- torchvision: 0.29.1+cu130
- NumPy: 2.4.6
- pycocotools: 2.0.11
- Ultralytics: NOT INSTALLED

The historical environment records PyTorch 2.11.0+cu130 and
Ultralytics 8.4.189. requirements.txt pins Ultralytics but is not
a validated environment lockfile. Exact environment reproduction
is not established.

Package versions at report creation:
- torch: 2.14.1+cpu
- torchvision: 0.29.1+cpu
- ultralytics: NOT INSTALLED
- numpy: 2.4.6
- pycocotools: 2.0.11

## Static code review

- INT4 and INT8 injection modules use XOR and signed two's-complement
  conversion for valid inputs. Tests were not independently rerun.
- Logical INT4 uses torch.int8 storage and dequantized FP32 inference.
  Native packed INT4 execution and hardware overhead are not established.
- The research notebook compares complete network pre-NMS outputs on
  a seeded synthetic input.
- The original notebook includes full DetectionModel forwards and
  post-NMS comparisons for single-image experiments.
- Extended scenarios use a target-layer-only quantized clean baseline.
- Reviewed NMS calls use confidence 0.25, NMS IoU 0.45, nc=80,
  and max_det=300.
- Detection comparison uses class-aware greedy one-to-one matching
  with IoU >= 0.5.
- Greedy matching does not guarantee maximum matching cardinality.
- Unmatched baseline/faulty detections are not ground-truth FP/FN.
- Aggregate raw MAE mixes coordinate and class-score quantities;
  it is insufficient as a detection-quality metric.
- The restoration arithmetic comparison covered 96 planned scenarios
  and reported zero mismatches in the audit runtime.
  Historical inference and accumulated model state were not rerun.

## Remaining limitations

- Stored results and report claims were not fully recomputed.
- Fault-injected COCO AP, multi-image sensitivity and generalization
  remain unverified.
- A complete reproducibility configuration covering environment,
  preprocessing, seeds, sampling, NMS and evaluation is still needed.
- Historical layer rankings are exploratory single-image findings.
- Preliminary logical TMR results do not establish measured runtime,
  energy, hardware overhead or a completed mitigation framework.

## Entry criteria for Notebook 05

1. Record and validate the chosen runtime and dependency versions.
2. Record checkpoint, dataset identity, image IDs, seeds, preprocessing,
   quantization, fault selection, NMS and evaluation settings.
3. Run a small controlled pilot with FP32, clean quantized and faulty
   models evaluated against the same ground truth.
4. Separate quantization degradation from fault-induced degradation.
5. Use ground-truth detection metrics as primary quality outcomes;
   retain raw-output and baseline-matching metrics as diagnostics.
6. Preserve historical artifacts and version any new results.

## Notebook and publication status

The user reports saving a Colab copy named
04b_reproducibility_audit.ipynb. Its path, contents and Git inclusion
have not yet been verified.

This report records an initial audit, not completed end-to-end
reproduction or final thesis validation. No commit or push is performed
by the reporting cell.

## Data separation and final evaluation policy

This is an agreed research requirement, not an implemented dataset split.

- Persian terminology: Fault = اشکال; Error = خطا.
- Separate training, development/validation and held-out final evaluation.
- Development includes selecting protected layers, thresholds, mitigation
  settings and any other method or hyperparameter choices.
- Do not use held-out final evaluation results to tune the method.
  Any subsequent tuning requires transparent disclosure and a new
  independent evaluation strategy.
- Audit prior image use and checkpoint pretraining provenance before
  describing images as unseen.
- Record image IDs, split-generation procedure, seeds and split hashes
  before Notebook 05 experiments.
- Do not use all 5000 COCO validation images for method development
  before the split and prior-use audit.
- Compare baseline and protected models on identical evaluation images
  and paired fault scenarios where applicable.
- Measure clean detection quality, fault-affected detection quality and
  mitigation overhead separately.
- Plan an independent second-dataset evaluation with compatible labels.
  Dataset selection, provenance and evaluation protocol remain pending.
- Report limitations and uncertainty; do not claim universal robustness.

## Audit notebook snapshot

- Destination: notebooks/04b_reproducibility_audit.ipynb
- Cell count reported after copying: 303
- SHA256 reported after copying:
  5d4ccc815c7fe1cf7d40239416b25b7b9df535b9c4fcc8169eaffde39867f57b
- Source and destination byte hashes matched.
- This is a full historical notebook copy with appended audit work,
  not a compact standalone audit notebook.
- Notebook content review before publication remains pending.

## Compact audit notebook update

This section supersedes the earlier current-notebook description;
the 303-cell snapshot above remains a historical provenance record.

- Current Git candidate: notebooks/04b_reproducibility_audit.ipynb
- Current cell count: 26 (one introduction and 25 audit cells).
- Current SHA256: ca54f956e5e7f09d6974788854610aaed2f1754126158bfb476fde12ee1fdfdc
- The first 278 cells of the full snapshot matched the cell types
  and source text of the previously reviewed working notebook.
- The compact version retains the 25 appended audit cells and their
  stored outputs.
- Full snapshot backup outside Git:
  notebook_backups/04b_full_history_before_extraction.ipynb
- The original tracked Notebook 04 was not modified.
- The compact notebook is a session record, not a Run-all pipeline.
- Notebook content review is complete for the 25 appended cell sources;
  this does not establish independent reproduction of their outputs.
- Git commit and remote publication remain pending.
