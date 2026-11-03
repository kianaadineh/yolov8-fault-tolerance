# Notebook 04 — Research Closure Report (v1)

## 1. Research identity

- Project: Fault-Tolerant YOLO Architecture in Resource-Constrained Computing Systems
- Model: YOLOv8s
- Model parameters: 11,166,560
- Model weights SHA256: `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`
- Research repository: https://github.com/kianaadineh/yolov8-fault-tolerance
- Published commit: `f5b989fa1bd600d369ac5c5e7ff59c3254990037`
- Commit timestamp: 2026-11-03T12:21:29+00:00
- Source of truth: committed research artifacts, not notebook display output

## 2. Experiment scope

- Eligible Conv2d layers for weight fault injection: 63
- Eligible Conv2d weight parameters: 11,146,080
- Excluded fixed DFL Conv2d: `model.22.dfl.conv` (16 weights)
- Initial targeted INT4 scenarios: 36
- Extended balanced INT4 scenarios: 96
- Selected layers in extended study: 12
- Additional random TMR scenarios: 12
- Primary image: Ultralytics bus image
- Detection metric: class-aware one-to-one NMS matching, IoU threshold 0.5
- Fault model: one logical signed-INT4 weight bit flip per unprotected scenario

The 36 targeted and 96 extended scenarios are distinct experimental
collections. The 12 random TMR scenarios form an additional collection
and must not be conflated with the 132 scenarios above.

## 3. Core results

### 3.1 Layer sensitivity

The extended study measured fault-induced raw-output MAE relative to
each layer's own target-layer-only INT4 clean baseline.

The provisional top three layers by mean fault-induced raw MAE were:

1. `15.cv2.conv` — 0.01691779
2. `1.conv` — 0.01486982
3. `4.cv1.conv` — 0.01194779

These rankings are descriptive for the selected scenarios and one
image. They are not population-wide layer criticality estimates.

### 3.2 Quantization baseline control

The FP32-to-clean-INT4 audit revealed a major baseline degradation
for `9.cv2.conv`: 5 FP32 detections versus 1 clean INT4 detection.
This layer was flagged for separate review and excluded from the
provisional protection priority ranking.

Preserved NMS detection counts for other layers do not imply
identical boxes, confidences, classes, or accuracy.

### 3.3 NMS-aware fault effects

The extended 96-scenario study recorded one scenario with an
additional unmatched detection (`EXT2-005`). An unmatched detection
is not automatically a ground-truth false positive.

### 3.4 Preliminary TMR results

The logical signed-INT4 majority voter was validated under the
assumption of one corrupted bit in one of three independent replicas.

In 12 independent random weight-bit scenarios, logical voting
corrected all 12 injected replica faults. A single combined protected
model forward reproduced the clean raw output and NMS results.
This does not constitute 12 independent protected inference runs.

### 3.5 Theoretical protection overhead

Full-layer TMR requires two extra logical INT4 copies. The three
provisional priority layers have the following theoretical
additional packed-INT4 storage costs:

- `15.cv2.conv`: 24 KiB
- `1.conv`: 18 KiB
- `4.cv1.conv`: 16 KiB
- Combined: 58 KiB

These values exclude real runtime, voting logic, buffers, latency,
energy, and physical implementation overhead.

## 4. Reproducibility and evidence

- Published research files: 167
- Initial scenario JSON files: 36
- Extended scenario JSON files: 96
- Commit: `f5b989fa1bd600d369ac5c5e7ff59c3254990037`
- File-level SHA256 index: `Notebook_04_Closure_File_Index_v1.csv`
- Main experiment manifest: `int4_extended_combined_layer_manifest_v1.json`
- Final artifact inventory: `notebook04_final_artifact_inventory_v1.json`

Every indexed result file was compared byte-for-byte against its
corresponding Git blob at the published commit.

## 5. Scientific limitations

1. Current sensitivity findings are based mainly on a single real
   image and deliberately selected scenarios.
2. Target-layer-only logical INT4 quantization is not a complete
   deployed INT4 inference implementation.
3. Detection matching is not ground-truth COCO mAP.
4. Weight faults have been emphasized; activation faults require
   further study.
5. TMR results are preliminary and do not establish the proposed
   near-zero-overhead hybrid fault-tolerance framework.
6. INT8/INT4 comparison, NMS-focused mitigation, latency/energy,
   COCO faulty-model evaluation and multi-objective trade-off
   analysis remain future research tasks.

## 6. Next research phase — Notebook 05

Notebook 05 should reuse the existing published artifacts and focus
on deeper NMS-aware and bit-level sensitivity analysis, baseline
quality, and decisions needed for selective protection. It must not
repeat the completed 96 scenarios without a documented reason.

## 7. Terminology

- Fault = اشکال
- Error = خطا
- Fault injection = تزریق اشکال
- Fault tolerance = تحمل‌پذیری در برابر اشکال
- Error propagation = انتشار خطا

## 8. Closure decision

Notebook 04 is closed as a reproducible preliminary weight-fault
injection research phase. The full thesis research, hybrid mitigation
method, and final evaluation are not yet complete.

This report is generated from published artifacts and must be reviewed
before being committed as a separate documentation update.
