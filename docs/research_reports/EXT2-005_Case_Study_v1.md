# EXT2-005: NMS-Aware INT4 Fault Propagation Case Study

## Status and scope

This is a retrospective, single-image case study based exclusively on
an existing Notebook 04 experimental artifact. No inference or new
fault injection was performed to produce this report.

## Source provenance

- Source: `results/fault_injection/int4_extended_EXT2-005_real_image_v1.json`
- Source SHA256: `18209e125ad5d8ad51081c243c76c78783c214a64c72fec38560124f0ffd6289`
- Scenario: `EXT2-005`
- Experiment: `extended_balanced_INT4_sensitivity`
- Input: `ultralytics_bus.jpg`
- Baseline: Only target Conv2d quantized to logical INT4

## Fault specification

- Target layer: `1.conv`
- Weight index: `[61, 28, 1, 2]`
- Logical INT4 bit: `2`
- Weight category: `positive`
- Clean quantized value: `2`
- Faulty quantized value: `6`
- Quantization scale: `0.06072126116071429`
- Dequantized weight change: `0.24288504464285715`

## NMS configuration

- Confidence threshold: `0.25`
- IoU threshold: `0.45`
- Number of classes: `80`
- Maximum detections: `300`

## Recorded observations

- Raw prediction MAE: `0.03917546197772026`
- Maximum raw absolute difference: `78.53591918945312`
- Clean detections: `5`
- Faulty detections: `6`
- Matched detections: `5`
- Unmatched clean detections: `0`
- Unmatched faulty detections: `1`
- Mean matched IoU: `0.997593653`
- Maximum absolute matched confidence change: `0.015235573`
- Maximum recorded matched coordinate difference: `0.898315` pixels

## Scientific interpretation

The recorded bit-flip fault was associated with an increase in
post-NMS detections from five to six. The five matched detections
remained geometrically similar, with a high mean matched IoU.

This observation supports evaluating fault effects at the detection
and NMS-output level rather than relying exclusively on raw-output
numerical differences.

The mechanism producing the additional detection cannot be
determined from the saved matched-detection records alone.

## Limitations

- Only one real image was evaluated in this case study.
- The extra detection's confidence, class and box were not saved
  as a separate unmatched-detection record.
- The extra detection cannot be classified as a false positive
  without comparison against ground truth.
- The precise pre-NMS mechanism is not established.
- The result is not a COCO AP or mAP measurement.
- No general fault rate or hardware reliability estimate follows
  from this single selected scenario.

## Data required in future experiments

1. Full clean and faulty detection lists, including unmatched boxes.
2. Relevant pre-NMS predictions and suppression diagnostics.
3. Explicit matching rules and all NMS settings.
4. Ground-truth annotations for detection-level correctness.
5. Repeated scenarios across images, layers, bits and weights.
