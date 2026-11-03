# Experimental Evaluation

## 4.1 Experimental Setup

The experimental evaluation was initiated using the pretrained
YOLOv8s model as the fault-free reference architecture.

The baseline was evaluated on 5,000 COCO 2017 validation images,
using 36,335 non-crowd annotations converted to YOLO format.

The input resolution was 640 x 640, batch size was 16,
confidence threshold was 0.001, NMS IoU threshold was 0.7,
and maximum detections per image was 300.

The environment included an NVIDIA Tesla T4 GPU,
PyTorch 2.11.0+cu130, and Ultralytics 8.4.189.

## 4.2 Fault-Free Baseline Results

| Metric | Value (%) |
|---|---:|
| Precision | 68.43 |
| Recall | 56.22 |
| mAP@0.50 | 61.12 |
| mAP@0.50:0.95 | 44.43 |

These measurements establish an initial fault-free reference
for subsequent fault-injection experiments.

**Evaluation distinction:** The table above reports Ultralytics
validation results using YOLO-format labels. A separate official
COCOeval baseline has also been completed, as reported below.

## 4.3 Official COCOeval Baseline

The verified YOLOv8s weights were evaluated using pycocotools
COCOeval on all 5,000 COCO 2017 validation images.

| Metric | Value (%) |
|---|---:|
| AP@0.50:0.95 | 44.2048 |
| AP@0.50 | 60.4912 |
| AP@0.75 | 47.8362 |
| AP small | 25.3802 |
| AP medium | 49.0224 |
| AP large | 60.2695 |

The official evaluation used an input size of 640, batch size 1,
confidence threshold 0.001, and NMS IoU threshold 0.7.
COCOeval used maximum detections [1, 10, 100].

The Ultralytics validation used batch size 16.
These are separate evaluation pipelines and their results
must not be treated as identical measurements.

## 4.4 Evidence and Limitations

- Ultralytics metrics: `results/baseline_validation/baseline_coco2017_metrics.json`
- Official metrics: `results/official_cocoeval/yolov8s_official_cocoeval_metrics.json`
- Both evaluations are fault-free baselines.
- Neither evaluation measures accuracy under injected faults.
- Dataset-wide fault-tolerance claims require separate experiments.
