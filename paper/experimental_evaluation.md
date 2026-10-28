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

**Limitation:** These values were obtained using Ultralytics
validation with YOLO-format labels, not official COCOeval.
Official COCO evaluation remains necessary.
