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

## 4.5 Later fault-injection and protection evidence (status at 2026-11-23)

**Do not merge these studies with Sections 4.1–4.3.** The clean baseline above uses 5,000 COCO val2017 images. Later injected-fault studies have different development cohorts, settings, and sampling assumptions; they are not a final matched evaluation.

| Study | Recorded study scope | Supported result / limitation |
| --- | --- | --- |
| Notebook 04 INT4 | 36 initial + 96 extended INT4 scenarios; 12 additional separate TMR cases | Selected-layer, exploratory NMS-aware effects; no dataset-wide faulty-model AP claim |
| Notebook 05b logical INT8 | 63 eligible Conv2d layers, eight bits, 10 weights per layer/bit; 5,040 planned scenarios on 200 fixed development images | **2,198** official evaluations completed; **2,842** remaining per latest archived status; no complete ranking |
| Notebook 06a | Known-position masked-bit completion with clean context | Predictor feasibility only; not unknown-fault detection, repair or AP recovery |
| Notebook 06b repair | Known target weight, C/D detection-and-repair candidates | Functional recovery/false-intervention statistics under restricted assumptions; no generalizable localization |
| Notebook 06b end-to-end | 16 selected fault cases; 17 unique clean/faulty model states, with paired C/D post-NMS COCOeval | Actual faulty-model COCOeval *pilot* exists; does not prove population mitigation efficacy |
| Notebook 07a | Invariant-bit functional baseline and a defined storage-budget design comparison | Correctness and cost/coverage proxies; no independent paired AP, live RAM or deployment energy gain |
| Notebook 08 | Deterministic 500 calibration + 500 development + 4,000 nominated held-out image split | Historical use of images must be audited before independence claim |

The 16-case 06b pilot reports mean **signed AP percentage-point** changes relative to faulty inference:

| Method | Mean signed gain (AP percentage points) | Positive / unchanged / negative cases |
| --- | ---: | --- |
| C | +0.0004132768 | 1 / 12 / 3 |
| D | -0.0014996570 | 0 / 14 / 2 |

These are very small selected-case observations, **not** a scientifically validated general reliability gain. Restoring a perturbed weight can sometimes lower AP if a fault happened to increase AP on the tested cohort.

### Fault model and quantization semantics

In the main 05b/06b path, selected convolutional weights are represented as **logical signed INT8 with per-output-channel scaling**. After bit perturbation, weights are **dequantized for FP32 computation**, and activations remain FP32. This is **not native INT8 accelerator fault injection** or a measured physical failure-rate model. Persistent weight corruption, transient activation corruption, and NMS-stage corruption are different fault models; conclusions must be scoped separately.

### Supporting primary records

- [Notebook 04 closure](../docs/research_reports/Notebook_04_Closure_Report_v1.md)
- [05b campaign status](../docs/research_reports/notebook05b_cpu_replay_v1/campaign_status_02198_v1.json) and [campaign design](../docs/research_reports/05b_campaign_runner_notes_v1.md)
- [06a masked-bit study](../docs/research_reports/notebook06a_feasibility_v1/feasibility_report_v1.md)
- [06b repair study](../docs/research_reports/notebook06b_pilot_v1.md) and [16-case end-to-end summary](../docs/research_reports/notebook06b_end_to_end_v1/summary_v1.json)
- [07a functional baseline](../docs/research_reports/notebook07a_baseline_summary_v1.json) and [07a design comparison](../docs/research_reports/notebook07a_design_comparison_v1/summary_v1.json)
- [November 21 image-split manifest](../docs/research_reports/notebook08_fixed_point_v1/coco_val2017_split_v1.json)

## 4.6 Requirements before final comparative conclusions

A defensible final study requires a versioned runtime preflight; exact stored prediction and fault-record identities; a crosswalk of historical image IDs against the newer split; and a method/protection selection **frozen before** independent testing. Keep image lists, NMS, evaluation settings and injected fault scenarios paired across interventions; define and report fault denominators, invalid outputs, SDC and DUE.

Primary proposed measure: official `pycocotools` COCO bbox AP@[0.50:0.95]. Supplement with AP50, AP75, AR, missed/extra detections, raw-vs-NMS contrasts, and uncertainty. Account for **matched protection budget**, clean-model degradation, real latency and resident memory. Theoretical storage is not live RAM, and energy savings require actual instrumentation.

Activation injection, training/clamping and a zero- or near-zero-overhead hybrid are **research goals rather than demonstrated results**. This file is a thesis-writing draft; primary-paper checking, template integration and supervisor review remain open.
