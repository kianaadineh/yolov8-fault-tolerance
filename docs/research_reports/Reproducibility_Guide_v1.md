# YOLOv8s Fault Tolerance — Reproducibility Guide v1

## 1. Purpose and scope

This document describes the recorded experimental environment, repository
structure, research artifacts, and known reproducibility limitations.

It distinguishes historical results, recovered outputs, validated stored
artifacts, and experiments that have not been independently rerun.

This guide is not a claim that the complete research pipeline has been
reproduced from scratch.

## 2. Repository and project layout

- Google Drive project root:
  `/content/drive/MyDrive/YOLOv8_Fault_Tolerance`
- Git repository:
  `/content/drive/MyDrive/YOLOv8_Fault_Tolerance/code`
- Model checkpoint:
  `/content/drive/MyDrive/YOLOv8_Fault_Tolerance/models/yolov8s.pt`
- Local fault-injection modules: `code/src/`
- Tracked research outputs: `code/results/fault_injection/`
- COCO evaluation artifacts: `code/results/official_cocoeval/`
- Baseline validation outputs: `code/results/baseline_validation/`
- Historical drafts and backups: `notebook_backups/`

Some large or intermediate artifacts are intentionally excluded from Git.
Their existence in Google Drive does not imply GitHub availability.

## 3. Recorded research environment

- Platform: Google Colab
- Recorded Python: 3.13.16
- Recorded PyTorch: 2.11.0+cu130
- Recorded Ultralytics: 8.4.189
- Recorded GPU: NVIDIA Tesla T4
- Dependencies: see `requirements.txt`

The requirements file is not a fully validated lockfile.
A platform-compatible PyTorch installation must be selected separately.

Some historical notebooks use Colab-specific APIs and absolute Drive paths.

## 4. Notebook roles

| Notebook | Recorded role |
| --- | --- |
| `00_environment_setup.ipynb` | Initial project and directory setup |
| `01_model_baseline.ipynb` | Initial YOLOv8s model baseline and repository work |
| `02_baseline_benchmark.ipynb` | Initial model benchmark |
| `03_baseline_validation.ipynb` | COCO 2017 baseline validation |
| `03_baseline_results_report.ipynb` | Read and report official COCOeval metrics |
| `04_fault_injection.ipynb` | Preliminary INT8 weight-level and boundary tests |
| `04_fault_injection_research.ipynb` | INT8/INT4 synthetic-input, pre-NMS research analysis |

This table describes notebook roles, not a guaranteed end-to-end
execution order. Historical notebooks contain environment-specific steps.

## 5. Model and data prerequisites

The YOLOv8s checkpoint is stored in the Drive `models/` directory.
The known recorded checkpoint has:

- Size: 22,588,772 bytes
- SHA256:
  `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`

A local test image is stored at `data/single_image_test/ultralytics_bus.jpg`.

Historical COCO validation used a runtime dataset location:
`/content/datasets/coco2017`.

The project-root `datasets/` directory was empty during the repository
structure audit. The full COCO dataset is not included in Git.

## 6. Recorded baseline evaluation

Two recorded baseline metrics must not be conflated:

- Ultralytics mAP50-95: approximately 0.44431391675
- Official pycocotools COCOeval AP50-95:
  approximately 0.4420484089204407

These were produced through different evaluation procedures.

The official COCOeval stored artifacts were checked for:

- 100 image-list chunks and 100 prediction chunks
- 5,000 unique image IDs, with no duplicates
- 570,662 prediction records
- No reported structural or numerical validation errors

This does not independently verify COCO ground-truth correspondence or
recompute the official AP metric.

See:
`docs/research_reports/COCOeval_Artifact_Integrity_Manifest_v1.json`

The manifest records paths, byte sizes, and SHA256 hashes for 203
artifacts, including three supplementary JSON files.

## 7. Fault-injection notebook provenance

### Preliminary INT8 notebook

`04_fault_injection.ipynb` documents preliminary INT8 bit-level
and boundary testing. It does not perform full INT8 model inference
or COCO evaluation.

Its historical source-module path uses `code_recovered/src`, which
does not match the currently inventoried repository layout.

A model-download statement also contains a Markdown-formatted URL;
the historical notebook should not be assumed to execute unchanged.

### INT8/INT4 research notebook

`04_fault_injection_research.ipynb` uses the current `code/src`
module location and the Drive `models/yolov8s.pt` checkpoint.

Its synthetic-input experiments concern pre-NMS predictions.
The historical INT4 synthetic fault-propagation result was recovered
from archived notebook outputs, not independently reproduced by
rerunning the inference.

The FP32 tensor-based quantization scale calculation was independently
checked against the historical numerical value. This narrower
verification does not establish complete inference reproducibility.

The statement about absent COCO mAP and real-image fault evaluation
in the notebook introduction describes that notebook's historical
scope, not the entire later project.

## 8. Notebook 04 extended research results

Later Notebook 04 research artifacts include exploratory INT4
weight-fault experiments on a real test image, with NMS-aware analysis.

These single-image results do not constitute COCO-scale fault-injected
mAP measurements or a completed fault-tolerance intervention.

See the Notebook 04 closure report and its accompanying artifact index
in `docs/research_reports/`.

The historical-manifest integrity audit identified eight mismatched
references across five unique CSV files. All were explained by
CRLF-to-LF line-ending normalization, and historical SHA256 values
were reproduced without changing research data.

See:
`docs/research_reports/Notebook_04_Integrity_Audit_v1.json`

## 9. Reproduction limitations and safeguards

- Preserve original historical notebooks and manifests.
- Do not overwrite validated research artifacts.
- Do not equate stored-output validation with rerunning inference.
- Do not equate single-image fault propagation with COCO mAP.
- Verify checkpoint hashes and artifact manifests before reuse.
- Review notebook paths and installation commands before execution.
- Do not assume historical Git-management cells are safe to rerun.
- Use explicit, versioned records for any future experiment changes.

## 10. Current status

The repository's documented baseline and Notebook 04 exploratory
research artifacts have undergone structural and provenance audits.

A full end-to-end experimental reproduction, a COCO-scale
fault-injected mAP study, and the proposed hybrid fault-tolerance
framework are not established by these audits.

This guide is a documentation draft pending review and publication.
