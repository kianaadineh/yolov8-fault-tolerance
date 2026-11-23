# YOLOv8s Fault Injection and Fault Tolerance Research

**MSc research — University of Tehran | Documentation status: 2026-11-23**

Study of **pretrained Ultralytics YOLOv8s** under injected faults, emphasizing complete object-detection output, NMS-aware evaluation, reproducibility, and resource-constrained mitigation. We use the existing YOLOv8s implementation; this repository does **not** rewrite YOLOv8 from scratch. A final, independently validated hybrid fault-tolerance method has **not yet** been demonstrated.

> **Evidence convention:** implemented code, stored experimental records, successful functional tests and validated generalizable gains are different levels of evidence. Dates and limitations below refer to the linked source artifacts.

## Research model and fault assumptions

| Property | Recorded setup |
| --- | --- |
| Model | YOLOv8s pretrained checkpoint, 11,166,560 parameters |
| Eligible convolution modules / weights | 63 Conv2d / 11,146,080 Conv2d weights |
| Checkpoint SHA256 | `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a` |
| Fixed framework baseline | Ultralytics **8.4.189** |
| Dataset | COCO 2017 `val2017`, 5,000 images |
| Historical research GPU environment | Python 3.13.16, PyTorch 2.11.0+cu130, NVIDIA Tesla T4 |
| Other recorded sessions | Python 3.13.18, PyTorch 2.14.1+cpu; PyTorch 2.14.1+cu130 also appears in stored GPU records |
| Primary research outcomes | Raw/decoded vs post-NMS output, COCOeval AP/AR, fault consequences, corrective effect, protection costs |

**Important:** Current 05b/06b weight experiments perturb **persistent signed logical INT8 bit positions** in per-output-channel-quantized Conv2d weights. They **dequantize weights and execute FP32 computation**, with FP32 activations. This is *not native INT8 hardware inference*, a physical hardware failure-rate estimate, or an activation-fault study. The early INT4 work is separately scoped.

## Current experimental status

| Workstream | Source evidence | Status and boundary |
| --- | --- | --- |
| Clean YOLOv8s baseline | [Experimental evaluation](paper/experimental_evaluation.md) | Stored Ultralytics validation and separate official 5,000-image COCOeval baselines |
| Notebook 04 INT4 injection and NMS | [Closure report](docs/research_reports/Notebook_04_Closure_Report_v1.md) and [results](results/fault_injection/) | 36 initial INT4 cases, 96 extended cases and 12 additional preliminary TMR cases; exploratory |
| Logical INT8 bit-injection and checks | [05b runner](src/screen05b_campaign_v1.py), [validation](src/fault05b_validation_v1.py), [methods](docs/research_reports/05b_campaign_runner_notes_v1.md) | Implemented reusable path; exact runtime and recorded artifact identity matter |
| Notebook 05b whole-network INT8 screening | [2,198-case status](docs/research_reports/notebook05b_cpu_replay_v1/campaign_status_02198_v1.json) | **2,198 / 5,040** official campaign evaluations recorded; **2,842 remaining**. No completed 63-layer ranking. The separate CPU replay of case 2,199 is *not* an official campaign completion |
| Notebook 06a contextual bit completion | [Feasibility report](docs/research_reports/notebook06a_feasibility_v1/feasibility_report_v1.md) | Masked-bit prediction only; not unknown-fault detection, automatic repair, or AP improvement |
| Notebook 06b detection/repair study | [Pilot](docs/research_reports/notebook06b_pilot_v1.md), [end-to-end summary](docs/research_reports/notebook06b_end_to_end_v1/summary_v1.json) | Known-target C/D repair tested; **16 exploratory fault scenarios** evaluated after NMS/COCOeval. No generalizable protection benefit established |
| Notebook 07a invariant protection | [Functional baseline](docs/research_reports/notebook07a_baseline_summary_v1.json), [cost/coverage study](docs/research_reports/notebook07a_design_comparison_v1/summary_v1.json) | Functional bit-recovery and memory-budget comparisons. Not a matched-budget AP superiority or measured hardware-overhead result |
| Notebook 08 dataset split | [Frozen split manifest](docs/research_reports/notebook08_fixed_point_v1/coco_val2017_split_v1.json) | 500 calibration / 500 development / 4,000 nominated held-out, seed 20261121; **historical image exposure must be audited** before an independence claim |
| Activation injection | [Proposed next protocol](docs/research_reports/2026-11-23_research_direction_and_execution_protocol_v1.md) | No completed **repository-verified** YOLOv8s activation campaign in the published evidence reviewed |
| Final hybrid protection + resource cost | [Research plan](docs/research_reports/2026-11-23_research_direction_and_execution_protocol_v1.md) | Pending design freeze, budget-matched tests, independent AP evaluation and runtime/memory measurements |

**06b interpretation:** across its 16 selected end-to-end cases, reported mean *signed* AP gain in **percentage points** relative to faulty inference was **+0.0004133 for C** and **−0.0014997 for D**. These effects are tiny, condition-specific, and not evidence of deployment-wide superiority. Exact recovery of a perturbed weight is not guaranteed to improve sample AP.

## Clean baseline: do not conflate pipelines

| Metric | Ultralytics validation (YOLO labels) | Official `pycocotools` COCOeval |
| --- | ---: | ---: |
| mAP/AP50:95 | **0.444314** | **0.442048** |
| mAP/AP50 | 0.611230 | 0.604912 |

Both are 5,000-image fault-free references using **different evaluation procedures**. The pilot studies use their own image cohorts and cannot be numerically compared without matching all protocol choices. See [experimental evaluation details](paper/experimental_evaluation.md).

## Repository map

| Location | Contents |
| --- | --- |
| [`notebooks/`](notebooks/) | 16 tracked notebooks, **00 through 07a**; setup, baseline, INT4/INT8 screening, feasibility, repair, protection |
| [`src/`](src/) | INT4/INT8 bit utilities and versioned 05a–07a runners, evaluation and correction code |
| [`tests/`](tests/) | Four tracked Python test modules; **not freshly executed** in this documentation audit |
| [`experiments/`](experiments/) | Frozen/versioned sample plans, protocols, decisions and manifests |
| [`docs/research_reports/`](docs/research_reports/) | Evidence reports, integrity checks, partial campaign progress, hypotheses and limitations |
| [`results/`](results/) | Selected smaller published metrics and INT4 artifacts; many large archives remain outside Git |
| [`paper/`](paper/) | [Experimental evaluation](paper/experimental_evaluation.md) and [Chapter 1 draft](paper/thesis/chapter01_introduction_draft_2026-11-23.md) |
| [`requirements.txt`](requirements.txt) | Research dependencies; **not a validated full environment lockfile** |

For historical Notebook 05 snapshot details, use the [Notebook 05 interim checkpoint](docs/research_reports/notebook05_checkpoint_v1/README.md). Its 60-case checkpoint concerns an **earlier separate exploratory plan**; do not confuse it with the later 05b 5,040-case plan.

## Reproduction and data safety

1. **Verify the exact experiment**: its checkpoint and manifest digests, image IDs, sampling plan, fixed version, seed, injection scope, device and NMS/COCOeval configuration.
2. **Locate external assets**: `yolov8s.pt` and the COCO files are not committed. Recorded Google Drive project root: `/content/drive/MyDrive/YOLOv8_Fault_Tolerance`; repository is in its `code/` subdirectory.
3. **Check environment, then do one smoke test**: `requirements.txt` pins Ultralytics 8.4.189 but not a GPU-compatible PyTorch build. A fresh CPU-only Colab without Ultralytics cannot run the YOLO inference pipeline. Historical notebooks may contain installs and side-effecting write cells: **do not blindly Run all**.
4. **Protect stored evidence**: large per-scenario predictions are mostly untracked/ignored. Verifying a summary JSON does not verify every external prediction archive.
5. **Protect local work while syncing Git**: use `git status` and a **read-only untracked-file inventory** before staging, pulling, merging, resetting, or cleaning. Colab `HEAD` and remote tracking data may lag GitHub's newest documentation commits.
6. **Separate development from final testing**: earlier 05b/06b image cohorts are **not automatically equivalent** to the later 500/500/4000 split. Audit image overlap before asserting a truly untouched held-out set.

Background records: [Reproducibility Guide v1 (historical)](docs/research_reports/Reproducibility_Guide_v1.md) and [current repository documentation/evidence audit](docs/research_reports/2026-11-23_repository_documentation_audit_v1.md).

## Roadmap and active work

- [ ] Finish current runtime/provenance checks, 05b completion inventory and historical-to-new-split image crosswalk.
- [ ] Verify injected site, once-only application, model restoration, and clean-vs-clean output stability.
- [ ] Validate an activation-hook feasibility path **only if needed** for the hybrid hypothesis; distinguish SiLU output and tensor semantics.
- [ ] Use matched post-NMS COCOeval, explicit DUE/SDC definitions and uncertainty for fresh, prespecified confirmation cases.
- [ ] Select **one** evidence-supported, budget-constrained primary mitigation; compare with unprotected, random and sensitivity-only allocation at equivalent resource budgets.
- [ ] Freeze the method before independent testing; report clean accuracy and measured memory and latency. Energy requires actual instrumentation.
- [ ] Write thesis chapters and citation checks **in parallel** with experiments.

[Current Research Sprint / Checklist — Issue #1](https://github.com/kianaadineh/yolov8-fault-tolerance/issues/1) · [Research execution protocol](docs/research_reports/2026-11-23_research_direction_and_execution_protocol_v1.md).

### Known limits and non-claims

The recorded 05b campaign is partial. Current correction rates use distinct bit coverage and trusted-information assumptions; they are not directly comparable as a general winner. Training, activation clipping and a zero/near-zero-overhead hybrid were proposed, not established. The present repository does **not** establish native INT8 hardware reliability, large-scale fault tolerance, truly unseen final validation, absolute fault rates, field safety or measured energy reduction.

*This file is an evidence-grounded **2026-11-23 status snapshot**. Link new immutable results and review their scope before updating claims.*
