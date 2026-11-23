# YOLOv8s Fault Tolerance — Evidence Audit and Execution Protocol v1
Date: 2026-11-23
Status: DESIGN DRAFT — NOT A FROZEN EXPERIMENT; no inference was run for this document.
Repository baseline reviewed: 86681f93322c450c1774bc3cb529c9c128592ed6.
Document type: read-only audit + proposed protocol. Historical experiments, source code, and model checkpoint are not modified.

## 1. Thesis alignment and boundaries
Original University of Tehran thesis proposal: fault tolerance in YOLO architecture for resource-constrained computing systems. Its research objective is a hybrid, NMS-aware, low-overhead approach combining fault-propagation analysis, appropriate software protections, and resource-aware selection. 'Zero overhead' is a research aspiration, not a measured result. The proposal mentions fault-aware training, activation clamping, selective quantization, and multiobjective optimization. Any narrowing of implemented scope must be disclosed and discussed with the supervisor; it is not silently redefined as weight-bit protection only.

Fixed research model: pretrained YOLOv8s, checkpoint identity SHA256 1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a. Recorded Ultralytics version: 8.4.189. Preserve the exact package and model metadata in every execution. The current upstream Ultralytics main branch is not a drop-in versioned replacement.

Definitions: injected fault = a perturbation to a specified storage/computation state; error = resulting deviation in values/outputs. Emulated signed logical INT8 weight injection is not native INT8 hardware injection. Temporary activation corruption and persistent memory-weight corruption must be studied as different fault models.

## 2. Evidence ledger — what exists and what is not proven

| Capability | Evidence inspected in repository | Status and limitations |
| --- | --- | --- |
| Baseline YOLOv8s | README; paper/experimental_evaluation.md; docs/research_reports/Reproducibility_Guide_v1.md | Stored baseline and independent COCOeval path exist; this audit did not rerun inference. |
| Logical INT4/INT8 bit operations | src/int4_fault_injection.py; src/int8_fault_injection.py | Bit arithmetic and tensor utilities exist; not equivalent to arbitrary FP32 faults or physical bit-cell probabilities. |
| Full-network logical INT8 weight injection | src/fault05b_validation_v1.py; src/screen05b_campaign_v1.py | Designed for 63 Conv2d layers, 8 bit positions, persistent single-weight bitflip and full inference. |
| 05b screening | docs/research_reports/05b_campaign_runner_notes_v1.md; progress records | Defined 5,040 scenarios on a shared 200-image development sample. Published GitHub progress artifacts are partial; do not claim a complete 63-layer ranking from this audit. |
| NMS + COCOeval | src/end_to_end06b_v1.py; docs/research_reports/notebook06b_end_to_end_v1/summary_v1.json | 16-scenario exploratory end-to-end pilot documented. No population-level treatment benefit. |
| Protection candidates | docs/research_reports/notebook06b_pilot_v1.md; docs/research_reports/notebook07a_baseline_summary_v1.json | 06b target-known bit-repair tests and 07a functional checks exist; deployment-wide localization, independent AP benefit and actual overhead remain unproven. |
| Activation injection | README (planned); no verified integrated YOLOv8s activation campaign in inspected repository files | NOT VERIFIED. Assess hook semantics and output contract before introducing a new campaign. |
| Current final test split | docs/research_reports/notebook08_fixed_point_v1/coco_val2017_split_v1.json | 500 calibration + 500 development + 4,000 held-out, seed 20261121. Held-out use requires leakage/previous-use audit before claims of independence. |
| Runtime / energy | reports distinguish estimated storage from actual runtime | Not established by these repository records. Do not claim zero-cost or native hardware energy benefits. |

Note: the historical 05b/06b cohorts and newer November 21 split need an explicit crosswalk; results computed on different image subsets are not numerically comparable without controlling protocol.

## 3. Concrete research questions
RQ1 — Under explicit weight and activation fault models, which stage/bit/layer groups most often produce meaningful final detection failures, beyond local numerical deviation?
RQ2 — At equal protection budget and matched injected faults, does an NMS-aware risk score for selecting protected components outperform uniform, random, and sensitivity-only selection?
RQ3 — Does a minimal hybrid design (verified weight protection plus a separately validated activation safeguard or detection-boundary check) improve post-NMS AP and reliability without unacceptable clean-accuracy loss?
RQ4 — Do gains persist on a held-out subset with honestly measured memory and latency costs?

These are questions, not positive claims, and do not yet establish novelty relative to SentinelNN, ProAct, DeepVigor, or published detection-fault studies.

## 4. Fastest defensible experiment sequence
### Gate A — identity and compatibility, no new campaign
1. Verify checkpoint SHA256, Ultralytics 8.4.189, Python/PyTorch/CUDA, model module inventory, category mapping, clean inference output shape, preprocessing and COCO annotation identity.
2. Verify that the 05b/06b protocol and stored outcomes match their hashes. Inventory successful scenarios, failures, interruptions and any duplicated records; never treat partial progress as completion.
3. Map older 05b/06b image IDs against the newer 500/500/4000 split; record any held-out exposure. No claim of fully unseen data until this audit passes.
4. Create a clean-vs-clean control with equivalent preprocessing, NMS and repeated inference to establish numerical stability; treat environment changes as separate versions.
Exit evidence: read-only signed environment/coverage report and one verified clean prediction fingerprint.

### Gate B — targeted pilot, not population evidence
1. Reuse the already-validated logical INT8 weight-path and run a fixed, precommitted small set of cases spanning backbone, neck and Detect, including sign and non-sign bit positions. Use fresh scenarios where confirmation is needed; leave frozen 05b unchanged.
2. For activation faults, use a minimal PyTorch forward-hook feasibility test only after checking exact hooked module semantics, SiLU positioning, tensor shape/type, nested/multiple invocations, and restoration. Distinguish 'conv output' from 'activated block output'. A PyTorchFI adapter is optional, not presumed necessary.
3. Record raw pre-NMS tensor changes and post-NMS detection changes with identical image IDs/thresholds. Unexpected NaN/Inf, exceptions and invalid boxes become separately counted outcomes, not silently dropped samples.
Exit evidence: fault applied exactly once at intended site; clean state restored; forward output valid; on/off contrasts saved and independently checkable.

### Gate C — screening and confirmation
1. Preserve existing 05b historical scope. Where feasible, finish the frozen 05b campaign exactly as defined; if resources prevent completion, report partial coverage and do not publish final ranks as though complete.
2. Avoid arbitrary statistical stopping after seeing attractive results. Predefine sampling, layer/bit strata, seeds, event duration, fixed input IDs and sampling coverage.
3. Separate discovery on development data from confirmation with newly sampled faults and untouched images; estimate uncertainty over both sampled faults and images where possible.
4. Use metrics conditional on fault-model population; do not interpret uniform artificial bit selection as field hardware failure rates.
Exit evidence: reproducible candidate vulnerability groups, uncertainty, and raw/NMS failure categories with clear denominators.

### Gate D — choose ONE bounded primary mitigation plus fair controls
Primary candidate, subject to evidence: cost-constrained, detection-impact-aware selective protection of quantized weight regions. Do not invent a final algorithm before a risk ranking and cost model exist.
Hybrid extension: after Gate B, add one lightweight SiLU-compatible activation-range or prediction-boundary safeguard if clean-output impact is acceptable. Existing ReLU clamping methods cannot be copied blindly into SiLU YOLOv8s.
Controls: unprotected faulty model; random allocation at exactly equal budget; simple layer-sensitivity allocation at equal budget; a practical reference or full protection where a comparable budget is meaningful; clean protected model. Include an ablation separating weight protection from any activation/NMS component.
Exit evidence: fixed algorithm, protection selection source, accounting assumptions and immutable evaluation config.

### Gate E — final evaluation and thesis evidence
- Freeze method using only calibration/development records before accessing the nominated held-out cohort.
- Use paired model and image/fault scenarios across methods. Primary: COCO bbox AP@[.50:.95]; secondary AP50, AP75, AR100, size breakdown, count/class/IoU differences, missed/extra detections, SDC/DUE/non-finite rates.
- Preserve pre-NMS and post-NMS observations separately. Use image-level paired bootstrap by recomputing COCOeval where needed; account for fault-scenario sampling, not only images.
- Measure real wall-clock latency under matched device/warmup/batch settings; separately state serialized storage, resident RAM and extra compute. Energy claims require instrumented measurement.
- Report negative results, failure cases, clean-model degradation and any dependence on the selected detection thresholds.
Exit evidence: final paired tables/figures, audit-ready artifacts, limitations and no unsupported deployment claims.

## 5. Tools: adopt only after documented capability gap
- Current project runners: first-choice for already tested logical INT8 weights and COCOeval.
- PyTorchFI: optional activation injection adapter, after exact YOLOv8s hook test.
- PyTorchALFI: campaign design / object-detection logging inspiration; bundled Faster R-CNN adapter is not direct YOLOv8s compatibility.
- DeepVigor+: reference for scalable vulnerability analysis, but its classification-centric output does not replace detection AP.
- SentinelNN: selective hardening/duplication comparator; built for CNN/ReLU and must not be called directly SiLU-compatible.
- RReLU/ProAct/FT-ClipAct/Ranger: activation-range conceptual benchmarks; preserve YOLOv8s healthy behavior.
- FaultForge: resumable unbiased sampling and ECC reference ideas, optional; not a verified YOLOv8s engine.
- The YOLOv8n raw-PT-byte flip repo is not a suitable controlled weight-tensor injector.

## 6. Sources and provenance for review
Repository files:
- https://github.com/kianaadineh/yolov8-fault-tolerance/blob/main/src/screen05b_campaign_v1.py
- https://github.com/kianaadineh/yolov8-fault-tolerance/blob/main/src/end_to_end06b_v1.py
- https://github.com/kianaadineh/yolov8-fault-tolerance/blob/main/docs/research_reports/notebook06b_end_to_end_v1/summary_v1.json
- https://github.com/kianaadineh/yolov8-fault-tolerance/blob/main/docs/research_reports/notebook07a_baseline_summary_v1.json
- https://github.com/kianaadineh/yolov8-fault-tolerance/blob/main/docs/research_reports/notebook08_fixed_point_v1/coco_val2017_split_v1.json

Literature and implementation candidates (relevance confirmed from repository descriptions/README; this is not a complete full-paper systematic review):
- PyTorchFI: https://github.com/pytorchfi/pytorchfi — Mahmoud et al., DSN-W 2020.
- PyTorchALFI: https://github.com/tunghoang290780/pytorchalfi_fault_injection — object detection scenarios and Qutub et al., 2022.
- DeepVigor+: https://github.com/mhahmadilivany/DeepVigor — arXiv:2410.15742.
- SentinelNN: https://github.com/mhahmadilivany/SentinelNN — IOLTS 2024, DOI 10.1109/IOLTS60994.2024.10616072.
- RReLU/ProAct: https://github.com/hamidmousavi0/reliable-relu-toolbox — arXiv:2406.06313.
- FaultForge: https://github.com/rezzubs/faultforge — ECC experiments, different supported model bundles.
- Official engine: https://github.com/ultralytics/ultralytics
- Official COCO API: https://github.com/cocodataset/cocoapi

## 7. Explicit non-claims
No new Colab run, fault injection, runtime benchmark, energy measurement, or methods comparison was performed while writing this protocol. No existing code/protocol/experiment artifact was changed. Claims of actual YOLOv8s activation injection, a final hybrid mitigation, safety certification, zero-overhead, independent generalization, or improved mAP require future experiments and supervisor review.

## 8. Immediate next executable action
Perform a READ-ONLY current-runtime preflight against the exact pinned YOLOv8s checkpoint, installed libraries, dataset paths, code HEAD, and split manifest. Record PASS/FAIL and fingerprints; do not launch a new large fault campaign in the same step.
