# A05 Phase 1A — Engineering Cohort Amendment (2026-11-23, v1)

**Status:** Versioned, evidence-based **cohort correction / proposed execution protocol**. **No 18-event run has been performed by this amendment.** Original A05, original split manifests, code, model parameters and existing results are unchanged.

## 1. Why this amendment exists

[A05 activation-injection specification](a05_activation_injection_specification_v1.md), §11.1, proposes **three exact YOLOv8s Conv-wrapper targets × first three sorted IDs from an older 50-image pilot × two independent repeats = 18 scheduled image-fault events**. The original rule selects **8844, 17627, 36660**. A later, independently versioned Notebook 08 COCO val2017 split assigns **8844** and **17627** to its 4,000-image nominated `heldout` partition and **36660** to its 500-image `development` partition. Proceeding under the original pilot selection would use newly reserved held-out IDs for ongoing engineering development.

The earlier pilot contains **50 IDs**, overlapping the newer split as **6 calibration / 4 development / 40 nominated held-out**. The older 1,000-image development set and the 200-image 05b screening subset also overlap nominated held-out images. Consequently the new `heldout` designation is a *forward-looking usage restriction*, **not proof of no historical exposure or independent testing**. This amendment does not certify independence; a separate historical-exposure crosswalk and final evaluation policy are still required.

## 2. Grounding and immutable selection rule

Authoritative file inputs (unchanged):

- Older pilot: [`experiments/notebook05/coco2017_split_manifest_v1.json`](../../experiments/notebook05/coco2017_split_manifest_v1.json), `image_ids.pilot`.
- Current split: [`docs/research_reports/notebook08_fixed_point_v1/coco_val2017_split_v1.json`](notebook08_fixed_point_v1/coco_val2017_split_v1.json), `splits.development.image_ids` and `splits.heldout.image_ids`.
- Parent specification: [`a05_activation_injection_specification_v1.md`](a05_activation_injection_specification_v1.md), especially §§3–8 and 11.1.

Apply **set intersection** of old `pilot` IDs with the current `development` IDs, sort numerically ascending, and take the first three. This is a reproducible deterministic rule; **do not select on fault effects, image difficulty, detection performance or annotations**.

Verified intersection from the committed manifest contents: **[36660, 96493, 298697, 533855]**.

**Amended Phase 1A three-image cohort: [36660, 96493, 298697].**

The intersection is disjoint from Notebook 08 `heldout`. The candidate rule was documented *after* an exploratory one-image smoke test on **36660**, so it must not be described as fully prospective or independent. This is acceptable only for **engineering acceptance**, not for held-out efficacy, vulnerability rankings, or a claim about generalization.

### Colab file availability / provenance checks

The researcher executed a read-only Colab preflight and reported that each selected COCO val2017 image exists and has the following SHA256. These hashes are **runtime-reported, not independently recomputed by the GitHub connector**, and must be asserted again immediately before any campaign.

| COCO image ID | JPEG SHA256 (user-reported Colab) | New partition |
|---|---|---|
| `36660` | `331fa26c3a7931f3752e56783e16f147d9b0e25cb63f4ea01cb4aea2d7e3e1ed` | development |
| `96493` | `cecec146af7564d8229b996b99d25289924c039bf41475b0c533909699b39351` | development |
| `298697` | `0fb0737fc0aaf2dc090149d77b0bce905b157375aa4b606b575d771ef658ac84` | development |

Local preflight message: **`ENGINEERING IMAGE PREFLIGHT: PASS`**. This proves neither annotation/preprocessing integrity nor image innocence from historical development exposure.

## 3. What stays fixed

Only A05 §11.1's **three image IDs** change. No other experimental lever is authorized to change:

- Pretrained **YOLOv8s**, original FP32 raw-model forward, verified unfused Ultralytics Conv wrapper **output after SiLU**, no substitution of leaf Conv2d pre-BN output.
- Exact canonical targets `model.0`, `model.5`, `model.15.cv2`; all observed in a clean CPU graph check as genuine Conv+BN+SiLU wrappers. Synthetic 640×640 clean observer forward yielded exactly one call, finite FP32 outputs and expected target shapes; no hook remained.
- Fault: transient **single-value replacement with +0.0**, exactly one addressed coordinate/event per faulty image, retaining true zero-to-zero no-ops. This is **not a bit flip** and cannot be presented as an empirical hardware fault distribution.
- A05's event seed **20261116**, canonical event-key/PRNG derivation, repeat identity, state machine, cleanup, and pre-frozen coordinates (do not choose an alternative location based on observed results).
- Exactly **3 targets × 3 development images × 2 repeat IDs = 18 scheduled faulty image events**. All 18 event records, including possible zero effect, failed or incomplete records, must remain accounted for. Clean/post-clean/observer controls and timing replays are separate and **not** counted as scientific fault events.
- Identical 640×640 LetterBox preprocessing, paired raw output and NMS conventions (`conf=0.001, iou=0.7, max_det=300, nc=80`), explicit identity and checksum validation, guard against automatic graph fusion, and separation of infrastructure failures from observed detector outcomes.
- Compute AP only on a complete, protocol-compliant image sweep when possible, labelled **descriptive engineering-subset AP**, never as population reliability or independent held-out effectiveness. Maintain no-AP status for incomplete sweeps.

## 4. Existing exploratory evidence is not the 18-event campaign

A *separate*, previously executed single-development-image smoke test used image **36660**, `model.0`, fixed coordinate `(0,0,0,0)`, +0.0 replacement; it produced one event, changed decoded raw output and tiny post-NMS numerical differences, and restored model state/clean outputs. After same-class IoU≥0.50 matching, **18 clean and 18 faulty detections matched**, with no unmatched detections; min IoU approximately **0.99999809**, maximum absolute confidence change **1.32024e-05**, max center shift **0.00026074 pixels** on the 640×640 input. The single-image event was **not seeded through the full frozen A05 event manifest**, was run after exploratory code development, and **must not be counted toward the future 18 scheduled slots**.

Differences in raw tensor elements are not equivalent to misdetections, errors against ground truth, AP degradation or mitigation gain. No AP, overhead or model-wide robustness estimate is established here.

## 5. Remaining execution gates / restrictions

- **Before** 18-event execution: code implementation and isolated correctness tests for deterministic coordinate manifest, one-shot scheduling, no-op and nonfinite handling, exception-safe hook ownership/cleanup, model-state invariance, exact graph/output contracts, clean replay stability, and saved raw/NMS evidence. An observation-only clean Hook probe and a one-shot smoke do **not** satisfy all A05 §13 gates.
- Lock and checksum the **actual preprocessed input tensors**, exact checkpoint, installed package/source versions, annotation identity, target graph fingerprint, frozen 18-event manifest and intended storage/compute budget. A05 phase 1A is a *proposal*, not evidence that these gates have passed.
- Use **current development only** for engineering work; do **not** sample nominated held-out to select or refine the method. Resolve historical overlap and independent final confirmation with a separate approved policy before claiming independent testing.
- Do not edit or reinterpret historical dataset manifests or old results. Record any further changes by a separately versioned amendment, with no silent scenario replacement.

## 6. Delivery accounting

**New artifact:** this amendment only. Historical A05 and both split manifests remain intact. No new inference, injection, automatic repair, metric rerun, notebook update or local Colab synchronization was performed while drafting and publishing this document.

**Next action after review:** implement and unit-test the A05 scheduling/hook lifecycle contracts on synthetic CPU data, then freeze the 18-event manifest *before* executing the amended three-image engineering pilot.
