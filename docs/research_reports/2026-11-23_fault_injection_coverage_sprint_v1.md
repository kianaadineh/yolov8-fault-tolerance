# Fault-Injection Coverage Sprint — YOLOv8s
**Planning date:** 2026-11-23. **Execution window:** Tuesday–Friday, 2026-11-24 to 2026-11-27.

**Status:** PROPOSED RESEARCH EXECUTION PLAN, **not a completed experiment**. This supersedes the November 23 sprint *priority*, not the original A05 specification, its safety gates, or historical experimental records. Thesis mitigation/protection implementation is intentionally deferred to the following week, after an evidence-based risk/coverage review. No notebook, source, weights, dataset or results are modified by this planning document.

## 1. Week-end target and meaning of "complete"

The objective is **complete, auditable *design coverage* of the major YOLOv8s fault-injection surfaces, with validated representative probes across those surfaces where feasible**. It is **not** exhaustive enumeration of every weight, activation, bit, image, instruction, or fault lifetime. A statistically robust complete network-wide vulnerability ranking, multi-dataset held-out confirmation and hardware fault-rate certification cannot honestly be promised in four days.

Friday acceptance should distinguish:
- **INVENTORIED:** surface located, tensor/representation and call boundary fingerprinted, correct inclusion/exclusion rationale.
- **IMPLEMENTED + UNIT-VERIFIED:** the specified fault operator can deterministically change intended tensor bits/values, preserve non-target state, and clean up on forced failure.
- **ENGINEERING TESTED:** a precommitted, one-event real/synthetic sample was executed and checked against clean/post-clean output; saved raw/NMS evidence.
- **SCREENED:** a frozen, stratified development-image sample with prespecified repeats and denominators was executed (or explicitly reported partial). Not a final unbiased ranking.
- **NOT SUPPORTED / DEFERRED:** reason, scientific risk, and required future implementation. Do not turn a missing result into an unmeasured PASS.

## 2. Coverage matrix — distinct fault surfaces and models

| Stratum | Targets / boundaries | Models to consider | Minimum honest Friday deliverable |
| --- | --- | --- | --- |
| W1 — learned Conv weights | Backbone, neck, Detect box/class branch Conv2d weights; existing 63 eligible modules excluding fixed DFL Conv | Existing **logical INT8** 1-bit change with FP32 dequantized inference; separate original **FP32 sign/exponent/fraction** single-bit arm if validated; persistent single-weight mutation, restored after scenario | Verify existing 05b path/identity and produce stage/bit coverage ledger; small new *frozen* representative replay only if resource gate passes; do not rebrand logical INT8 as native INT8 |
| W2 — other state | Eligible bias, BN affine and running buffers (if model is unfused); fixed DFL coefficients only under a dedicated study | Exact-type bit/value changes, with persistent-vs-transient semantics separately specified | Explicit population and risk inventory; safe representative test only where semantics and restoration are verified; no blanket “all weights covered” |
| A1 — Conv-wrapper features | Early/mid/late backbone and neck at post-SiLU output; pre-BN leaf Conv2d only as a separately labelled boundary | **+0.0 one-value replacement** (A05 gate), then validated **FP32 single-bit** sign/exponent/fraction; line/block spatial replacement as separate correlated models | Finish A05 engine and amended 18-event engineering check first; additional fault types require versioned designs and tests |
| A2 — composite and routing features | C2f, SPPF, Upsample, Concat and feature fan-out | Output-tensor transient replacement/bit flip if single-tensor boundary is verified; reject ambiguous/multiple-input hooks | Architecture-wide module/output tree manifest, one safe representative probe per supported family; note each unsupported branch |
| H — detection and postprocess | Detect box/class prediction branches at P3/P4/P5; DFL projection/decode; decoded pre-NMS tensor; NMS output stage | Branch-aware tensor perturbations, finite/nonfinite propagation, and optionally postprocess corruption as an **explicit separate pipeline fault** | Do not assume head leaf hook == feature wrapper hook; separate adapters and controls; prove GT-free NMS differences and label the boundary accurately |
| I — input / preprocessing | RGB input, resize/letterbox, coordinate transforms | Optional input-value/bit corruption, not the same estimand as in-network transient faults | Record verified uncorrupted preprocessing, choose inclusion/exclusion explicitly; never pool input corruptions with internal activation sensitivity |

**Important:** Some operators (Concat, Upsample, SiLU) have no trainable weights. They can be *activation* fault surfaces without being *weight* fault surfaces. NMS belongs to the detector pipeline, not the learned neural network, and must not be conflated with convolution vulnerability. Different representations/boundaries/fault lifetimes require separate comparisons; pooled percentage “fault vulnerability” without an estimand is invalid. SiLU has a potentially shared activation instance in the installed Ultralytics code; parent-wrapper targeting and module alias checks avoid accidental multi-site targeting.

**Fault taxonomy:** persistent vs transient; single-bit sign/exponent/fraction FP32 vs logical signed-INT8 sign/magnitude; one-value erasure versus large/out-of-range replacement; single site vs spatial line/block/multi-site correlated faults; changed/no-op, finite/nonfinite, DUE/integrity failure. There is no physical SRAM BER or native INT8 proof here. **Do not try all Cartesian products**; publish a justified stratified coverage design and per-stratum denominators.

## 3. Four-day work and daily exit gates

### Tue 24 — engine, precise graph and safety
- Safely sync Colab from verified local `df5ea5d` to GitHub including the [A05 cohort amendment](a05_phase1a_cohort_amendment_2026-11-23_v1.md); do not overwrite local results or use `reset --hard`/`clean`.
- Read-only inventory of raw YOLOv8s graph including 64 Conv2d, 57 BN under current un-fused CPU snapshot, 63 prior weight-eligible Conv2d, aliases, actual module calls, activation boundaries and Detect/DFL output types; checkpoint/source/input provenance and clean replay.
- Implement in **new code/tests** a deterministic one-shot fault event scheduler and exception-safe hook/context scope; test one-event, no-op, replay ordering, forced-exception cleanup, state restoration and exactly-once semantics; no unreviewed historical changes.
- Freeze the amended 18-event A05 engineering manifest: 3 Conv-wrapper targets `model.0`, `model.5`, `model.15.cv2`; images `36660, 96493, 298697`; two independent repeat IDs, frozen image SHA256 and coordinate RNG, zero replacement at `conv_block_out`.
- **Exit:** source + tests + versioned complete target/fault inventory + event manifest and test outputs; otherwise mark gate incomplete.

### Wed 25 — activation fault arm
- Conditional on Tuesday safety gates: execute exactly the **18 A05 events** and clean/replay/post-clean controls, retaining zeros/no-ops and error/infrastructure outcomes. Save paired raw/NMS and event records.
- Extend by **new, separately versioned** and unit-tested arms: one true FP32 bit flip (explicit sign/exponent/fraction positions, exact uint32 XOR, nonfinite policy); one line/block replacement geometry; no silent reuse of A05 zero-replacement labels.
- Verify representative clean/fault pairs at backbone, neck, C2f/SPPF and routing boundaries; inspect shared objects, repeated calls, data-tree outputs before attachment.
- **Exit:** validated A05 engineering report plus bounded activation coverage ledger; if an adapter fails, produce explicit unsupported/partial status rather than quietly omitting the family.

### Thu 26 — weight, Detect/DFL, boundary separation
- Audit and reuse existing 05b weight-fault source/results. Frozen 05b plan: **5,040** scenarios; latest verified status **2,198** official completed and **2,842** remaining. Do not claim the ranking complete or substitute the separate CPU replay.
- Add/verify explicit **weight FP32 bit-level** versus **logical INT8** fault semantics where feasible; distinguish persistent parameter mutation from transient intermediate-state mutation, and check exact restoration before/after paired outputs.
- Exercise representative weight sites in backbone/neck/Detect (box/class branches) and a carefully adapted head/DFL activation boundary; treat DFL fixed coefficients/bias/BN under their own policy.
- **Exit:** stratified layer/bit/fault-model design, verified representative head and weight cases or documented blockers; report resource estimates. Do **not** promise the other 2,842 × 200-image 05b evaluations on CPU this week.

### Fri 27 — bounded multi-surface screening, analysis, freeze
- After resource and integrity gate, run the **largest prespecified feasible development-only, balanced stratified campaign**, with explicit layer/bit/fault-model/image/repeat denominators. Do **not** select faults after viewing failures, expand a preferred site, or stop selectively; incomplete sweep = PARTIAL.
- Evaluate **both** raw decoded boxes/class scores and post-NMS matched class/IoU/center/score differences, unmatched detections, and nonfinite/DUE outcomes. Official COCOeval bbox AP/AP50/AR is only computed for **complete matched sweeps** with correct GT/category IDs; no per-image AP shortcut and no surrogate from detection-count equality.
- Generate reproducible figures/tables: stage × model × boundary coverage, screened vs unrun, no-op/changed/nonfinite/DUE fractions, post-NMS damage and uncertainty where supportable, timing and storage. Keep change in output distinct from actual correctness loss.
- Crosswalk previous 05b/06b image exposure to Notebook 08 current 500 calibration / 500 development / 4000 nominated held-out split; the new held-out has known overlap with historical pilot/screening subsets and is **not certified independent**.
- **Exit:** one complete **fault-injection evidence dossier** with machine-readable manifests/results, source/test commits, coverage gaps, scientific conclusions bounded to observed settings, and a proposed fault set/risk definition for *next week's* mitigation decision. No protection mechanism or efficacy claim is required this week.

## 4. Dataset, resource and inferential gates

- All **new exploratory development** runs use the current 500-image development partition (or a deterministic subset thereof); no adaptive use of nominated held-out. The amended three engineering images were checked in Colab: `36660` SHA256 `331fa26c3a7931f3752e56783e16f147d9b0e25cb63f4ea01cb4aea2d7e3e1ed`, `96493` SHA256 `cecec146af7564d8229b996b99d25289924c039bf41475b0c533909699b39351`, and `298697` SHA256 `0fb0737fc0aaf2dc090149d77b0bce905b157375aa4b606b575d771ef658ac84`.
- One previously observed `model.0` +0.0 single-image 36660 smoke had **18/18** same-class NMS matches and sub-millipixel shifts; it is **not** one of the new frozen A05 18 slots and does **not** establish broad tolerance.
- CPU/GPU selection: CPU for contracts/hook lifecycle; scale up only after actual timing, memory and available-GPU/quota check. Do not promise a run duration or use an unbounded full-layer/full-image grid.
- The 05b 200-image screening cohort, older 1,000 development set and new split have different provenance. Re-evaluation requires protocol-matched image IDs and cannot silently pool old AP.
- Never edit frozen experiment records; append versioned artifacts and preserve SHA256/hashes. Inference/code changes and thesis prose are documented separately.

## 5. Next week (from Mon 30 Nov): fault tolerance / mitigation

Use Friday's **measured** failure map—not an assumed rank—to choose a primary resource-budgeted mitigation and fair controls. Compare clean, unprotected-faulty, protected-faulty, random allocation and sensitivity-only allocation on identical faults/images; investigate activation protection if its measurable risk and overhead justify it. Evaluate true post-NMS/AP and cost, then freeze the design for independent confirmation. Writing Chapters 2–4 proceeds in parallel as available, but **no efficacy result is claimed before execution**.

**Change control:** This is a coverage-sprint plan, not an implicit rewrite of [A05](a05_activation_injection_specification_v1.md) or permission to expand A05 Phase 1B automatically. Every newly added fault class/target adapter needs an explicit versioned contract, unit test and execution gate.
