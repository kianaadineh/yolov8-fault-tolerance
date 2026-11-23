# Repository documentation audit — 2026-11-23 (v1)

**Scope:** GitHub `main` tree and selected primary documents / source / JSON evidence, at the latest available repository revision. **No Colab inference, unit-test rerun, full prediction-archive checksum pass, or local-file mutation took place.** This is a documentation/traceability review, not an experimental revalidation.

## Scope inventory

A recursive GitHub tree listing returned **431 entries** (including directories; not 431 manually reviewed files), non-truncated. Top-level source locations present: `.gitignore`, `README.md`, `docs/`, `experiments/`, `notebooks/`, `paper/`, `requirements.txt`, `results/`, `src/`, `tests/`. Representative source files, the main README, research guide, plans, pilot summary and recorded result statuses were fetched and examined. **This review does not assert manual line-by-line inspection of all 431 tree entries or verification of all stored results.**

| Area | Observation | Documentation action / status |
| --- | --- | --- |
| `README.md` | Described only 00–04 notebooks and INT4 pilots; said faulty-model COCO evaluation simply “planned”; omitted 05b, 06a/b, 07a, 08 | Replace with date-stamped evidence ledger, notebook map, setup and explicit limits |
| `notebooks/` | 16 tracked notebook files from 00 through 07a, including 05, 05a/b/c, 06a/b and 07a | Link examples; do not claim fully reproducible or safe to run all |
| `src/` | Recorded 05a baseline, 05b logical INT8 campaign, 06a/b feasibility and detection, 07a invariants, INT4/INT8 operations | Describe real implemented scope; no engine rewrite |
| `tests/` | Four tracked Python test modules | Document presence only; not freshly executed |
| `experiments/` | Versioned plans and manifests for 05–07a, including interrupted 05b | Preserve plans and sampling conditions |
| `results/` | Tracked baseline and INT4 evidence; large per-scenario archives may be external/ignored | Link to available evidence without claiming all predictions in Git |
| `docs/research_reports/` | Audit/feasibility/repair records and partial progress through 05b | Retain historic versions, publish current crosswalk |
| `paper/experimental_evaluation.md` | Ended at fault-free baseline and could be misread as the current entire research status | Add a clearly separate exploratory-study section; do not mix datasets |
| `paper/thesis/` | Chapter 1 draft newly tracked | Link and label non-final, bibliography/template pending |
| `requirements.txt` | Pins Ultralytics 8.4.189 and **unbounded** PyTorch install; historical baseline Python 3.13.16 / torch 2.11.0+cu130 | Retain as historical dependency record; do not pretend this is a tested lockfile |
| `.gitignore` | Ignores large checkpoints, datasets and most results | Preserve; excluded/ignored assets require external provenance |
| GitHub issues | Research Sprint Issue #1 created for checklists | Link README |

## Source-grounded scientific status

1. **Baseline:** stored 5,000-image Ultralytics validation mAP50:95 = 0.44431391675; separate official `pycocotools` bbox COCOeval AP50:95 = 0.44204840892. These are different evaluation pipelines, not a contradiction. See `paper/experimental_evaluation.md` and `results/official_cocoeval/yolov8s_official_cocoeval_metrics.json`.
2. **INT4 exploratory study:** Notebook 04 closure documents 36 initial and 96 extended scenarios across 12 selected layers, plus 12 separate preliminary TMR scenarios. Single-image/matched NMS does not establish 5,000-image AP.
3. **05b logical INT8 screening:** defined 63 Conv2d layers × 8 bits × 10 sampled weights = **5,040 planned cases** on fixed 200 development images. The archived status `docs/research_reports/notebook05b_cpu_replay_v1/campaign_status_02198_v1.json` reports **2,198 complete official evaluations**, **2,842 remaining**, **no final ranking**, and a separate CPU replay for S05B-02199 that **does not count** as an official campaign completion. Its prediction archive hashes were not all rechecked in that snapshot. Do not inflate completion from replay records or progress files.
4. **06a:** contextual masked-bit prediction feasibility, not unknown-fault localization or verified correction; see `docs/research_reports/notebook06a_feasibility_v1/feasibility_report_v1.md`.
5. **06b bit-repair pilot:** known target-weight location. C and D show functional recovery and false interventions on specified cohorts; neither is established as deployed fault localization.
6. **06b end-to-end pilot:** `summary_v1.json` reports **16 selected fault scenarios**, 17 unique states including clean, and post-NMS COCOeval of clean/faulty/C/D. Reported mean signed AP percentage-point gains vs faulty were **+0.0004132768 for C** and **−0.0014996570 for D**. These are descriptive, tiny, sometimes negative effects, **not population efficacy**.
7. **07a functional protection:** 16,061 common targets; invariant exact recovery fraction 0.01180655 vs C 0.17424195 vs D 0.12257954, but bit coverage and information/cost assumptions differ; no independent AP winner. Cost study reports candidate `g9_b4567` fits an engineering 12% *defined storage* budget (~11.19%) and is a **coverage/cost screening**, not deployed effectiveness or overhead-free inference.
8. **08 split:** COCO 2017 val2017 deterministic manifest seed 20261121, calibration 500, development 500, nominated held-out 4000. Earlier experiments may have exposed some of the nominated held-out images: audit exact ID overlap before *independent* testing claims.
9. **Activation/hybrid:** no tracked repository evidence examined here proves a completed integrated YOLOv8s activation injection campaign, a final hybrid method or actual zero-overhead performance. Two local untracked activation audit/spec documents exist in the user's Colab, but their contents are **not reviewed or published** by this remote audit.
10. **Quantization meaning:** logical INT8 per-channel Conv2d weights are dequantized for FP32 forward calculations in 05b; never call the current measured work native INT8 GPU kernel resilience.

## File/version safety

- At user's last **read-only Colab check**, local `HEAD` and locally cached `origin/main` were `86681f93322c450c1774bc3cb529c9c128592ed6`, behind newer GitHub documentation commits. This is a snapshot **before a verified local fetch**; do not assume the local remote-tracking ref reflects current GitHub.
- Three **untracked local** files were explicitly reported and not touched: `docs/research_reports/a04_activation_injection_audit_v1.md`, `docs/research_reports/a05_activation_injection_specification_v1.md`, `docs/research_reports/notebook07b_sign_protection_v1/single_image_smoke_v1.json`. Never auto-stage, force-pull, reset, clean, delete, or overwrite them.
- Historical published docs are left as historical records. In particular `docs/research_reports/Reproducibility_Guide_v1.md` is a time-stamped earlier audit; its old “current status” section is not a live project summary. Use README and this dated audit for the November 23 status.
- No existing numerical experimental artifact or versioned experiment plan was edited in this documentation pass.

## Required maintenance checks after each verified milestone

- Update README status and link primary evidence, not console recollection.
- Update `paper/experimental_evaluation.md` with **separately labeled** new experimental cohorts and numbers, without replacing historical baseline values.
- Update the GitHub sprint issue only when a task is actually verified; add commit URLs.
- For result rows, record data ID list/hash, environment, checkpoint hash, fault model, quantization semantics, seed, sampling population, inference/NMS/COCOeval settings, repeated controls, failure definitions, units and caveats.
- Before local Git synchronization, do a read-only inventory of local tracked/untracked changes, then fetch and validate divergence; never assume remote documentation commits imply the Colab tree is synchronized.
- Thesis chapters remain drafts until primary-literature and university LaTeX-template checks are completed.

## Next action

Carry out the local Colab fetch / divergence and untracked-file read-only review, followed by pinned runtime checks; then plan a versioned targeted single-fault test. Do not start a 5,040-case campaign or make claims about held-out independence in the same step.

**Source links:**
- [Primary README](../../README.md)
- [Baseline](../../paper/experimental_evaluation.md)
- [05b status](notebook05b_cpu_replay_v1/campaign_status_02198_v1.json)
- [06b pilot](notebook06b_end_to_end_v1/summary_v1.json)
- [07a design](notebook07a_design_comparison_v1/summary_v1.json)
- [08 split](notebook08_fixed_point_v1/coco_val2017_split_v1.json)
- [Research Sprint](https://github.com/kianaadineh/yolov8-fault-tolerance/issues/1)
