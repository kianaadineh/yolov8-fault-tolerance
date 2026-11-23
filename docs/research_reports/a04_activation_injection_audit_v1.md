# A04 — Read-Only Audit of Activation Fault Injection

Audit date: 2026-11-16. Method: static source and saved-output inspection only.

**Verdict:** No integrated Activation Fault Injection implementation was found in the current GitHub repository or the inspected Drive code tree. A real earlier hook-based activation injector exists in `/content/drive/MyDrive/Colab Notebooks/project.ipynb`, outside the repository. It includes executable single-value, line and block replacement fault models, and saved evidence of single-value injection on YOLOv8n. It is a historical prototype, not a verified, reproducible YOLOv8s campaign tool. Its core tensor operations appear adaptable to ordinary YOLOv8s Conv2d outputs, subject to unresolved structural, fusion, targeting and lifecycle assumptions. Notebook 07b remains a separate Weight Fault Protection pilot.

## 1. Audit Scope and Repository Snapshot

Repository: `kianaadineh/yolov8-fault-tolerance`.

Pinned GitHub main snapshot:

```text
8425d543128dd115a810d6219b9cc39b2b6ecdc6
```

The recursive GitHub tree returned 426 entries with `truncated: false`: 399 blobs, including 398 readable text files and one PNG. All 398 text files were retrieved and searched, including all 23 Python files, 15 notebooks, README, requirements, documentation, research reports, experiment plans, CSVs and JSON results. Source cells were distinguished from notebook outputs. No retrieval failures occurred. The PNG was not treated as executable source.

Expected Drive code root, confirmed accessible through the Drive connector:

```text
/content/drive/MyDrive/YOLOv8_Fault_Tolerance/code
```

The Drive source/documentation tree was recursively listed under `src`, `tests`, `notebooks`, `docs`, `experiments`, `fault_injection`, `fault_tolerance`, `evaluation`, `training`, `losses`, `configs` and `paper`. This yielded 248 entries; 228 relevant text files were read and searched with no read failures. All common paths compared as decoded text matched the pinned GitHub contents. This is decoded-text comparison, not an independent byte-hash verification of every file. The extra Drive smoke report was included. Root README and requirements were inspected in the GitHub snapshot.

Supplementary search: 19 relevant working notebooks in `/content/drive/MyDrive/Colab Notebooks/`, plus nine notebooks in `/content/drive/MyDrive/YOLOv8_Fault_Tolerance/notebook_backups/`. Unrelated coursework notebooks were excluded. Drive `.git` internals, caches, datasets, checkpoint binaries, and the untracked/ignored detailed result tree were not exhaustively read; the tracked result tree on GitHub was searched. Completeness for arbitrary files outside these scopes is UNKNOWN.

This process had no live Colab shell. Its local working directory was `/workspace/scratch/013d81bc6172`, where the expected `/content/drive/.../code` path did not exist. Drive and GitHub were accessible through connectors, so the stored-source audit could proceed. Live runtime state, installed packages, model fusion state and local untracked-file completeness are UNKNOWN. No claim of live-runtime execution is made.

Evidence locations used below:

- `R/<path>` = `/content/drive/MyDrive/YOLOv8_Fault_Tolerance/code/<path>`, with GitHub source references anchored to the pinned commit above.
- `P` = `/content/drive/MyDrive/Colab Notebooks/project.ipynb`.
- `S` = `/content/drive/MyDrive/Colab Notebooks/07b_selective_sign_protection_pilot.ipynb`.
- Notebook cell references are 1-based document order, not execution counts; source-file line numbers are 1-based.

P identity at read time:

```text
Size: 87184 bytes
Drive modified time: 2026-07-20T21:13:20.777Z
SHA256: fb59323e42372a96ea8074461dcbac2d6a592065fe5cbb39b025acb2fab40a7f
Drive ID: 1tXKSDK2YnqGJAWKGYjBFYcj-yB6Ivg05
```

## 2. Search Results

Case-insensitive searches included `register_forward_hook`, `register_forward_pre_hook`, `forward_hook`, `fault_hook`, `activation fault`, `activation injection`, `activation protection` with whitespace/underscore/hyphen variants, `FaultInjector`, `FaultModel`, `SingleValueFault`, `LineFault`, and `BlockFault`. A broader repository code check for hook references, `register_*hook`, fault class declarations and injection function declarations also found no activation implementation.

| Location | Exact evidence | Classification |
|---|---|---|
| `R/README.md`, line 23 | `Activation fault injection | Planned` | Roadmap, not implementation |
| `R/README.md`, line 78 | `Investigate activation faults.` | Proposed work |
| `R/docs/research_reports/Notebook_04_Closure_Report_v1.md`, lines 107–108 | Activation faults require further study | Limitation |
| `R/docs/research_reports/05b_campaign_runner_notes_v1.md`, lines 6–12 | Weight-only campaign; activation faults excluded; computation and activations FP32 | Explicit scope |
| `R/experiments/notebook07a/invariant_bit_baseline_protocol_v1.json`, line 69 | Activation faults outside scope | Explicit scope |
| `R/results/fault_injection/weight_fault_scope_v1.json`, line 22 | Activation faults outside weight-only scope | Explicit scope |
| `R/results/fault_injection/int4_phase1_coverage_audit_v1.json`, line 26 | Activation fault sensitivity listed among limitations/future coverage | Not executable injection |
| `R/notebooks/07a_invariant_bit_protection.ipynb`, cell 2 | Embedded protocol excludes activation faults | Protocol generation, not activation injection |
| P, cells 14–15 | Forward observer hooks over Conv2d | Executable instrumentation |
| P, cells 18–21 | Tensor replacement and `FaultInjector` callback | Actual activation injector prototype |
| P, cells 27–31, 37–43 | Conv2d layer inventory and hook attachment | Actual layer selection |
| P, cells 51–61 | Fault interface plus single/line/block models | Actual replacement-model code |
| P, cells 65–71 | Experiment manager and saved single-value run | Historical executable experiment |

No `register_forward_pre_hook` implementation was found in the searched code. The repository negative finding is based on the complete tracked text-tree scan and Drive source scan, not on 07b or README alone. Backup matches contained embedded roadmap/scope text and audit/report-generation code, not a second activation injector. Working 04/04b matches likewise concerned weight-scope reports (cells 233, 250 and 277).

## 3. Existing Implementations

### 3.1. Observer hook — P, cells 14–16

```python
def hook_fn(module, input, output):
    global counter
    counter += 1
    print(counter, module.__class__.__name__, output.shape)

hooks = []
for module in model.model.modules():
    if isinstance(module, nn.Conv2d):
        hooks.append(module.register_forward_hook(hook_fn))
```

This is executable forward-output instrumentation, not fault injection: it prints shape and returns no replacement. Handles are retained in `hooks`, but no removal loop for that list exists in P. Observer prints continue in saved later inference outputs (P, cells 23, 45 and 71). P cell 16 generates an unseeded NumPy image. These observer hooks do not deliberately modify weights or activations, but they add logging and persistent callbacks.

### 3.2. Standalone draft and FaultInjector — P, cells 18–23

Cell 18 defines a standalone `inject(self, x)` function requiring `self.probability`, `self.fault_value` and a `random` binding; it is a draft helper, not independently wired to a hook. Cell 19 provides a complete class and callback:

```python
class FaultInjector:
    def __init__(self, probability=1.0, fault_value=0.0):
        self.probability = probability
        self.fault_value = fault_value

    def inject(self, x):
        if random.random() > self.probability:
            return x
        y = x.clone()
        b = random.randint(0, y.shape[0]-1)
        c = random.randint(0, y.shape[1]-1)
        h = random.randint(0, y.shape[2]-1)
        w = random.randint(0, y.shape[3]-1)
        y[b, c, h, w] = self.fault_value
        return y
```

The callback returns `injector.inject(output)`. Cell 21 attaches it to `model.model.model[0].conv`. Cell 23 has saved output `Injecting fault in Conv2d` and `Fault injected at (0, 5, 131, 64)`, followed by detector output. This is evidence of an earlier activation replacement run, not independent reproduction. The probability gate is per callback invocation, not per-neuron bit-error rate. It is not range-validated (cell 19).

### 3.3. Layer selector — P, cells 27–33 and 37–47

```python
for name, module in model.model.named_modules():
    if isinstance(module, nn.Conv2d):
        conv_layers.append((name, module))
```

`attach_hook(layer_index, hook_function)` indexes the global `conv_layers` and calls `layer.register_forward_hook(hook_function)` (cells 30 and 41). This is executable code, not pseudocode. Saved inventories list 64 Conv2d modules, including `model.22.dfl.conv` (cells 28 and 39); saved registrations select index 0 / `model.0.conv` (cells 31 and 43). Index selection has no explicit type/bounds check or stable name manifest. Python negative indices are accepted implicitly. Handles are returned and manually removed in cells 33 and 47; the first directly attached handle is removed in cell 26.

### 3.4. Modular fault models — P, cells 51–61

| Implementation | Reference | Actual operation | Saved execution evidence |
|---|---|---|---|
| `BaseFaultModel` | Cell 51 | `inject` raises `NotImplementedError` | Interface only |
| `SingleValueFault` | Cell 53 | Clone, select random N/C/H/W, set one element to `fault_value` (default 1000) | Single-value run in cell 71 |
| `LineFault` | Cell 55 | Clone, select random N/C/H, set entire width row | No dedicated line-fault run found; UNKNOWN |
| `BlockFault` | Cell 57 | Clone, select random N/C and top-left H/W, set square block (default 8×8) | No dedicated block-fault run found; UNKNOWN |

Cell 59 chooses `SingleValueFault()`. Cell 61 defines a hook returning `fault_model.inject(output)`. The classes are real Python implementations, but the statement in markdown cell 62 that all three were successfully implemented is not evidence that all three were executed successfully.

### 3.5. Experiment manager — P, cells 65–74

`FaultExperiment.run` sets global `fault_model_instance`, attaches a hook through the global selector, runs `self.model(image)`, removes the handle, and returns results (cell 65). Cell 67 defines the callback using that global instance. Cell 71 invokes index 0, `SingleValueFault()`, and `bus.jpg`.

Saved cell 71 output includes:

```text
Hook attached to: model.0.conv
Layer Index : 0
Fault Model : SingleValueFault
257 Conv2d torch.Size([1, 16, 320, 240])
Injecting fault into Conv2d
Single Fault -> (0,3,217,18)
...
Experiment Finished
```

The manager is executable historical code. It is not a self-contained library: it depends on prior cells, global lists and callbacks. `ExperimentLogger` only prints layer/fault labels; it does not persist seed, coordinates, model identity or metrics (cells 73–74).

## 4. Hook Mechanics and Injection Boundary

All injection registrations found in P are **forward hooks on leaf `nn.Conv2d` modules** (cells 21, 30 and 41). The injector consumes and returns the module's **output**, not the `inputs` tuple (cells 19, 61 and 67). No pre-forward input injector was found.

The saved architecture in P cell 8 shows the first Ultralytics `Conv` wrapper contains `.conv`, `.bn` and `.act` (`SiLU`). Therefore, **if that unfused structure is the executed structure**, the hooked boundary is after Conv2d and before BatchNorm and SiLU. This is a source-based conditional interpretation, not verification of the historical runtime's boundary.

The code calls the high-level `YOLO` prediction wrapper (cells 23, 45 and 71), but does not record/assert fusion state or module identity after predictor setup. If BatchNorm is fused into the convolution, a Conv2d-output hook observes the fused convolution output; a separate pre-BatchNorm boundary no longer exists there. Whether this happened in the stored runs or current YOLOv8s runtime is UNKNOWN. No fusion behavior is inferred merely from the initial printed architecture.

Leaf head Conv2d modules can have no following BatchNorm/SiLU; the layer inventory includes those and the fixed DFL projection (P, cells 28/39; contrast R weight scope JSON, lines 9–18). Thus “after Conv2d” cannot be generalized to “after every Conv–BN–SiLU block.” No hook on a complete C2f/Detect block or post-SiLU module is implemented here.

## 5. Fault Models and Tensor Representation

| Property | Evidence and assessment |
|---|---|
| Fault representation | Value replacement, not bit flipping: assignment in P cells 19, 53, 55 and 57 |
| Supported spatial patterns | Single element, full row, square block; references above |
| Sign/exponent/mantissa flips | No representation or implementation found |
| Activation quantization | No activation scale/zero-point/INT4/INT8 path in the injector |
| Dtype | Preserved by `clone`; no explicit required dtype or validation. Historical runtime output dtype is UNKNOWN |
| Shape | Assumes nonempty four-dimensional NCHW tensors; reads `shape[0]` through `shape[3]`; no rank/type checks |
| Device | Clone and scalar assignment use the output tensor's existing device; no explicit CPU/CUDA transfer. Actual current-device compatibility is NOT VERIFIED |
| Tuple/list outputs | Unsupported by these methods: they expect tensor `.clone()` and `.shape`; ordinary leaf Conv2d is the intended target |
| Block dimensions | Requires height and width at least `block_size`; no validation for too-large, zero or negative values (P cell 57) |
| Finite/range checks | None for fault value or post-injection output in these methods |
| Event lifetime | Callback runs for each target module invocation while attached; no one-shot flag, image ID gate, or warm-up exclusion (P cells 19, 65, 67) |
| Original activation | Injecting branches clone before assignment; skipped branch returns original input unchanged (P cell 19) |
| Original model parameters | No deliberate parameter mutation in activation injector bodies; no full state-hash/BN-buffer invariance proof (P cells 19, 53–57, 65) |

Generic tensor bit-flip helpers do exist in the repository: `R/src/int8_fault_injection.py`, lines 4–31, and `R/src/int4_fault_injection.py`, lines 5–39. They require `torch.int8` storage, clone a tensor and flip a selected bit. The INT4 helper restricts logical values to [-8, 7]. These helpers are not intrinsically limited to weights, but their project usage targets weights. Their existence is not evidence of an activation hook, an activation quantizer, or an end-to-end activation bit-flip experiment.

## 6. YOLOv8s Compatibility

**Static assessment:** the core clone-and-replace operation does not hard-code a channel count, so ordinary nonempty NCHW YOLOv8s Conv2d outputs are plausible targets. This is conditional technical plausibility, not confirmed operational compatibility.

| Component | Compatibility evidence / unresolved condition |
|---|---|
| Model identity | P cell 8 explicitly loads `yolov8n.pt`, not `yolov8s.pt`; saved first Conv output has 16 channels (cells 16/71) |
| Current project | R README lines 8–11 specifies YOLOv8s; requirements line 7 pins Ultralytics 8.4.189 |
| s-model evidence | R `notebooks/04_fault_injection_research.ipynb`, cell 9, expects first-layer weights `(32, 3, 3, 3)`; S cell 3 likewise uses YOLOv8s checkpoint and 32 first-layer output channels |
| Internal nesting | P cell 21 assumes `YOLO wrapper -> .model -> .model[0] -> .conv`; incompatible with directly substituting a raw DetectionModel into the same expression |
| Selector nesting | P cells 27/37 use `model.model.named_modules()`; the meaning of `.model` depends on whether `model` is a YOLO wrapper or a raw PyTorch model |
| Layer ordering | P cells 30/41 use traversal indices; names/counts/order must be re-established for the actual loaded model, not transferred from YOLOv8n by index |
| Predictor/fusion | No assertions that hooked module references survive predictor setup/fusion; exact current boundary UNKNOWN (P cells 21/23 and 65/71) |
| Version | P cell 6 installs unpinned `ultralytics`; no version assertion in that prototype. Current pinned-version execution is NOT VERIFIED |
| DFL | Selector includes the fixed DFL Conv2d, unlike weight campaigns' explicit exclusion; intended activation scope needs its own policy (P cells 28/39; R scope JSON lines 9–24) |
| Other backends | Code assumes in-process PyTorch modules; exported ONNX/TensorRT/compiled compatibility is UNKNOWN and not established |

No model was loaded to resolve these conditions. The prototype should not be called a verified usable current YOLOv8s implementation merely because it imports `YOLO`.

## 7. Implementation Quality and Risks

1. **No reproducible random stream.** Python `random.random/randint` is used for event/location sampling; no `random.seed`, private generator, frozen location manifest, or replay coordinates exist in P. NumPy synthetic-image generation is also unseeded (cells 16, 19, 53–57). Seeding Torch elsewhere would not by itself seed Python random. The “ensures reproducibility” description in P cell 63 is unsupported by its manager implementation.
2. **Exception-unsafe cleanup.** `handle.remove()` occurs after `self.model(image)` without `try/finally`; an exception leaves the injection hook attached (cell 65). Manual cleanup in cells 26/33/47 also relies on reaching and executing those cells.
3. **Observer accumulation.** Observer handles from cell 15 have no removal loop. Re-running that cell replaces the Python list but does not remove previously registered callbacks. Re-running injector registration before cleanup can likewise accumulate mutation hooks (cells 21, 31, 43).
4. **Shared global targeting.** Manager uses global `conv_layers` and `fault_hook` rather than resolving from `self.model`; passing another model instance can attach to a different model or stale module objects (cells 37, 41 and 65). Global `fault_model_instance` makes concurrent/reentrant experiments unsafe to assume (cells 65/67).
5. **Notebook state and redefinition.** `fault_hook` is redefined in cells 19, 61 and 67. A previously registered callback is not replaced merely by rebinding that name; globals it reads may still change. Current runtime hook set is UNKNOWN.
6. **Event count ambiguous.** Probability is per callback invocation. Predictor warm-up or repeated module calls can inject additional events; no schedule gates them (cells 19/65/67). Saved single logs do not establish a general one-fault-per-image guarantee.
7. **Input validation missing.** Tensor rank, nonempty dimensions, index validity, probability range, finite value and block size are not guarded (cells 19, 30, 41, 53–57).
8. **No scientific activation campaign evidence.** Saved demonstrations report class counts and logs, not paired clean/faulty COCOeval AP, raw/NMS delta tables, checkpoint/image hashes, structured scenario records or a held-out protocol (cells 23, 45, 71, 73–74). Actual numerical tensor change can be zero if the selected element already equals the replacement; no changed-value assertion exists.
9. **Preservation limits.** Clone protects the caller's tensor from the explicit assignment, but forward output is intentionally replaced. The code does not assert inference/eval mode or verify all model state, so training-mode BN-buffer changes or wrapper preparation changes cannot be ruled out (cells 19, 53–57, 65).
10. **Memory/logging overhead unmeasured.** Whole-output cloning and per-layer observers are explicit (cells 14–15, 19, 53–57). No overhead result is established.

These are static findings; no repairs or new implementation were made.

## 8. Weight vs Activation: Evidence-Based Classification

| Category | Current repository | Supplementary Drive evidence | Limits |
|---|---|---|---|
| A. Weight Fault Injection | Implemented: logical INT4/INT8 helpers and campaign scalar weight mutation | Earlier weight notebooks | R `src/int4_fault_injection.py` lines 5–39; `src/int8_fault_injection.py` lines 4–31; `src/screen05b_campaign_v1.py` lines 469–493 explicitly changes `.weight` and restores in `finally` |
| B. Activation Fault Injection | No integrated executable implementation found in audited repository/Drive code tree; documented as Planned | Actual historical replacement injector in P cells 19–23 and 51–71 | YOLOv8n demonstration; no validated current YOLOv8s bit-flip campaign |
| C. Weight Fault Protection | Implemented research code, including invariant correction; protection efficacy is a separate question | 07b sign-reference protection pilot in S | R `src/invariant07a_check_v1.py` lines 24–54 corrects INT8 weights via trusted masks; S cells 1, 4–6 protect weight sign bits |
| D. Activation Fault Protection | No executable protection mechanism found in audited scope | None found in inspected supplementary notebooks | UNKNOWN outside audited scope; changing activations to inject faults is not protection |

07b is explicitly separate: S cell 5 flips bit 7 at flat weight index 704 in `model.0.conv`, writes dequantized weights, then repairs between forwards. Its JSON `R/docs/research_reports/notebook07b_sign_protection_v1/single_image_smoke_v1.json`, `payload.fault.representation`, says `logical_INT8_weights_FP32_inference`; `payload.limitations` says manual correction and no automatic inference integration. It does not implement activation injection/protection.

## 9. Verified Findings

- The audited repository contains no actual integrated Activation Fault Injection implementation; the whole tracked text scan and Drive source scan support this bounded negative finding (section 1; explicit scope references in section 2).
- A real earlier activation replacement prototype exists outside the repository, in P, including actual forward callbacks and saved single-value execution output (cells 19/21/23, 53/65/67/71).
- It targets leaf Conv2d output tensors, not parameters or module inputs (P cells 19, 21, 30, 41, 61 and 67).
- It supports three replacement geometries in source, not activation bit flips (P cells 53, 55 and 57). Only single-value execution was found in saved demonstrations.
- It loads YOLOv8n and lacks seeded Python randomness and exception-safe cleanup; current YOLOv8s compatibility is not verified (P cells 8, 19, 53–57, 65).
- Weight injection and weight protection are independently implemented and must not be relabelled as activation injection/protection (section 8 source references).

## 10. Unknowns and Missing Evidence

- Live Colab mount, working tree, current installed package versions, loaded model and attached hooks: UNKNOWN.
- Actual current YOLOv8s module identity and fusion state during a hooked forward: UNKNOWN.
- Verified selected layer-name mapping for YOLOv8s under the pinned Ultralytics version: UNKNOWN.
- Dtype/device of historical hooked tensors: UNKNOWN; saved shapes alone do not establish these.
- Whether line/block models ran successfully in another uninspected session: UNKNOWN.
- Whether failures leak hooks in an actual run: NOT EXECUTED; source exposes the failure path.
- Bit-level activation format, injection schedule, per-image multiplicity, deterministic replay and activation-protection protocol: not implemented in the inspected prototype.
- Baseline-versus-fault activation AP, robust raw/NMS comparison, held-out generalization, overhead, or scientific efficacy: UNKNOWN.
- Untracked/ignored source files outside inspected Drive source folders, other workspaces, or uninspected notebooks: UNKNOWN; the negative finding is not a claim about all possible project history.
- Full provenance of saved notebook outputs: not independently replayed or cryptographically tied to an execution log. Saved results are historical evidence only.

## 11. Recommended Next Step

Prepare one versioned activation-injection specification for review before any implementation: define named YOLOv8s targets and fused/unfused boundary, tensor representation and fault model, deterministic event/coordinate schedule, exception-safe hook lifecycle, and paired whole-network metrics. Treat P as historical reference and 07b as a separate weight-protection pilot. Do not start an activation campaign from the existing prototype without resolving these points. This recommendation was not executed.

Only this new A04 report was authorized for creation. No project code, notebook, JSON report or experiment was executed or edited during the audit.

NO NOTEBOOK EXECUTED
NO INFERENCE EXECUTED
NO EXISTING FILES MODIFIED
NO COMMIT OR PUSH PERFORMED
