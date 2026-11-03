# INT4/INT8 Bit-Flip Unit Test Report — v1

## Purpose

Validate the software behavior of the INT4 and INT8 bit-flip fault injection helpers.

## Execution record

- Date: 2026-11-03
- Environment: Google Colab
- Device: CPU
- Baseline Git commit: `3c6530120602d881dfa02114b656eb5911a6d27d`
- Reported successful checks: **8,515**
- Reported failures: **0**
- GPU used: No
- Research results modified: No

## Source files

- `src/int4_fault_injection.py`: SHA256 `9f1aaa131ef863585f70b360e116bdb28ee36497ecc28160b81c1b170b43efc6`
- `src/int8_fault_injection.py`: SHA256 `a5f6ef9ff49c156b630742d07b254ab561d80b523d0b886018362612c5e057bc`

## Test coverage

1. Exhaustive INT4 scalar values from -8 through 7, bits 0–3.
2. Representative logical INT4 tensor bit flips.
3. Exhaustive INT8 values from -128 through 127, bits 0–7.
4. Reversibility by flipping the same bit twice.
5. Preservation of original input tensors.
6. Selected invalid INT4 input and dtype checks.

## Recorded Colab output

```text
[PASS] Exhaustive INT4 scalar validation
[PASS] Logical INT4 tensor validation
[PASS] Exhaustive INT8 validation
[PASS] Invalid INT4 input validation

Checks passed: 8515
Execution device: CPU
Git: CLEAN
Files modified: NO
Research results modified: NO
GPU used: NO
Notebook 05 started: NO
```

## Interpretation and limitations

The completed checks support correctness of the tested bit-flip operations and selected validation paths.

They do not establish full YOLOv8 fault tolerance, COCO accuracy under faults, or hardware latency, energy, and memory overhead.

Logical INT4 values are stored in int8 tensors rather than physically packed 4-bit storage.

This report documents the previously completed Colab run; the tests were not rerun during report creation.

