# One-scenario engineering validation

This module validates a complete paired path on the frozen 200 development
images. It is not the 5,040-scenario campaign and cannot establish sensitivity,
ranking, rare-event discovery or a representative runtime distribution.

- Reconstruct the exact shared logical INT8 clean baseline from 05a on CPU.
- Compare the full state hash before and after GPU transfer and warmup.
- Fixed convenience scenario: model.0.conv, weight [0,0,0,0], sign bit 7.
- Decode cached JPEG bytes, LetterBox 640 square, RGB CHW, FP32 /255.
- GPU FP32 forward, CPU NMS at confidence .001, IoU .7, max_det 300.
- Convert coordinates back to original image geometry and COCO categories.
- Verify exactly one weight changed and the other state remains clean.
- Keep the same injected weight across all images; verify fault state afterward.
- Restore exact clean weights in finally, including on a fault-path exception.
- Save paired predictions, evaluate original annotations with COCOeval, maxDets
  [1,10,100], retain all 200 image IDs including images with no valid boxes.
- Save metrics, source and image hashes, package versions and timing components.
- Use the verified 05a atomic exclusive storage helper. Existing output directory
  or partial records block rerun; do not delete evidence to bypass a failure.

The synthetic tests cover exhaustive INT8 bit arithmetic, subset separation and
source checksums only. CUDA, restoration, quantization reconstruction, COCOeval
and actual runtime are checked during Colab execution, not established locally.
The execution aborts on invalid outputs; a failed scenario is not silently
removed or assigned zero damage. The full campaign failure policy remains open.

The clean and faulted model share a 200-image subset. Their AP values must not be
directly compared with AP on 50 or 1,000 different images. This module does not
claim native INT8 execution, a hardware fault rate or fault-tolerant protection.

Usage (only after installation and verified imports):

    from fault05b_validation_v1 import run
    result = run('/content/drive/MyDrive/YOLOv8_Fault_Tolerance')

No model file is modified; new artifacts live under
results/notebook05b/engineering_fault_validation_v1/ and remain excluded by the
existing Git ignore rules. Publish a small reviewed report later.
