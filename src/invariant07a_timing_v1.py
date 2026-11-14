"""CPU timing of complete NumPy mask scans; not inference latency."""
import hashlib
import importlib.util
import io
import json
import platform
import time
from pathlib import Path

MANIFEST_SHA = "bee8dc2d17c1612ba1eddaeab8960f5d265e73c75c907c5cb80c62b0bb10f08a"
CHECK_SHA = "b0ec6384fdf6a5123710a4806f5848cf8cc93d72bf2f35c5ea0aab2f5aa7c206"
WARMUPS = 3
REPEATS = 20

def sha(content):
    return hashlib.sha256(content).hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode()

def verified(path, expected):
    content = path.read_bytes()
    if sha(content) != expected:
        raise RuntimeError(f"Checksum mismatch: {path.name}")
    return content

def unpack(content):
    envelope = json.loads(content)
    if sha(canonical(envelope["payload"])) != envelope["payload_sha256"]:
        raise RuntimeError("Payload checksum mismatch.")
    return envelope["payload"]

def load_module(name, path, expected):
    verified(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run(repo):
    import numpy as np
    import torch
    from importlib.metadata import version
    from threadpoolctl import threadpool_limits

    repo = Path(repo)
    folder = repo / "results/notebook07a/invariant_masks_v1"
    destination = folder / "cpu_scan_timing_v1.json"
    if destination.exists():
        raise FileExistsError("Timing report exists; inspect before rerunning.")
    manifest = unpack(verified(folder / "manifest.json", MANIFEST_SHA))
    checks = unpack(verified(folder / "correctness_v1.json", CHECK_SHA))
    checker = load_module(
        "verified07a_corrector", repo / "src/invariant07a_check_v1.py",
        checks["module_sha256"]
    )
    screen = load_module(
        "verified07a_timing_screen", repo / "src/screen05b_campaign_v1.py",
        manifest["source_screen_sha256"]
    )
    torch.set_num_threads(1)
    prepared = screen.Campaign(repo.parent)
    prepared._load_sources()
    prepared._model("cpu")
    if prepared.clean_hash != manifest["clean_model_state_sha256"]:
        raise RuntimeError("Clean state mismatch.")

    raw = verified(folder / "masks.npz", manifest["mask_archive_sha256"])
    clean, masks = [], []
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        for item in manifest["inventory"]:
            q = prepared.quantized[item["layer"]][0].cpu().contiguous().numpy().copy()
            packed = archive[item["array_key"]].copy()
            if sha(q.tobytes()) != item["clean_integer_array_sha256"]:
                raise RuntimeError("Clean tensor mismatch.")
            if sha(packed.tobytes()) != item["mask_array_sha256"]:
                raise RuntimeError("Mask mismatch.")
            clean.append(q)
            masks.append(packed)

    faulty = list(clean)
    selected_fault = None
    for layer_index, (q, packed) in enumerate(zip(clean, masks)):
        protected = (packed & 7) | ((packed >> 3) & 7)
        available = np.flatnonzero(protected)
        if not len(available):
            continue
        group = int(available[0])
        offset = next(b for b in range(3) if int(protected[group]) & (1 << b))
        bit = offset + 4
        o, i, kh, kw = q.shape
        if (kh, kw) == (3, 3):
            flat = group * 9
        else:
            per_channel = i // 9
            flat = (group // per_channel) * i + (group % per_channel) * 9
        faulty[layer_index] = q.copy()
        faulty[layer_index].view(np.uint8).reshape(-1)[flat] ^= np.uint8(1 << bit)
        selected_fault = {
            "layer": manifest["inventory"][layer_index]["layer"],
            "flat_index": flat, "bit": bit,
            "selection": "first_protected_position_to_exercise_correction_only"
        }
        break
    if selected_fault is None:
        raise RuntimeError("No protected fault available.")

    def scan(arrays):
        return [checker.correct(q, mask) for q, mask in zip(arrays, masks)]

    results = {}
    with threadpool_limits(limits=1):
        for label, arrays in (("clean_scan", clean), ("one_protected_fault_scan", faulty)):
            for _ in range(WARMUPS):
                output = scan(arrays)
                del output
            times = []
            for _ in range(REPEATS):
                started = time.perf_counter()
                output = scan(arrays)
                times.append(time.perf_counter() - started)
                # Validation and output deletion are outside the timed interval.
                if not all(np.array_equal(a, b) for a, b in zip(output, clean)):
                    raise RuntimeError("Scan output differs from clean reference.")
                del output
            results[label] = {
                "seconds": times,
                "median_seconds": float(np.median(times)),
                "min_seconds": float(np.min(times)),
                "empirical_p95_seconds": float(np.quantile(times, 0.95))
            }

    prepared._assert_state(prepared.clean_hash)
    payload = {
        "status": "CPU_NUMPY_SCAN_TIMING_COMPLETE_NOT_INFERENCE_OVERHEAD",
        "manifest_sha256": MANIFEST_SHA,
        "correctness_file_sha256": CHECK_SHA,
        "module_sha256": sha(Path(__file__).read_bytes()),
        "warmups": WARMUPS, "repeats": REPEATS,
        "order": ["clean_scan", "one_protected_fault_scan"],
        "weights_scanned_per_repeat": sum(q.size for q in clean),
        "input_array_bytes": sum(q.nbytes for q in clean),
        "mask_array_bytes": sum(m.nbytes for m in masks),
        "output_array_bytes_per_scan": sum(q.nbytes for q in clean),
        "selected_fault": selected_fault,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": version("numpy"), "torch": version("torch"),
            "threadpool_limit": 1
        },
        "results": results,
        "limitations": [
            "In-memory CPU NumPy implementation; outputs allocate new arrays.",
            "Model preparation, file reads, validation and output deletion excluded.",
            "Temporary allocation peaks and live RAM not measured.",
            "Warm-cache repeated measurements; not deployment latency guarantees.",
            "Selected protected fault is a functional timing workload, not an efficacy sample.",
            "No measured impact on complete YOLO inference latency."
        ]
    }
    content = (json.dumps({
        "payload_sha256": sha(canonical(payload)), "payload": payload
    }, indent=2, allow_nan=False) + "\n").encode()
    with destination.open("xb") as stream:
        stream.write(content)
    if destination.read_bytes() != content:
        raise RuntimeError("Write verification failed.")

    for label, result in results.items():
        print(
            f"{label}: median={result['median_seconds']:.6f}s | "
            f"empirical p95={result['empirical_p95_seconds']:.6f}s"
        )
    print("Weights per scan:", payload["weights_scanned_per_repeat"])
    print("Selected timing fault:", selected_fault)
    print("Saved:", destination)
    print("SHA256:", sha(content))
    return payload
