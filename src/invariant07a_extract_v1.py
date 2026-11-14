"""Extract frozen invariant-bit masks on CPU; no COCO image inference."""
import hashlib
import importlib.util
import io
import json
import os
import platform
import time
from pathlib import Path

PROTOCOL_SHA = "74785069999784731bcc2917a5fd44fa8b698c102f76a9043b1f022b8303a799"

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

def publish(path, content):
    partial = path.with_name(path.name + ".part")
    if path.exists() or partial.exists():
        raise FileExistsError(f"Existing output requires review: {path.name}")
    with partial.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    if partial.read_bytes() != content:
        raise RuntimeError("Write verification failed.")
    partial.rename(path)

def run(repo):
    import numpy as np
    import torch
    from importlib.metadata import version

    repo = Path(repo)
    out = repo / "results/notebook07a/invariant_masks_v1"
    if out.exists():
        raise FileExistsError("Output directory exists; inspect before rerunning.")

    protocol = json.loads(verified(
        repo / "experiments/notebook07a/invariant_bit_baseline_protocol_v1.json",
        PROTOCOL_SHA
    ))
    plan = json.loads(verified(
        repo / "experiments/notebook05b/int8_screening_plan_v1.json",
        protocol["source_screening_plan_sha256"]
    ))
    screen_path = repo / "src/screen05b_campaign_v1.py"
    verified(screen_path, plan["source_sha256"])
    spec = importlib.util.spec_from_file_location("screen07a_verified", screen_path)
    screen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(screen)

    torch.set_num_threads(1)
    prepared = screen.Campaign(repo.parent)
    prepared._load_sources()
    prepared._model("cpu")
    if prepared.clean_hash != protocol["clean_model_state_sha256"]:
        raise RuntimeError("Shared clean model mismatch.")

    started = time.perf_counter()
    masks, inventory = {}, []
    popcount = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)

    for index, item in enumerate(plan["inventory"]):
        name = item["name"]
        q = prepared.quantized[name][0].cpu().contiguous().numpy()
        if list(q.shape) != item["weight_shape"] or str(q.dtype) != "int8":
            raise RuntimeError("Quantized tensor shape or dtype mismatch.")
        o, i, kh, kw = q.shape
        unsigned = q.view(np.uint8)

        if (kh, kw) == (3, 3):
            groups = unsigned.reshape(-1, 9)
            tail = 0
        elif (kh, kw) == (1, 1):
            complete_inputs = (i // 9) * 9
            groups = unsigned.reshape(o, i)[:, :complete_inputs].reshape(-1, 9)
            tail = o * (i - complete_inputs)
        else:
            raise RuntimeError("Unsupported kernel shape.")

        all_one = np.bitwise_and.reduce(groups, axis=1) & 0x70
        all_zero = np.bitwise_not(
            np.bitwise_or.reduce(groups, axis=1)
        ) & 0x70
        if np.any(all_one & all_zero):
            raise RuntimeError("Conflicting masks.")

        packed = (
            (all_zero >> 4) | ((all_one >> 4) << 3)
        ).astype(np.uint8)
        if np.any(packed & 0xC0):
            raise RuntimeError("Reserved mask bits are not zero.")

        key = f"layer_{index:03d}"
        masks[key] = packed
        protected = int(popcount[packed].sum()) * 9
        count = int(q.size)
        if len(groups) * 9 + tail != count:
            raise RuntimeError("Grouping count mismatch.")

        inventory.append({
            "layer_id": index,
            "layer": name,
            "shape": list(q.shape),
            "array_key": key,
            "clean_integer_array_sha256": sha(q.tobytes()),
            "mask_array_sha256": sha(packed.tobytes()),
            "weights": count,
            "groups": len(groups),
            "tail_weights": tail,
            "mask_array_bytes": packed.nbytes,
            "protected_weight_bit_positions": protected,
            "candidate_bit_coverage": protected / (3 * count),
            "all_bit_coverage": protected / (8 * count),
            "protected_positions_by_bit": {
                str(bit): int(np.count_nonzero(
                    (all_zero | all_one) & (1 << bit)
                )) * 9
                for bit in (4, 5, 6)
            }
        })

    prepared._assert_state(prepared.clean_hash)
    extraction_seconds = time.perf_counter() - started
    if len(inventory) != 63:
        raise RuntimeError("Expected 63 layers.")

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **masks)
    archive_bytes = buffer.getvalue()
    with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as archive:
        if set(archive.files) != set(masks):
            raise RuntimeError("Mask archive keys mismatch.")
        for key, array in masks.items():
            if not np.array_equal(array, archive[key]):
                raise RuntimeError("Mask archive round-trip mismatch.")

    total = sum(row["weights"] for row in inventory)
    groups = sum(row["groups"] for row in inventory)
    protected = sum(row["protected_weight_bit_positions"] for row in inventory)
    payload = {
        "status": "MASKS_EXTRACTED_CORRECTION_TESTS_PENDING",
        "protocol_sha256": PROTOCOL_SHA,
        "module_sha256": sha(Path(__file__).read_bytes()),
        "source_screen_sha256": plan["source_sha256"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "clean_model_state_sha256": prepared.clean_hash,
        "mask_archive_sha256": sha(archive_bytes),
        "environment": {
            "python": platform.python_version(),
            "packages": {p: version(p) for p in ("torch", "numpy", "ultralytics")},
            "device": "cpu",
            "torch_threads": torch.get_num_threads()
        },
        "totals": {
            "weights": total,
            "groups": groups,
            "tail_weights": sum(row["tail_weights"] for row in inventory),
            "mask_array_bytes": sum(array.nbytes for array in masks.values()),
            "ideal_six_bit_storage_bytes": (6 * groups + 7) // 8,
            "compressed_mask_archive_bytes": len(archive_bytes),
            "protected_weight_bit_positions": protected,
            "candidate_bit_coverage": protected / (3 * total),
            "all_bit_coverage": protected / (8 * total),
            "extraction_seconds_excluding_model_preparation": extraction_seconds
        },
        "inventory": inventory,
        "limitations": [
            "Trusted masks, metadata and scales.",
            "Actual protection coverage differs from grouped weight fraction.",
            "Compressed file size is not runtime mask memory.",
            "Correction correctness and AP effectiveness are not yet tested."
        ]
    }
    manifest_bytes = (json.dumps({
        "payload_sha256": sha(canonical(payload)), "payload": payload
    }, indent=2, allow_nan=False) + "\n").encode()

    out.mkdir(parents=True, exist_ok=False)
    publish(out / "masks.npz", archive_bytes)
    publish(out / "manifest.json", manifest_bytes)

    print(json.dumps(payload["totals"], indent=2))
    print("Serialized manifest bytes:", len(manifest_bytes))
    print("Total archive + manifest bytes:", len(archive_bytes) + len(manifest_bytes))
    print("Saved:", out)
    print("Manifest SHA256:", sha(manifest_bytes))
    print("No COCO images, correction experiment, or 05b execution.")
    return payload
