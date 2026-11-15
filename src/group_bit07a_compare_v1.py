"""Frozen CPU comparison of four invariant-mask designs; no COCO inference."""
import csv
import hashlib
import importlib.util
import io
import json
import os
import platform
import time
from pathlib import Path

PROTOCOL_SHA = "9ca97fde22132dbb403c586f69d6cc24906f223cac6113e74108e46157325c43"
REFERENCE_MANIFEST_SHA = "bee8dc2d17c1612ba1eddaeab8960f5d265e73c75c907c5cb80c62b0bb10f08a"

def sha(content):
    return hashlib.sha256(content).hexdigest()

def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")
    ).encode("utf-8")

def verified(path, expected):
    content = path.read_bytes()
    if sha(content) != expected:
        raise RuntimeError(f"Checksum mismatch: {path.name}")
    return content

def unpack(content):
    envelope = json.loads(content)
    if sha(canonical(envelope["payload"])) != envelope["payload_sha256"]:
        raise RuntimeError("Record payload checksum mismatch.")
    return envelope["payload"]

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
    out = repo / "results/notebook07a/group_bit_design_comparison_v1"
    if out.exists():
        raise FileExistsError("Comparison directory exists; inspect before rerunning.")

    protocol = json.loads(verified(
        repo / "experiments/notebook07a/group_bit_design_comparison_protocol_v1.json",
        PROTOCOL_SHA
    ))
    plan = json.loads(verified(
        repo / "experiments/notebook05b/int8_screening_plan_v1.json",
        protocol["source_screening_plan_sha256"]
    ))
    reference_folder = repo / "results/notebook07a/invariant_masks_v1"
    reference = unpack(verified(
        reference_folder / "manifest.json", REFERENCE_MANIFEST_SHA
    ))
    reference_bytes = verified(
        reference_folder / "masks.npz", reference["mask_archive_sha256"]
    )
    with np.load(io.BytesIO(reference_bytes), allow_pickle=False) as archive:
        reference_masks = {key: archive[key].copy() for key in archive.files}

    source_path = repo / "src/screen05b_campaign_v1.py"
    verified(source_path, plan["source_sha256"])
    spec = importlib.util.spec_from_file_location("screen07a_design", source_path)
    screen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(screen)

    torch.set_num_threads(1)
    prepared = screen.Campaign(repo.parent)
    prepared._load_sources()
    prepared._model("cpu")
    if (
        prepared.clean_hash != protocol["clean_model_state_sha256"]
        or prepared.clean_hash != reference["clean_model_state_sha256"]
    ):
        raise RuntimeError("Clean model identity mismatch.")

    reference_by_name = {row["layer"]: row for row in reference["inventory"]}
    weights = {}
    for item in plan["inventory"]:
        name = item["name"]
        q = prepared.quantized[name][0].cpu().contiguous().numpy().copy()
        previous = reference_by_name[name]
        if (
            list(q.shape) != item["weight_shape"]
            or q.dtype != np.int8
            or sha(q.tobytes()) != previous["clean_integer_array_sha256"]
        ):
            raise RuntimeError("Clean integer tensor mismatch.")
        weights[name] = q
    total_weights = sum(q.size for q in weights.values())
    popcount = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)

    rows, layer_rows, outputs = [], [], []
    for candidate in protocol["candidates"]:
        started = time.perf_counter()
        group_size = candidate["group_size"]
        bits = candidate["bits"]
        nbits = len(bits)
        arrays, descriptors = {}, []
        protected_total = groups_total = tails_total = 0
        by_bit = {str(bit): 0 for bit in range(8)}

        for layer_id, item in enumerate(plan["inventory"]):
            name = item["name"]
            q = weights[name]
            o = q.shape[0]
            width = q.size // o
            complete = (width // group_size) * group_size
            blocks = q.view(np.uint8).reshape(o, width)[:, :complete]
            blocks = blocks.reshape(-1, group_size)

            all_one = np.bitwise_and.reduce(blocks, axis=1)
            all_zero = np.bitwise_not(np.bitwise_or.reduce(blocks, axis=1))
            packed = np.zeros(len(blocks), dtype=np.uint8)
            counts = {str(bit): 0 for bit in range(8)}

            for offset, bit in enumerate(bits):
                zero = ((all_zero >> bit) & 1).astype(np.uint8)
                one = ((all_one >> bit) & 1).astype(np.uint8)
                if np.any(zero & one):
                    raise RuntimeError("Conflicting invariant flags.")
                packed |= zero << offset
                packed |= one << (offset + nbits)
                counts[str(bit)] = int(np.count_nonzero(zero | one)) * group_size
                by_bit[str(bit)] += counts[str(bit)]

            key = f"layer_{layer_id:03d}"
            if candidate["name"] == "reference_g9_b456":
                old_key = reference_by_name[name]["array_key"]
                if not np.array_equal(packed, reference_masks[old_key]):
                    raise RuntimeError(f"Reference mask reproduction failed: {name}")

            protected = int(popcount[packed].sum()) * group_size
            if protected != sum(counts.values()):
                raise RuntimeError("Invariant coverage accounting mismatch.")
            tails = o * (width - complete)
            if len(blocks) * group_size + tails != q.size:
                raise RuntimeError("Grouping accounting mismatch.")

            arrays[key] = packed
            protected_total += protected
            groups_total += len(blocks)
            tails_total += tails
            descriptors.append({
                "layer": name,
                "shape": list(q.shape),
                "array_key": key,
                "groups": len(blocks),
                "mask_array_sha256": sha(packed.tobytes())
            })
            layer_rows.append({
                "candidate": candidate["name"],
                "layer": name,
                "weights": int(q.size),
                "groups": len(blocks),
                "tail_weights": tails,
                "protected_positions": protected,
                **{f"protected_bit_{bit}": counts[str(bit)] for bit in range(8)}
            })

        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        archive_bytes = buffer.getvalue()
        with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise RuntimeError("Archive schema mismatch.")
            for key in arrays:
                if not np.array_equal(arrays[key], archive[key]):
                    raise RuntimeError("Archive round-trip mismatch.")

        descriptor = {
            "version": "1.0",
            "protocol_sha256": PROTOCOL_SHA,
            "candidate": candidate,
            "representation": "logical_signed_INT8",
            "grouping": protocol["grouping"],
            "encoding": "low_n_bits_force_zero_next_n_bits_force_one",
            "mask_dtype": "uint8",
            "mask_archive": candidate["name"] + "_masks.npz",
            "mask_archive_sha256": sha(archive_bytes),
            "reference_metadata_trusted": True,
            "layers": descriptors
        }
        descriptor_bytes = canonical(descriptor) + b"\n"
        mask_bytes = sum(a.nbytes for a in arrays.values())
        cost = mask_bytes + len(descriptor_bytes)
        overhead = cost / total_weights
        row = {
            "candidate": candidate["name"],
            "group_size": group_size,
            "candidate_bits": ",".join(map(str, bits)),
            "weights": total_weights,
            "groups": groups_total,
            "tail_weights": tails_total,
            "protected_positions": protected_total,
            "all_bit_coverage_pct": 100 * protected_total / (8 * total_weights),
            "raw_mask_bytes": mask_bytes,
            "descriptor_bytes": len(descriptor_bytes),
            "budget_bytes": cost,
            "storage_overhead_pct": 100 * overhead,
            "protected_positions_per_budget_byte": protected_total / cost,
            "within_budget": overhead <= protocol["selection"]["engineering_storage_budget_fraction"],
            "compressed_archive_bytes": len(archive_bytes),
            "construction_and_serialization_seconds": time.perf_counter() - started,
            **{f"protected_bit_{bit}": by_bit[str(bit)] for bit in range(8)}
        }
        rows.append(row)
        outputs.extend([
            (candidate["name"] + "_masks.npz", archive_bytes),
            (candidate["name"] + "_descriptor.json", descriptor_bytes)
        ])

    eligible = [row for row in rows if row["within_budget"]]
    eligible.sort(key=lambda row: (
        -row["protected_positions_per_budget_byte"],
        row["budget_bytes"], row["group_size"], row["candidate"]
    ))
    nominee = eligible[0]["candidate"] if eligible else None
    prepared._assert_state(prepared.clean_hash)

    payload = {
        "status": "CPU_COVERAGE_COST_SCREEN_COMPLETE_NOT_AP_EVALUATION",
        "protocol_sha256": PROTOCOL_SHA,
        "module_sha256": sha(Path(__file__).read_bytes()),
        "reference_manifest_sha256": REFERENCE_MANIFEST_SHA,
        "clean_model_state_sha256": prepared.clean_hash,
        "reference_masks_reproduced_exactly": True,
        "environment": {
            "python": platform.python_version(),
            "packages": {p: version(p) for p in ("numpy", "torch", "ultralytics")},
            "device": "cpu", "torch_threads": torch.get_num_threads()
        },
        "budget_fraction": protocol["selection"]["engineering_storage_budget_fraction"],
        "rows": rows,
        "nominee": nominee,
        "cost_definition": (
            "Raw mask arrays plus exact serialized runtime descriptor bytes. "
            "Compressed archives and audit-only result tables are reported separately."
        ),
        "limitations": protocol["limitations"] + [
            "New candidate correction functions and scan timings are not yet validated.",
            "Selection maximizes uniform protected-position efficiency, not AP benefit.",
            "No sensitivity-informed allocation has been implemented."
        ]
    }
    summary_bytes = (json.dumps({
        "payload_sha256": sha(canonical(payload)), "payload": payload
    }, indent=2, allow_nan=False) + "\n").encode()

    for filename, records in (
        ("comparison_v1.csv", rows), ("per_layer_v1.csv", layer_rows)
    ):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
        outputs.append((filename, stream.getvalue().encode()))
    outputs.append(("summary_v1.json", summary_bytes))

    out.mkdir(parents=True, exist_ok=False)
    for filename, content in outputs:
        publish(out / filename, content)

    print("Reference masks reproduced exactly: PASS")
    print("Candidate | all-bit coverage % | storage overhead % | protected/byte | within budget")
    for row in rows:
        print(
            f"{row['candidate']} | {row['all_bit_coverage_pct']:.6f} | "
            f"{row['storage_overhead_pct']:.6f} | "
            f"{row['protected_positions_per_budget_byte']:.6f} | "
            f"{row['within_budget']}"
        )
    print("Coverage/cost nominee:", nominee)
    print("Saved:", out)
    print("Summary SHA256:", sha(summary_bytes))
    print("No COCO inference, AP selection, or changes to 05b.")
    return payload
