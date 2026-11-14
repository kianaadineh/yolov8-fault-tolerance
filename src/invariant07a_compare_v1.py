"""Retrospective comparison on existing 06b test targets; no inference."""
import hashlib
import io
import json
from pathlib import Path

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

def record(content):
    envelope = json.loads(content)
    if sha(canonical(envelope["payload"])) != envelope["payload_sha256"]:
        raise RuntimeError("Record checksum mismatch.")
    return envelope["payload"]

def run(repo):
    import numpy as np
    repo = Path(repo)
    folder = repo / "results/notebook07a/invariant_masks_v1"
    manifest = record(verified(
        folder / "manifest.json",
        "bee8dc2d17c1612ba1eddaeab8960f5d265e73c75c907c5cb80c62b0bb10f08a"
    ))
    correctness = record(verified(
        folder / "correctness_v1.json",
        "b0ec6384fdf6a5123710a4806f5848cf8cc93d72bf2f35c5ea0aab2f5aa7c206"
    ))
    if correctness["manifest_sha256"] != (
        "bee8dc2d17c1612ba1eddaeab8960f5d265e73c75c907c5cb80c62b0bb10f08a"
    ):
        raise RuntimeError("Correctness provenance mismatch.")

    source06a = repo / "results/notebook06a"
    session = record(
        (source06a / "masked_bit_comparison_v1/session.json").read_bytes()
    )
    dataset_bytes = verified(
        source06a / "masked_bit_dataset_v1/dataset.npz",
        session["dataset_sha256"]
    )
    with np.load(io.BytesIO(dataset_bytes), allow_pickle=False) as archive:
        test = {key: archive[f"test_{key}"].copy()
                for key in ("bits", "layer", "channel", "flat_index")}

    folder06b = repo / "results/notebook06b/evaluation_v1"
    summary = record((folder06b / "summary.json").read_bytes())
    if sha(canonical(session)) != summary["source_session_fingerprint"]:
        raise RuntimeError("06a/06b session mismatch.")
    outcome_bytes = verified(
        folder06b / "outcomes.npz", summary["outcomes_sha256"]
    )
    with np.load(io.BytesIO(outcome_bytes), allow_pickle=False) as archive:
        outcomes = {key: archive[key].copy() for key in archive.files}
    for key in ("layer", "channel", "flat_index"):
        if not np.array_equal(test[key], outcomes[key]):
            raise RuntimeError("Test target alignment mismatch.")

    n = len(test["bits"])
    recovered = np.zeros((n, 8), dtype=bool)
    mask_bytes = verified(folder / "masks.npz", manifest["mask_archive_sha256"])

    with np.load(io.BytesIO(mask_bytes), allow_pickle=False) as archive:
        for item in manifest["inventory"]:
            selected = np.flatnonzero(test["layer"] == item["layer_id"])
            o, i, kh, kw = item["shape"]
            flat = test["flat_index"][selected]
            if np.any(flat < 0) or np.any(flat >= o * i * kh * kw):
                raise RuntimeError("Invalid target index.")
            channel = flat // (i * kh * kw)
            if not np.array_equal(channel, test["channel"][selected]):
                raise RuntimeError("Target channel mismatch.")

            if (kh, kw) == (3, 3):
                valid = np.ones(len(selected), dtype=bool)
                group_index = flat // 9
            elif (kh, kw) == (1, 1):
                within = flat % i
                valid = within < (i // 9) * 9
                group_index = channel * (i // 9) + within // 9
            else:
                raise RuntimeError("Unsupported kernel.")

            packed = archive[item["array_key"]]
            if sha(packed.tobytes()) != item["mask_array_sha256"]:
                raise RuntimeError("Layer mask mismatch.")
            rows = selected[valid]
            codes = packed[group_index[valid]]
            zero, one = codes & 7, (codes >> 3) & 7

            for offset, bit in enumerate((4, 5, 6)):
                force_zero = ((zero >> offset) & 1).astype(bool)
                force_one = ((one >> offset) & 1).astype(bool)
                clean_bits = test["bits"][rows, bit]
                if np.any(clean_bits[force_zero] != 0) or np.any(clean_bits[force_one] != 1):
                    raise RuntimeError("Mask disagrees with clean sampled targets.")
                recovered[rows, bit] = force_zero | force_one

    methods = {
        "invariant_group_baseline": {
            "exact_recovery_fraction": float(recovered.mean()),
            "exact_recovery_by_bit": recovered.mean(axis=0).tolist(),
            "clean_false_intervention_fraction": 0.0,
            "clean_control_basis": "Full clean-weight functional check and invariant agreement.",
        }
    }
    for method in ("C", "D"):
        values = outcomes[f"{method}_exact_recovery"]
        methods[method] = {
            "exact_recovery_fraction": float(values.mean()),
            "exact_recovery_by_bit": values.mean(axis=0).tolist(),
            "clean_false_intervention_fraction": float(
                outcomes[f"{method}_clean_intervention"].mean()
            )
        }

    payload = {
        "status": "RETROSPECTIVE_COMMON_TARGET_COMPARISON_NOT_EQUAL_BUDGET",
        "targets": n,
        "single_bit_cases_per_method": n * 8,
        "dataset_sha256": session["dataset_sha256"],
        "outcomes_sha256": summary["outcomes_sha256"],
        "mask_archive_sha256": manifest["mask_archive_sha256"],
        "module_sha256": sha(Path(__file__).read_bytes()),
        "methods": methods,
        "limitations": [
            "Existing inspected 06b test targets; exploratory comparison.",
            "Recovery of the invariant baseline derived from verified bit coverage.",
            "Predictor and mask memory budgets are not matched.",
            "Uniformity over sampled targets and bits is not whole-network uniform sampling.",
            "No new COCO inference or AP benefit established."
        ]
    }
    content = (json.dumps({
        "payload_sha256": sha(canonical(payload)), "payload": payload
    }, indent=2, allow_nan=False) + "\n").encode()
    destination = folder / "comparison_on_06b_targets_v1.json"
    if destination.exists():
        if destination.read_bytes() != content:
            raise RuntimeError("Existing comparison differs; nothing overwritten.")
    else:
        with destination.open("xb") as stream:
            stream.write(content)
    if destination.read_bytes() != content:
        raise RuntimeError("Write verification failed.")

    print("Common test targets:", n)
    print("Single-bit cases per method:", n * 8)
    for method, result in methods.items():
        print(
            f"{method}: exact recovery={100 * result['exact_recovery_fraction']:.6f}% | "
            f"clean false intervention={100 * result['clean_false_intervention_fraction']:.6f}%"
        )
        print("Recovery by bit, %:",
              [round(100 * x, 6) for x in result["exact_recovery_by_bit"]])
    print("Saved:", destination)
    print("SHA256:", sha(content))
    return payload
