"""Check invariant-mask correction on CPU; no COCO image inference."""
import hashlib
import importlib.util
import io
import json
import time
from pathlib import Path

MANIFEST_SHA = "bee8dc2d17c1612ba1eddaeab8960f5d265e73c75c907c5cb80c62b0bb10f08a"

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

def grouped(values):
    o, i, kh, kw = values.shape
    if (kh, kw) == (3, 3):
        return values.reshape(-1, 9)
    if (kh, kw) == (1, 1):
        end = (i // 9) * 9
        return values.reshape(o, i)[:, :end].reshape(-1, 9)
    raise ValueError("Unsupported kernel shape.")

def correct(observed, packed):
    """Use observed weights and trusted masks only; no clean-weight input."""
    import numpy as np
    if observed.dtype != np.int8 or packed.dtype != np.uint8:
        raise ValueError("Expected int8 weights and uint8 masks.")
    result = observed.copy()
    unsigned = result.view(np.uint8)
    blocks = grouped(unsigned)
    if packed.shape != (len(blocks),):
        raise ValueError("Mask count mismatch.")
    zero = ((packed & 7) << 4).astype(np.uint8)
    one = (((packed >> 3) & 7) << 4).astype(np.uint8)
    if np.any(zero & one) or np.any(packed & 0xC0):
        raise ValueError("Invalid masks.")
    fixed = (blocks & np.bitwise_not(zero[:, None])) | one[:, None]
    o, i, kh, kw = result.shape
    if (kh, kw) == (3, 3):
        unsigned.reshape(-1, 9)[:] = fixed
    else:
        end = (i // 9) * 9
        unsigned.reshape(o, i)[:, :end] = fixed.reshape(o, end)
    return result

def run(repo):
    import numpy as np
    import torch
    repo = Path(repo)
    folder = repo / "results/notebook07a/invariant_masks_v1"
    destination = folder / "correctness_v1.json"
    if destination.exists():
        raise FileExistsError("Correctness report exists; inspect before rerunning.")

    envelope = json.loads(verified(folder / "manifest.json", MANIFEST_SHA))
    manifest = envelope["payload"]
    if sha(canonical(manifest)) != envelope["payload_sha256"]:
        raise RuntimeError("Manifest payload mismatch.")
    protocol = json.loads(verified(
        repo / "experiments/notebook07a/invariant_bit_baseline_protocol_v1.json",
        manifest["protocol_sha256"]
    ))
    plan = json.loads(verified(
        repo / "experiments/notebook05b/int8_screening_plan_v1.json",
        protocol["source_screening_plan_sha256"]
    ))
    screen_path = repo / "src/screen05b_campaign_v1.py"
    verified(screen_path, manifest["source_screen_sha256"])
    if plan["source_sha256"] != manifest["source_screen_sha256"]:
        raise RuntimeError("Source provenance mismatch.")
    spec = importlib.util.spec_from_file_location("screen07a_check", screen_path)
    screen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(screen)

    torch.set_num_threads(1)
    prepared = screen.Campaign(repo.parent)
    prepared._load_sources()
    prepared._model("cpu")
    if prepared.clean_hash != manifest["clean_model_state_sha256"]:
        raise RuntimeError("Clean model mismatch.")

    raw = verified(folder / "masks.npz", manifest["mask_archive_sha256"])
    cases = recovered = unchanged_unprotected = clean_weights = 0
    rows = []
    started = time.perf_counter()

    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        for item in manifest["inventory"]:
            q = prepared.quantized[item["layer"]][0].cpu().contiguous().numpy()
            packed = archive[item["array_key"]].copy()
            if sha(q.tobytes()) != item["clean_integer_array_sha256"]:
                raise RuntimeError("Clean integer tensor mismatch.")
            if sha(packed.tobytes()) != item["mask_array_sha256"]:
                raise RuntimeError("Layer mask checksum mismatch.")

            # Independently reconstruct invariant flags from all nine weights.
            blocks = grouped(q.view(np.uint8))
            expected = np.zeros(len(blocks), dtype=np.uint8)
            for offset, bit in enumerate((4, 5, 6)):
                values = (blocks >> bit) & 1
                expected |= np.all(values == 0, axis=1).astype(np.uint8) << offset
                expected |= np.all(values == 1, axis=1).astype(np.uint8) << (offset + 3)
            if not np.array_equal(packed, expected):
                raise RuntimeError("Masks do not match clean group invariants.")

            if not np.array_equal(correct(q, packed), q):
                raise RuntimeError("Clean weights were changed.")
            clean_weights += q.size
            layer_recovered = 0

            for bit in range(8):
                # Batch independent position-wise flips under fixed masks.
                faulty = (q.view(np.uint8) ^ np.uint8(1 << bit)).view(np.int8)
                final = correct(faulty, packed)
                protected_positions = np.zeros(q.shape, dtype=bool)

                if bit in (4, 5, 6):
                    offset = bit - 4
                    flags = (((packed & 7) | ((packed >> 3) & 7)) >> offset) & 1
                    expanded = np.repeat(flags.astype(bool), 9)
                    o, i, kh, kw = q.shape
                    if (kh, kw) == (3, 3):
                        protected_positions.reshape(-1)[:] = expanded
                    else:
                        end = (i // 9) * 9
                        protected_positions.reshape(o, i)[:, :end] = expanded.reshape(o, end)

                if not np.array_equal(final[protected_positions], q[protected_positions]):
                    raise RuntimeError("Protected flip was not restored.")
                if not np.array_equal(final[~protected_positions], faulty[~protected_positions]):
                    raise RuntimeError("Unprotected flip or tail was altered.")

                count = int(protected_positions.sum())
                cases += q.size
                recovered += count
                layer_recovered += count
                unchanged_unprotected += q.size - count

            if layer_recovered != item["protected_weight_bit_positions"]:
                raise RuntimeError("Layer recovery count mismatch.")
            rows.append({
                "layer": item["layer"],
                "position_bit_cases": int(q.size * 8),
                "exactly_recovered_protected_cases": layer_recovered
            })

    prepared._assert_state(prepared.clean_hash)
    if recovered != manifest["totals"]["protected_weight_bit_positions"]:
        raise RuntimeError("Total recovery count mismatch.")
    payload = {
        "status": "CORRECTION_FUNCTIONAL_CHECKS_PASS_NO_AP_EVALUATION",
        "manifest_sha256": MANIFEST_SHA,
        "protocol_sha256": manifest["protocol_sha256"],
        "module_sha256": sha(Path(__file__).read_bytes()),
        "clean_weights_checked_unchanged": int(clean_weights),
        "position_bit_cases_checked": int(cases),
        "protected_cases_exactly_recovered": int(recovered),
        "unprotected_cases_left_unchanged": int(unchanged_unprotected),
        "check_seconds_excluding_model_preparation": time.perf_counter() - started,
        "method": "Vectorized position-wise checks; fixed masks make corrections separable.",
        "limits": [
            "Not COCO inference or an AP benefit measurement.",
            "Trusted masks, scales, metadata and checker.",
            "This does not validate hardware fault handling or scan performance."
        ],
        "layers": rows
    }
    content = (json.dumps({
        "payload_sha256": sha(canonical(payload)), "payload": payload
    }, indent=2, allow_nan=False) + "\n").encode()
    with destination.open("xb") as stream:
        stream.write(content)
    if destination.read_bytes() != content:
        raise RuntimeError("Write verification failed.")

    for key in (
        "status", "clean_weights_checked_unchanged",
        "position_bit_cases_checked", "protected_cases_exactly_recovered",
        "unprotected_cases_left_unchanged",
        "check_seconds_excluding_model_preparation"
    ):
        print(f"{key}: {payload[key]}")
    print("Saved:", destination)
    print("SHA256:", sha(content))
    return payload
