"""Functional checks for the frozen selected group-bit design."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path

SUMMARY_SHA = "c7fb36fadf4083b0abe816fedbc3d1d4c73fb12649490eda5341400ccfa00423"

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

def correct(observed, packed, group_size, bits):
    """Inputs: observed int8 tensor and trusted masks; no clean target values."""
    import numpy as np
    if observed.dtype != np.int8 or packed.dtype != np.uint8:
        raise ValueError("Invalid dtypes.")
    result = observed.copy()
    o = result.shape[0]
    width = result.size // o
    end = (width // group_size) * group_size
    matrix = result.view(np.uint8).reshape(o, width)
    blocks = matrix[:, :end].reshape(-1, group_size)
    if packed.shape != (len(blocks),):
        raise ValueError("Mask length mismatch.")

    zero = np.zeros(len(packed), dtype=np.uint8)
    one = np.zeros(len(packed), dtype=np.uint8)
    for offset, bit in enumerate(bits):
        zero |= ((packed >> offset) & 1) << bit
        one |= ((packed >> (offset + len(bits))) & 1) << bit
    if np.any(zero & one):
        raise ValueError("Conflicting masks.")

    fixed = (blocks & np.bitwise_not(zero[:, None])) | one[:, None]
    matrix[:, :end] = fixed.reshape(o, end)
    return result

def run(repo):
    import numpy as np
    import torch
    repo = Path(repo)
    folder = repo / "results/notebook07a/group_bit_design_comparison_v1"
    destination = folder / "nominee_correctness_v1.json"
    if destination.exists():
        raise FileExistsError("Correctness report exists; inspect before rerunning.")

    summary = unpack(verified(folder / "summary_v1.json", SUMMARY_SHA))
    protocol = json.loads(verified(
        repo / "experiments/notebook07a/group_bit_design_comparison_protocol_v1.json",
        summary["protocol_sha256"]
    ))
    nominee = summary["nominee"]
    if nominee is None:
        raise RuntimeError("No eligible nominee.")
    candidate = next(c for c in protocol["candidates"] if c["name"] == nominee)
    descriptor_path = folder / f"{nominee}_descriptor.json"
    descriptor_bytes = descriptor_path.read_bytes()
    descriptor = json.loads(descriptor_bytes)
    if descriptor["candidate"] != candidate or descriptor["protocol_sha256"] != summary["protocol_sha256"]:
        raise RuntimeError("Descriptor provenance mismatch.")

    plan = json.loads(verified(
        repo / "experiments/notebook05b/int8_screening_plan_v1.json",
        protocol["source_screening_plan_sha256"]
    ))
    screen_path = repo / "src/screen05b_campaign_v1.py"
    verified(screen_path, plan["source_sha256"])
    spec = importlib.util.spec_from_file_location("screen07a_nominee", screen_path)
    screen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(screen)

    torch.set_num_threads(1)
    prepared = screen.Campaign(repo.parent)
    prepared._load_sources()
    prepared._model("cpu")
    if prepared.clean_hash != summary["clean_model_state_sha256"]:
        raise RuntimeError("Clean model mismatch.")

    group_size, bits = candidate["group_size"], candidate["bits"]
    raw = verified(folder / descriptor["mask_archive"], descriptor["mask_archive_sha256"])
    total_weights = total_cases = recovered = 0

    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != {r["array_key"] for r in descriptor["layers"]}:
            raise RuntimeError("Mask archive schema mismatch.")
        for item in descriptor["layers"]:
            q = prepared.quantized[item["layer"]][0].cpu().contiguous().numpy()
            if list(q.shape) != item["shape"]:
                raise RuntimeError("Layer shape mismatch.")
            packed = archive[item["array_key"]].copy()
            if sha(packed.tobytes()) != item["mask_array_sha256"]:
                raise RuntimeError("Layer mask checksum mismatch.")

            o = q.shape[0]
            width = q.size // o
            end = (width // group_size) * group_size
            blocks = q.view(np.uint8).reshape(o, width)[:, :end].reshape(-1, group_size)
            expected = np.zeros(len(blocks), dtype=np.uint8)
            protection = {}

            # Independently derive invariant bits from the clean groups.
            for offset, bit in enumerate(bits):
                values = (blocks >> bit) & 1
                zeros = np.all(values == 0, axis=1)
                ones = np.all(values == 1, axis=1)
                expected |= zeros.astype(np.uint8) << offset
                expected |= ones.astype(np.uint8) << (offset + len(bits))
                protection[bit] = zeros | ones
            if not np.array_equal(packed, expected):
                raise RuntimeError("Stored masks disagree with clean invariants.")
            if not np.array_equal(correct(q, packed, group_size, bits), q):
                raise RuntimeError("Clean weights changed.")

            for bit in range(8):
                faulty = (q.view(np.uint8) ^ np.uint8(1 << bit)).view(np.int8)
                final = correct(faulty, packed, group_size, bits)
                selected = np.zeros((o, width), dtype=bool)
                if bit in protection:
                    selected[:, :end] = np.repeat(
                        protection[bit], group_size
                    ).reshape(o, end)
                selected = selected.reshape(q.shape)
                if not np.array_equal(final[selected], q[selected]):
                    raise RuntimeError("Protected bit was not restored.")
                if not np.array_equal(final[~selected], faulty[~selected]):
                    raise RuntimeError("Unprotected bit or tail changed.")
                recovered += int(selected.sum())
                total_cases += q.size
            total_weights += q.size

    expected_row = next(row for row in summary["rows"] if row["candidate"] == nominee)
    if recovered != expected_row["protected_positions"]:
        raise RuntimeError("Recovery count differs from coverage report.")
    prepared._assert_state(prepared.clean_hash)

    payload = {
        "status": "NOMINEE_FUNCTIONAL_CHECKS_PASS_NO_AP_EVALUATION",
        "candidate": candidate,
        "comparison_summary_sha256": SUMMARY_SHA,
        "descriptor_sha256": sha(descriptor_bytes),
        "mask_archive_sha256": descriptor["mask_archive_sha256"],
        "module_sha256": sha(Path(__file__).read_bytes()),
        "clean_weights_unchanged": int(total_weights),
        "position_bit_cases_checked": int(total_cases),
        "protected_cases_exactly_recovered": int(recovered),
        "unprotected_cases_left_unchanged": int(total_cases - recovered),
        "limitations": [
            "Vectorized position-wise checks, not repeated network executions.",
            "Trusted reference masks, metadata, scales and checker.",
            "No AP benefit or deployment overhead established."
        ]
    }
    content = (json.dumps({
        "payload_sha256": sha(canonical(payload)), "payload": payload
    }, indent=2, allow_nan=False) + "\n").encode()
    with destination.open("xb") as stream:
        stream.write(content)
    if destination.read_bytes() != content:
        raise RuntimeError("Write verification failed.")

    print(json.dumps(payload, indent=2))
    print("Saved:", destination)
    print("SHA256:", sha(content))
    return payload
