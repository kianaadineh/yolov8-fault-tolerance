"""Calibrate the frozen 06b pilot using saved clean calibration predictions."""
import hashlib
import io
import json
from pathlib import Path

PROTOCOL_SHA = "c996c54c5d8c14a0c57267617f4967ce48b2343a7c43e03669aff48d52ba1361"

def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")
    ).encode("utf-8")

def digest(content):
    return hashlib.sha256(content).hexdigest()

def verified(path, expected):
    content = path.read_bytes()
    if digest(content) != expected:
        raise RuntimeError(f"Checksum mismatch: {path.name}")
    return content

def record(path):
    envelope = json.loads(path.read_bytes())
    if digest(canonical(envelope["payload"])) != envelope["payload_sha256"]:
        raise RuntimeError(f"Payload mismatch: {path.name}")
    return envelope["payload"]

def run(repo):
    import numpy as np
    repo = Path(repo)
    protocol_path = repo / "experiments/notebook06b/detection_repair_pilot_protocol_v1.json"
    protocol = json.loads(verified(protocol_path, PROTOCOL_SHA))
    previous = repo / "results/notebook06a/masked_bit_comparison_v1"
    session_bytes = verified(
        previous / "session.json", protocol["source_session_file_sha256"]
    )
    session_envelope = json.loads(session_bytes)
    session = session_envelope["payload"]
    if digest(canonical(session)) != session_envelope["payload_sha256"]:
        raise RuntimeError("Session payload mismatch.")
    fingerprint = digest(canonical(session))

    dataset_path = repo / "results/notebook06a/masked_bit_dataset_v1/dataset.npz"
    raw = verified(dataset_path, protocol["dataset_sha256"])
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        bits = archive["calibration_bits"].copy()
    if bits.ndim != 2 or bits.shape[1] != 8 or not np.isin(bits, [0, 1]).all():
        raise RuntimeError("Invalid calibration labels.")

    n = len(bits)
    if n == 0:
        raise RuntimeError("Empty calibration partition.")
    scores = {method: np.empty((n, 8)) for method in ("C", "D")}
    prediction_hashes = {}
    for bit in range(8):
        saved = record(previous / f"bit_{bit}_result.json")
        if saved["bit"] != bit or saved["session_fingerprint"] != fingerprint:
            raise RuntimeError("Bit provenance mismatch.")
        prediction_path = previous / f"bit_{bit}_probabilities.npz"
        raw = verified(prediction_path, saved["prediction_sha256"])
        prediction_hashes[str(bit)] = saved["prediction_sha256"]
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            for method in scores:
                p = archive[f"calibration_{method}"].copy()
                if (
                    p.shape != (n,) or not np.isfinite(p).all()
                    or np.any(p < 0) or np.any(p > 1)
                ):
                    raise RuntimeError("Invalid calibration probabilities.")
                scores[method][:, bit] = np.where(bits[:, bit] == 0, p, 1 - p)

    # Use exact integer arithmetic for the frozen one-percent budget.
    k = n // 100
    methods = {}
    for method, values in scores.items():
        maximum = values.max(axis=1)
        threshold = float(np.sort(maximum)[n - k - 1])
        count = int(np.count_nonzero(maximum > threshold))
        if count > k:
            raise RuntimeError("Calibration budget exceeded.")
        methods[method] = {
            "threshold": threshold,
            "comparison": "strictly_greater",
            "clean_targets": n,
            "allowed_interventions": k,
            "observed_interventions": count,
            "observed_fraction": count / n
        }

    payload = {
        "status": "CALIBRATION_COMPLETE_NO_TEST_EVALUATION",
        "protocol_sha256": PROTOCOL_SHA,
        "source_session_fingerprint": fingerprint,
        "dataset_sha256": protocol["dataset_sha256"],
        "prediction_sha256_by_bit": prediction_hashes,
        "module_sha256": digest(Path(__file__).read_bytes()),
        "numpy_version": np.__version__,
        "methods": methods,
        "limitation": "Empirical clean calibration budget; no test-rate guarantee."
    }
    envelope = {"payload_sha256": digest(canonical(payload)), "payload": payload}
    content = (json.dumps(envelope, indent=2, allow_nan=False) + "\n").encode()
    out = repo / "results/notebook06b/calibration_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if out.read_bytes() != content:
            raise RuntimeError("Existing calibration differs; nothing overwritten.")
    else:
        with out.open("xb") as stream:
            stream.write(content)
    if out.read_bytes() != content:
        raise RuntimeError("Write verification failed.")
    print(json.dumps(methods, indent=2))
    print("Saved:", out)
    print("SHA256:", digest(content))
    print("No test arrays or test probabilities accessed; no model fitting.")
    return payload
