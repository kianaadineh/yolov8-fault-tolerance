"""Frozen exploratory weight-level evaluation; no fitting or threshold tuning."""
import hashlib
import importlib.util
import io
import json
import platform
import time
from pathlib import Path

CALIBRATION_SHA = "87bfe4072cc4f8b1eb90fe961d4c348552f674223d6d49a85f5e7e0c7e1931e0"
PROTOCOL_SHA = "c996c54c5d8c14a0c57267617f4967ce48b2343a7c43e03669aff48d52ba1361"

def sha(content):
    return hashlib.sha256(content).hexdigest()

def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")
    ).encode()

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

def run(repo):
    import joblib
    import numpy as np
    from importlib.metadata import version
    from threadpoolctl import threadpool_limits

    repo = Path(repo)
    out = repo / "results/notebook06b/evaluation_v1"
    if out.exists():
        raise FileExistsError("Evaluation directory exists; inspect before any rerun.")

    protocol = json.loads(verified(
        repo / "experiments/notebook06b/detection_repair_pilot_protocol_v1.json",
        PROTOCOL_SHA
    ))
    calibration = unpack(verified(
        repo / "results/notebook06b/calibration_v1.json", CALIBRATION_SHA
    ))
    if calibration["protocol_sha256"] != PROTOCOL_SHA:
        raise RuntimeError("Calibration protocol mismatch.")

    previous = repo / "results/notebook06a/masked_bit_comparison_v1"
    session = unpack(verified(
        previous / "session.json", protocol["source_session_file_sha256"]
    ))
    fingerprint = sha(canonical(session))
    if calibration["source_session_fingerprint"] != fingerprint:
        raise RuntimeError("Calibration session mismatch.")
    for package, expected in session["environment"]["packages"].items():
        if version(package) != expected:
            raise RuntimeError(f"Package version mismatch: {package}")

    runner_path = repo / "src/contextual06a_runner_v1.py"
    verified(runner_path, session["source_sha256"])
    spec = importlib.util.spec_from_file_location("trusted06a_eval", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    raw = verified(
        repo / "results/notebook06a/masked_bit_dataset_v1/dataset.npz",
        protocol["dataset_sha256"]
    )
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        test = {
            key: archive[f"test_{key}"].copy()
            for key in ("bits", "context", "layer", "channel", "flat_index")
        }
    n = len(test["bits"])
    if n == 0 or test["bits"].shape != (n, 8):
        raise RuntimeError("Invalid test shape.")
    if not np.isin(test["bits"], [0, 1]).all():
        raise RuntimeError("Invalid test bits.")

    methods = ("C", "D")
    models = {}
    clean_scores = {m: np.empty((n, 8)) for m in methods}
    thresholds = {m: calibration["methods"][m]["threshold"] for m in methods}
    started = time.perf_counter()

    with threadpool_limits(limits=1):
        for bit in range(8):
            record = unpack((previous / f"bit_{bit}_result.json").read_bytes())
            if record["bit"] != bit or record["session_fingerprint"] != fingerprint:
                raise RuntimeError("Bit provenance mismatch.")
            if record["prediction_sha256"] != calibration["prediction_sha256_by_bit"][str(bit)]:
                raise RuntimeError("Prediction provenance mismatch.")
            model_bytes = verified(
                previous / f"bit_{bit}_models.joblib", record["models_sha256"]
            )
            models[bit] = joblib.load(io.BytesIO(model_bytes))
            raw_predictions = verified(
                previous / f"bit_{bit}_probabilities.npz",
                record["prediction_sha256"]
            )
            with np.load(io.BytesIO(raw_predictions), allow_pickle=False) as archive:
                for m in methods:
                    p = archive[f"test_{m}"].copy()
                    if p.shape != (n,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                        raise RuntimeError("Invalid saved probabilities.")
                    clean_scores[m][:, bit] = np.where(test["bits"][:, bit] == 0, p, 1-p)

        arrays = {
            key: test[key] for key in ("layer", "channel", "flat_index")
        }
        summary = {}
        for m in methods:
            arrays[f"{m}_clean_intervention"] = clean_scores[m].max(axis=1) > thresholds[m]
            arrays[f"{m}_fault_candidate"] = np.empty((n, 8), dtype=np.uint8)
            arrays[f"{m}_fault_intervention"] = np.empty((n, 8), dtype=bool)

        for injected_bit in range(8):
            observed = dict(test)
            observed["bits"] = test["bits"].copy()
            observed["bits"][:, injected_bit] ^= 1
            scores = {m: np.empty((n, 8)) for m in methods}

            for candidate_bit in range(8):
                for m in methods:
                    estimator = models[candidate_bit][m]
                    features = runner.features(
                        observed, candidate_bit, with_context=(m == "C")
                    )
                    if len(estimator.classes_) == 1:
                        p = np.full(n, float(estimator.classes_[0]))
                    else:
                        probabilities = estimator.predict_proba(features)
                        ones = np.flatnonzero(estimator.classes_ == 1)
                        p = probabilities[:, int(ones[0])] if len(ones) else np.zeros(n)
                    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                        raise RuntimeError("Invalid fault-case probabilities.")
                    scores[m][:, candidate_bit] = np.where(
                        observed["bits"][:, candidate_bit] == 0, p, 1-p
                    )

            for m in methods:
                # NumPy argmax selects the first index when scores tie.
                arrays[f"{m}_fault_candidate"][:, injected_bit] = scores[m].argmax(axis=1)
                arrays[f"{m}_fault_intervention"][:, injected_bit] = (
                    scores[m].max(axis=1) > thresholds[m]
                )
            print(f"Completed injected bit {injected_bit}/7", flush=True)

        for m in methods:
            acted = arrays[f"{m}_fault_intervention"]
            correct = arrays[f"{m}_fault_candidate"] == np.arange(8)[None, :]
            recovered = acted & correct
            wrong = acted & ~correct
            arrays[f"{m}_exact_recovery"] = recovered
            summary[m] = {
                "threshold": thresholds[m],
                "clean_false_intervention_fraction": float(arrays[f"{m}_clean_intervention"].mean()),
                "fault_intervention_fraction": float(acted.mean()),
                "exact_recovery_fraction": float(recovered.mean()),
                "wrong_bit_intervention_fraction": float(wrong.mean()),
                "abstention_fraction": float((~acted).mean()),
                "correct_bit_fraction_among_interventions": (
                    float(recovered.sum() / acted.sum()) if acted.any() else None
                ),
                "exact_recovery_by_injected_bit": recovered.mean(axis=0).tolist()
            }
        summary["no_intervention"] = {
            "clean_false_intervention_fraction": 0.0,
            "fault_intervention_fraction": 0.0,
            "exact_recovery_fraction": 0.0,
            "wrong_bit_intervention_fraction": 0.0,
            "abstention_fraction": 1.0
        }

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    outcome_bytes = buffer.getvalue()
    payload = {
        "status": "EXPLORATORY_WEIGHT_LEVEL_TEST_COMPLETE",
        "protocol_sha256": PROTOCOL_SHA,
        "calibration_sha256": CALIBRATION_SHA,
        "source_session_fingerprint": fingerprint,
        "module_sha256": sha(Path(__file__).read_bytes()),
        "clean_targets": n,
        "fault_scenarios_per_method": n * 8,
        "methods": summary,
        "outcomes_sha256": sha(outcome_bytes),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in session["environment"]["packages"]},
        "wall_seconds": time.perf_counter() - started,
        "limitations": protocol["limitations"],
        "uncertainty": "Pending analysis of saved outcomes; no repeated model evaluation needed."
    }
    envelope = {"payload_sha256": sha(canonical(payload)), "payload": payload}
    summary_bytes = (json.dumps(envelope, indent=2, allow_nan=False) + "\n").encode()

    # Preserve a single evaluation attempt; never overwrite existing outputs.
    out.mkdir(parents=True, exist_ok=False)
    for name, content in (
        ("outcomes.npz", outcome_bytes), ("summary.json", summary_bytes)
    ):
        with (out / name).open("xb") as stream:
            stream.write(content)
        if (out / name).read_bytes() != content:
            raise RuntimeError("Saved output verification failed.")

    print(json.dumps(summary, indent=2))
    print("Saved:", out)
    print("No fitting, threshold adjustment, COCO inference, or changes to 05b.")
    return payload
