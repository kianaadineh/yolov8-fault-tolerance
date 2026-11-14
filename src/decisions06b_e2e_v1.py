"""Materialize frozen 06b decisions without fitting or predictor execution."""
import hashlib
import io
import json
from pathlib import Path

PROTOCOL_SHA = "89aa9b56a93fd72c19bceb2e5deff64f06f43e82e47ff1e764ac4e318e3413a8"

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

def record(path):
    envelope = json.loads(path.read_bytes())
    if sha(canonical(envelope["payload"])) != envelope["payload_sha256"]:
        raise RuntimeError("Record checksum mismatch.")
    return envelope["payload"]

def flip(value, bit):
    unsigned = (int(value) & 255) ^ (1 << int(bit))
    return unsigned - 256 if unsigned >= 128 else unsigned

def run(repo):
    import numpy as np
    repo = Path(repo)
    protocol = json.loads(verified(
        repo / "experiments/notebook06b/end_to_end_pilot_protocol_v1.json",
        PROTOCOL_SHA
    ))
    calibration = json.loads(verified(
        repo / "results/notebook06b/calibration_v1.json",
        protocol["calibration_sha256"]
    ))
    if sha(canonical(calibration["payload"])) != calibration["payload_sha256"]:
        raise RuntimeError("Calibration payload mismatch.")
    calibration = calibration["payload"]

    previous = repo / "results/notebook06b/evaluation_v1"
    summary = record(previous / "summary.json")
    if (
        summary["calibration_sha256"] != protocol["calibration_sha256"]
        or summary["protocol_sha256"] != calibration["protocol_sha256"]
        or summary["source_session_fingerprint"] != calibration["source_session_fingerprint"]
    ):
        raise RuntimeError("Evaluation provenance mismatch.")
    raw = verified(previous / "outcomes.npz", summary["outcomes_sha256"])
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        outcomes = {key: archive[key].copy() for key in archive.files}

    dataset = verified(
        repo / "results/notebook06a/masked_bit_dataset_v1/dataset.npz",
        protocol["dataset_sha256"]
    )
    with np.load(io.BytesIO(dataset), allow_pickle=False) as archive:
        test = {key: archive[f"test_{key}"].copy()
                for key in ("bits", "layer", "channel", "flat_index")}

    selected = np.array([s["dataset_test_row"] for s in protocol["scenarios"]])
    clean_scores = {m: np.empty((len(selected), 8)) for m in ("C", "D")}
    source06a = repo / "results/notebook06a/masked_bit_comparison_v1"
    for bit in range(8):
        result = record(source06a / f"bit_{bit}_result.json")
        expected = calibration["prediction_sha256_by_bit"][str(bit)]
        if (
            result["prediction_sha256"] != expected
            or result["session_fingerprint"] != summary["source_session_fingerprint"]
            or result["bit"] != bit
        ):
            raise RuntimeError("Saved predictor output provenance mismatch.")
        raw = verified(source06a / f"bit_{bit}_probabilities.npz", expected)
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            for m in clean_scores:
                p = archive[f"test_{m}"][selected]
                clean_scores[m][:, bit] = np.where(
                    test["bits"][selected, bit] == 0, p, 1-p
                )

    rows = []
    states = {("clean",)}
    for index, scenario in enumerate(protocol["scenarios"]):
        row_index = scenario["dataset_test_row"]
        injected = scenario["injected_bit"]
        for key in ("layer", "channel", "flat_index"):
            expected = scenario["layer_id"] if key == "layer" else scenario[key]
            if int(test[key][row_index]) != expected:
                raise RuntimeError("Selected dataset location mismatch.")
            if int(outcomes[key][row_index]) != expected:
                raise RuntimeError("Selected outcome location mismatch.")

        unsigned = int(np.sum(
            test["bits"][row_index].astype(np.int64) * (1 << np.arange(8))
        ))
        clean = unsigned - 256 if unsigned >= 128 else unsigned
        if clean != scenario["q_clean"] or flip(clean, injected) != scenario["q_faulty"]:
            raise RuntimeError("Selected integer mismatch.")

        row = {"scenario": scenario, "conditions": {}}
        row["conditions"]["shared_INT8_clean"] = {"q_final": clean}
        row["conditions"]["faulty_no_intervention"] = {"q_final": scenario["q_faulty"]}

        for m in ("C", "D"):
            acted = bool(outcomes[f"{m}_fault_intervention"][row_index, injected])
            candidate = int(outcomes[f"{m}_fault_candidate"][row_index, injected])
            final = flip(scenario["q_faulty"], candidate) if acted else scenario["q_faulty"]
            recovered = final == clean
            if recovered != bool(outcomes[f"{m}_exact_recovery"][row_index, injected]):
                raise RuntimeError("Recovery decision mismatch.")
            row["conditions"][f"faulty_after_{m}"] = {
                "acted": acted, "candidate_bit": candidate,
                "q_final": final, "exact_recovery": recovered
            }

            scores = clean_scores[m][index]
            candidate = int(scores.argmax())
            acted = bool(scores[candidate] > calibration["methods"][m]["threshold"])
            if acted != bool(outcomes[f"{m}_clean_intervention"][row_index]):
                raise RuntimeError("Clean decision mismatch.")
            row["conditions"][f"clean_target_after_{m}"] = {
                "acted": acted, "candidate_bit": candidate,
                "q_final": flip(clean, candidate) if acted else clean
            }

        for condition in row["conditions"].values():
            final = condition["q_final"]
            states.add(
                ("clean",) if final == clean else
                (scenario["layer"], scenario["flat_index"], final)
            )
        rows.append(row)

    payload = {
        "status": "FROZEN_DECISIONS_MATERIALIZED_NO_IMAGE_INFERENCE",
        "protocol_sha256": PROTOCOL_SHA,
        "calibration_sha256": protocol["calibration_sha256"],
        "evaluation_summary_payload_sha256": sha(canonical(summary)),
        "outcomes_sha256": summary["outcomes_sha256"],
        "module_sha256": sha(Path(__file__).read_bytes()),
        "unique_weight_states_including_clean": len(states),
        "rows": rows
    }
    content = (json.dumps(
        {"payload_sha256": sha(canonical(payload)), "payload": payload},
        indent=2, allow_nan=False
    ) + "\n").encode()
    out = repo / "results/notebook06b/end_to_end_decisions_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if out.read_bytes() != content:
            raise RuntimeError("Existing decisions differ; nothing overwritten.")
    else:
        with out.open("xb") as stream:
            stream.write(content)
    if out.read_bytes() != content:
        raise RuntimeError("Write verification failed.")

    for row in rows:
        c = row["conditions"]
        print(
            row["scenario"]["scenario_id"],
            f"C: acted={c['faulty_after_C']['acted']}, recovered={c['faulty_after_C']['exact_recovery']}",
            f"D: acted={c['faulty_after_D']['acted']}, recovered={c['faulty_after_D']['exact_recovery']}"
        )
    for m in ("C", "D"):
        count = sum(r["conditions"][f"clean_target_after_{m}"]["acted"] for r in rows)
        print(f"{m}: clean-target interventions={count}/16")
    print("Unique weight states including clean:", len(states))
    print("Saved:", out)
    print("SHA256:", sha(content))
    print("No model fitting, predictor execution, or image inference.")
    return payload
