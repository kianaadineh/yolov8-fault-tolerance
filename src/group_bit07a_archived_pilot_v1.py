"""Retrospective reuse of archived metrics for identical logical weight states."""
import gzip
import hashlib
import io
import json
from pathlib import Path

CHECK_SHA = "5c2bdc9e1f1b275b0627057b8cbcc45a670b0798c2cf454a8c4898f8f783de29"
DECISIONS_SHA = "89a9a3c3e2b4f8187089325a765b1a715b7059d40ed8446fe1f08e3881fab2c0"

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

def compressed_record(path):
    return unpack(gzip.decompress(path.read_bytes()))

def run(repo):
    import numpy as np
    repo = Path(repo)
    design = repo / "results/notebook07a/group_bit_design_comparison_v1"
    destination = design / "archived_16_scenario_pilot_v1.json"
    if destination.exists():
        raise FileExistsError("Report exists; inspect before rerunning.")

    checks = unpack(verified(design / "nominee_correctness_v1.json", CHECK_SHA))
    descriptor_bytes = verified(
        design / (checks["candidate"]["name"] + "_descriptor.json"),
        checks["descriptor_sha256"]
    )
    descriptor = json.loads(descriptor_bytes)
    if descriptor["candidate"] != checks["candidate"]:
        raise RuntimeError("Candidate mismatch.")
    raw_masks = verified(
        design / descriptor["mask_archive"], checks["mask_archive_sha256"]
    )
    layers = {row["layer"]: row for row in descriptor["layers"]}
    decisions = unpack(verified(
        repo / "results/notebook06b/end_to_end_decisions_v1.json", DECISIONS_SHA
    ))

    previous = repo / "results/notebook06b/end_to_end_pilot_v1"
    session = compressed_record(previous / "session.json.gz")
    fingerprint = sha(canonical(session))
    if session["decisions_sha256"] != DECISIONS_SHA:
        raise RuntimeError("Archived decisions mismatch.")

    protocol_path = repo / "experiments/notebook06b/end_to_end_pilot_protocol_v1.json"
    archived_protocol = json.loads(verified(protocol_path, session["protocol_sha256"]))
    comparison = unpack(verified(
        design / "summary_v1.json", checks["comparison_summary_sha256"]
    ))
    if comparison["clean_model_state_sha256"] != archived_protocol["clean_model_state_sha256"]:
        raise RuntimeError("Clean model reference mismatch.")

    metrics = {}
    evaluation_hashes = {}
    for index, state in enumerate(session["states"]):
        evaluation_path = previous / f"state_{index:03d}_evaluation.json.gz"
        evaluation = compressed_record(evaluation_path)
        prediction_path = previous / f"state_{index:03d}_predictions.json.gz"
        prediction_bytes = verified(prediction_path, evaluation["prediction_sha256"])
        prediction = unpack(gzip.decompress(prediction_bytes))
        if (
            evaluation["fingerprint"] != fingerprint
            or prediction["fingerprint"] != fingerprint
            or evaluation["index"] != index or prediction["index"] != index
            or evaluation["state"] != state
            or evaluation["state_sha256"] != prediction["state_sha256"]
            or prediction["data"]["image_ids"] != session["image_ids"]
            or evaluation["exact_clean_restoration"] is not True
        ):
            raise RuntimeError("Archived state provenance mismatch.")
        metrics[index] = evaluation["metrics"]
        evaluation_hashes[str(index)] = sha(evaluation_path.read_bytes())

    bits = checks["candidate"]["bits"]
    group_size = checks["candidate"]["group_size"]
    rows = []
    with np.load(io.BytesIO(raw_masks), allow_pickle=False) as archive:
        for item in decisions["rows"]:
            scenario = item["scenario"]
            sid = scenario["scenario_id"]
            layer = layers[scenario["layer"]]
            packed = archive[layer["array_key"]]
            if sha(packed.tobytes()) != layer["mask_array_sha256"]:
                raise RuntimeError("Layer mask mismatch.")

            o, i, kh, kw = layer["shape"]
            width = i * kh * kw
            flat = scenario["flat_index"]
            if not 0 <= flat < o * width:
                raise RuntimeError("Invalid target position.")
            channel, within = divmod(flat, width)
            if channel != scenario["channel"]:
                raise RuntimeError("Target channel mismatch.")
            complete = (width // group_size) * group_size
            zero = one = 0
            if within < complete:
                group_index = channel * (width // group_size) + within // group_size
                code = int(packed[group_index])
                for offset, bit in enumerate(bits):
                    zero |= ((code >> offset) & 1) << bit
                    one |= ((code >> (offset + len(bits))) & 1) << bit

            clean_unsigned = scenario["q_clean"] & 255
            faulty_unsigned = scenario["q_faulty"] & 255
            if (
                clean_unsigned & zero
                or (clean_unsigned & one) != one
                or zero & one
            ):
                raise RuntimeError("Masks disagree with clean target.")
            if clean_unsigned ^ faulty_unsigned != 1 << scenario["injected_bit"]:
                raise RuntimeError("Not the declared single-bit fault.")

            final_unsigned = (faulty_unsigned & (~zero & 255)) | one
            final = final_unsigned - 256 if final_unsigned >= 128 else final_unsigned
            clean_index = session["condition_states"][sid + "/shared_INT8_clean"]
            fault_index = session["condition_states"][sid + "/faulty_no_intervention"]

            if session["states"][clean_index] is not None:
                raise RuntimeError("Archived clean-state mapping mismatch.")
            if session["states"][fault_index] != [
                scenario["layer"], flat, scenario["q_faulty"]
            ]:
                raise RuntimeError("Archived fault-state mapping mismatch.")

            if final == scenario["q_clean"]:
                reuse_index = clean_index
            elif final == scenario["q_faulty"]:
                reuse_index = fault_index
            else:
                raise RuntimeError("New weight state requires new inference.")

            clean_ap = metrics[clean_index]["AP50_95"]
            fault_ap = metrics[fault_index]["AP50_95"]
            after_ap = metrics[reuse_index]["AP50_95"]
            rows.append({
                "scenario_id": sid,
                "layer": scenario["layer"],
                "bit": scenario["injected_bit"],
                "q_final": final,
                "exact_recovery": final == scenario["q_clean"],
                "reused_state_index": reuse_index,
                "clean_AP50_95": clean_ap,
                "faulty_AP50_95": fault_ap,
                "after_AP50_95": after_ap,
                "AP_gain_pp": 100 * (after_ap - fault_ap),
                "all_12_metrics_after": metrics[reuse_index]
            })

    gains = [row["AP_gain_pp"] for row in rows]
    payload = {
        "status": "RETROSPECTIVE_ARCHIVED_STATE_REUSE_NOT_NEW_INFERENCE",
        "candidate": checks["candidate"],
        "nominee_correctness_sha256": CHECK_SHA,
        "descriptor_sha256": checks["descriptor_sha256"],
        "archived_session_fingerprint": fingerprint,
        "evaluation_sha256_by_state": evaluation_hashes,
        "module_sha256": sha(Path(__file__).read_bytes()),
        "scenario_count": len(rows),
        "exact_recoveries": sum(row["exact_recovery"] for row in rows),
        "mean_signed_AP_gain_pp": sum(gains) / len(gains),
        "positive_gain_scenarios": sum(value > 0 for value in gains),
        "negative_gain_scenarios": sum(value < 0 for value in gains),
        "rows": rows,
        "limitations": [
            "Previously inspected 16 development scenarios; retrospective only.",
            "Metrics reused for identical logical clean or single-weight fault states.",
            "No fresh CPU or GPU model inference; no new COCOeval run.",
            "Current image files and reconstructed model states were not rechecked here.",
            "State equivalence relies on validated clean-mask invariance and single-weight fault scope.",
            "No independent AP efficacy or novelty confirmation."
        ]
    }
    content = (json.dumps({
        "payload_sha256": sha(canonical(payload)), "payload": payload
    }, indent=2, allow_nan=False) + "\n").encode()
    with destination.open("xb") as stream:
        stream.write(content)
    if destination.read_bytes() != content:
        raise RuntimeError("Write verification failed.")

    print("Scenario | exact recovery | AP gain, percentage points")
    for row in rows:
        print(
            f"{row['scenario_id']} | {row['exact_recovery']} | "
            f"{row['AP_gain_pp']:+.8f}"
        )
    print("Exact recoveries:", payload["exact_recoveries"], "/", len(rows))
    print("Mean signed AP gain, pp:", payload["mean_signed_AP_gain_pp"])
    print("Saved:", destination)
    print("SHA256:", sha(content))
    print("No inference, parameter changes, or 05b campaign modifications.")
    return payload
