"""Verify saved end-to-end records and report all frozen conditions."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

def sha(content):
    return hashlib.sha256(content).hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode()

def unpack(content):
    envelope = json.loads(content)
    if sha(canonical(envelope["payload"])) != envelope["payload_sha256"]:
        raise RuntimeError("Record payload checksum mismatch.")
    return envelope["payload"]

def read_gzip(path):
    return unpack(gzip.decompress(path.read_bytes()))

def publish(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Existing file differs: {path.name}")
    else:
        with path.open("xb") as stream:
            stream.write(content)
    if path.read_bytes() != content:
        raise RuntimeError("Write verification failed.")

def run(repo):
    repo = Path(repo)
    folder = repo / "results/notebook06b/end_to_end_pilot_v1"
    session = read_gzip(folder / "session.json.gz")
    fingerprint = sha(canonical(session))

    decision_path = repo / "results/notebook06b/end_to_end_decisions_v1.json"
    decision_bytes = decision_path.read_bytes()
    if sha(decision_bytes) != session["decisions_sha256"]:
        raise RuntimeError("Decisions checksum mismatch.")
    decisions = unpack(decision_bytes)

    evaluations = {}
    hashes = {}
    for index, state in enumerate(session["states"]):
        evaluation_path = folder / f"state_{index:03d}_evaluation.json.gz"
        record = read_gzip(evaluation_path)
        prediction_path = folder / f"state_{index:03d}_predictions.json.gz"
        prediction = read_gzip(prediction_path)
        if (
            record["fingerprint"] != fingerprint
            or prediction["fingerprint"] != fingerprint
            or record["index"] != index
            or prediction["index"] != index
            or record["state"] != state
            or record["state_sha256"] != prediction["state_sha256"]
            or record["prediction_sha256"] != sha(prediction_path.read_bytes())
            or prediction["data"]["image_ids"] != session["image_ids"]
            or record["exact_clean_restoration"] is not True
        ):
            raise RuntimeError(f"State provenance mismatch: {index}")
        evaluations[index] = record
        hashes[str(index)] = sha(evaluation_path.read_bytes())

    rows, detailed = [], []
    for item in decisions["rows"]:
        scenario = item["scenario"]
        sid = scenario["scenario_id"]
        metrics = {}
        for condition, action in item["conditions"].items():
            state_index = session["condition_states"][sid + "/" + condition]
            expected_state = (
                None if action["q_final"] == scenario["q_clean"] else
                [scenario["layer"], scenario["flat_index"], action["q_final"]]
            )
            if session["states"][state_index] != expected_state:
                raise RuntimeError("Condition-to-state mapping mismatch.")
            metrics[condition] = evaluations[state_index]["metrics"]
            detailed.append({
                "scenario_id": sid,
                "condition": condition,
                "state_index": state_index,
                "q_final": action["q_final"],
                **metrics[condition]
            })

        clean = metrics["shared_INT8_clean"]["AP50_95"]
        faulty = metrics["faulty_no_intervention"]["AP50_95"]
        row = {
            "scenario_id": sid,
            "layer": scenario["layer"],
            "bit": scenario["injected_bit"],
            "clean_AP50_95": clean,
            "faulty_AP50_95": faulty,
            "fault_minus_clean_pp": 100 * (faulty - clean),
        }
        for method in ("C", "D"):
            after = metrics[f"faulty_after_{method}"]["AP50_95"]
            clean_after = metrics[f"clean_target_after_{method}"]["AP50_95"]
            row[f"{method}_AP50_95"] = after
            row[f"{method}_gain_vs_faulty_pp"] = 100 * (after - faulty)
            row[f"{method}_change_vs_clean_pp"] = 100 * (after - clean)
            row[f"{method}_clean_control_change_pp"] = 100 * (clean_after - clean)
            row[f"{method}_exact_weight_recovery"] = item["conditions"][
                f"faulty_after_{method}"
            ]["exact_recovery"]
        rows.append(row)

    aggregates = {}
    for method in ("C", "D"):
        gains = [r[f"{method}_gain_vs_faulty_pp"] for r in rows]
        aggregates[method] = {
            "mean_signed_AP_gain_pp": sum(gains) / len(gains),
            "positive_gain_scenarios": sum(x > 0 for x in gains),
            "zero_gain_scenarios": sum(x == 0 for x in gains),
            "negative_gain_scenarios": sum(x < 0 for x in gains),
            "exact_weight_recoveries": sum(
                r[f"{method}_exact_weight_recovery"] for r in rows
            )
        }

    payload = {
        "status": "EXPLORATORY_END_TO_END_PILOT_COMPLETE",
        "session_fingerprint": fingerprint,
        "session_file_sha256": sha((folder / "session.json.gz").read_bytes()),
        "evaluation_sha256_by_state": hashes,
        "report_module_sha256": sha(Path(__file__).read_bytes()),
        "scenario_count": len(rows),
        "unique_states": len(evaluations),
        "aggregates": aggregates,
        "rows": rows,
        "limitations": [
            "16 exploratory scenarios, two per bit; no population efficacy claim.",
            "Selected target weight is given; no network-wide localization.",
            "Development images; previous test outcomes inspected.",
            "Exact weight restoration can reduce AP when the fault increased sample AP.",
            "No clean-target interventions in these 16 controls does not imply zero false alarms.",
            "05b screening remains separate and unchanged."
        ]
    }
    destination = repo / "docs/research_reports/notebook06b_end_to_end_v1"
    for name, records in (
        ("scenario_comparison_v1.csv", rows),
        ("all_conditions_cocoeval_v1.csv", detailed),
    ):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        publish(destination / name, stream.getvalue().encode())

    content = (json.dumps(
        {"payload_sha256": sha(canonical(payload)), "payload": payload},
        indent=2, allow_nan=False
    ) + "\n").encode()
    publish(destination / "summary_v1.json", content)

    print("Verified states:", len(evaluations))
    print("Scenario | fault-clean pp | C gain pp | D gain pp")
    for row in rows:
        print(
            f"{row['scenario_id']} | {row['fault_minus_clean_pp']:+.8f} | "
            f"{row['C_gain_vs_faulty_pp']:+.8f} | "
            f"{row['D_gain_vs_faulty_pp']:+.8f}"
        )
    print(json.dumps(aggregates, indent=2))
    print("Saved:", destination)
    print("Summary SHA256:", sha(content))
    print("No inference, threshold changes, or 05b execution.")
    return payload
