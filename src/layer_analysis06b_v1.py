"""Descriptive per-layer analysis of saved 06b outcomes."""
import csv
import hashlib
import io
import json
from pathlib import Path

def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")
    ).encode()

def read_record(path):
    envelope = json.loads(path.read_bytes())
    if hashlib.sha256(canonical(envelope["payload"])).hexdigest() != envelope["payload_sha256"]:
        raise RuntimeError("Record checksum mismatch.")
    return envelope["payload"]

def run(repo):
    import numpy as np
    repo = Path(repo)
    previous = repo / "results/notebook06b/evaluation_v1"
    summary = read_record(previous / "summary.json")
    raw = (previous / "outcomes.npz").read_bytes()
    if hashlib.sha256(raw).hexdigest() != summary["outcomes_sha256"]:
        raise RuntimeError("Outcome checksum mismatch.")

    manifest_path = repo / "results/notebook06a/masked_bit_dataset_v1/manifest.json"
    session = read_record(
        repo / "results/notebook06a/masked_bit_comparison_v1/session.json"
    )
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != session["manifest_sha256"]:
        raise RuntimeError("Manifest checksum mismatch.")
    if hashlib.sha256(canonical(session)).hexdigest() != summary["source_session_fingerprint"]:
        raise RuntimeError("Source session mismatch.")
    manifest = json.loads(manifest_bytes)

    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}

    rows = []
    for item in manifest["inventory"]:
        mask = arrays["layer"] == item["layer_id"]
        n = int(mask.sum())
        if n == 0:
            raise RuntimeError("Missing test layer.")
        row = {
            "layer_id": item["layer_id"],
            "layer": item["layer"],
            "test_targets": n,
            "observed_test_channels": int(len(np.unique(arrays["channel"][mask]))),
            "fault_scenarios_per_method": n * 8,
        }
        for method in ("C", "D"):
            acted = arrays[f"{method}_fault_intervention"][mask]
            recovered = arrays[f"{method}_exact_recovery"][mask]
            row[f"{method}_recovery_pct"] = float(recovered.mean() * 100)
            row[f"{method}_clean_false_intervention_pct"] = float(
                arrays[f"{method}_clean_intervention"][mask].mean() * 100
            )
            row[f"{method}_wrong_bit_intervention_pct"] = float(
                (acted & ~recovered).mean() * 100
            )
            for bit in range(8):
                row[f"{method}_recovery_bit_{bit}_pct"] = float(
                    recovered[:, bit].mean() * 100
                )
        row["recovery_C_minus_D_pp"] = row["C_recovery_pct"] - row["D_recovery_pct"]
        row["clean_harm_C_minus_D_pp"] = (
            row["C_clean_false_intervention_pct"]
            - row["D_clean_false_intervention_pct"]
        )
        rows.append(row)

    if len(rows) != 63 or sum(r["test_targets"] for r in rows) != summary["clean_targets"]:
        raise RuntimeError("Layer coverage mismatch.")

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    content = stream.getvalue().encode("utf-8")
    out = repo / "docs/research_reports/notebook06b_per_layer_v1.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if out.read_bytes() != content:
            raise RuntimeError("Existing table differs; nothing overwritten.")
    else:
        with out.open("xb") as handle:
            handle.write(content)
    if out.read_bytes() != content:
        raise RuntimeError("Write verification failed.")

    better = sum(r["recovery_C_minus_D_pp"] > 0 for r in rows)
    equal = sum(r["recovery_C_minus_D_pp"] == 0 for r in rows)
    worse = len(rows) - better - equal
    print(f"Layers: {len(rows)} | C recovery higher: {better}, equal: {equal}, lower: {worse}")
    print("Layer | targets | C recovery % | D recovery % | recovery delta pp | clean harm delta pp")
    for row in rows:
        print(
            f"{row['layer']} | {row['test_targets']} | "
            f"{row['C_recovery_pct']:.3f} | {row['D_recovery_pct']:.3f} | "
            f"{row['recovery_C_minus_D_pp']:+.3f} | "
            f"{row['clean_harm_C_minus_D_pp']:+.3f}"
        )
    print("Saved:", out)
    print("CSV SHA256:", hashlib.sha256(content).hexdigest())
    print("Descriptive only; no per-layer uncertainty or 05b sensitivity ranking.")
    return rows
