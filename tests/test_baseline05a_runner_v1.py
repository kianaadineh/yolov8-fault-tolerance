"""Standard-library tests; no model inference or dependency installation."""
import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline05a_runner_v1 import (MODELS, canonical, read_record,
                                  validate_chunk, validate_split, write_record)


def fixture():
    prediction = {"image_id": 1, "category_id": 1, "bbox": [0, 0, 3, 4], "score": 0.8}
    return {
        "fingerprint": "known", "status": "PAIRED_PREDICTIONS_COMPLETE", "image_ids": [1, 2],
        "predictions": {name: [copy.deepcopy(prediction)] for name in MODELS},
        "per_image": [
            {"image_id": 1, "image_sha256": "a" * 64, "counts": {name: 1 for name in MODELS}},
            {"image_id": 2, "image_sha256": "b" * 64, "counts": {name: 0 for name in MODELS}},
        ],
    }


class StorageAndProvenanceTests(unittest.TestCase):
    def test_round_trip_and_exclusive_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json.gz"
            write_record(path, fixture())
            self.assertEqual(read_record(path), fixture())
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_record(path, {"different": True})
            self.assertEqual(path.read_bytes(), original)

    def test_partial_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json.gz"
            part = Path(str(path) + ".part")
            part.write_bytes(b"interrupted")
            with self.assertRaises(FileExistsError):
                write_record(path, fixture())
            self.assertEqual(part.read_bytes(), b"interrupted")
            self.assertFalse(path.exists())

    def test_payload_tampering_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json.gz"
            write_record(path, fixture())
            envelope = json.loads(gzip.decompress(path.read_bytes()))
            envelope["payload"]["image_ids"] = [999]
            path.write_bytes(gzip.compress(canonical(envelope)))
            with self.assertRaises(RuntimeError):
                read_record(path)

    def test_truncated_archive_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json.gz"
            write_record(path, fixture())
            path.write_bytes(path.read_bytes()[:12])
            with self.assertRaises((EOFError, OSError)):
                read_record(path)

    def test_nan_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json.gz"
            with self.assertRaises(ValueError):
                write_record(path, {"bad": float("nan")})
            self.assertFalse(path.exists())

    def test_split_duplicate_and_overlap_rejected(self):
        valid = {"image_ids": {"development": list(range(1000)), "held_out": list(range(1000, 5000))}}
        self.assertEqual(len(validate_split(valid)), 1000)
        duplicate = copy.deepcopy(valid)
        duplicate["image_ids"]["development"][-1] = 0
        with self.assertRaises(ValueError):
            validate_split(duplicate)
        overlap = copy.deepcopy(valid)
        overlap["image_ids"]["held_out"][0] = 0
        with self.assertRaises(ValueError):
            validate_split(overlap)

    def test_zero_prediction_image_retained(self):
        validate_chunk(fixture(), "known", [1, 2])

    def test_configuration_mismatch_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_chunk(fixture(), "different", [1, 2])

    def test_missing_execution_image_rejected(self):
        payload = fixture()
        payload["per_image"].pop()
        with self.assertRaises(RuntimeError):
            validate_chunk(payload, "known", [1, 2])

    def test_missing_model_rejected(self):
        payload = fixture()
        del payload["predictions"][MODELS[1]]
        with self.assertRaises(RuntimeError):
            validate_chunk(payload, "known", [1, 2])

    def test_wrong_per_image_counts_rejected(self):
        payload = fixture()
        payload["per_image"][1]["counts"][MODELS[0]] = 1
        with self.assertRaises(RuntimeError):
            validate_chunk(payload, "known", [1, 2])

    def test_non_finite_prediction_rejected(self):
        payload = fixture()
        payload["predictions"][MODELS[0]][0]["score"] = float("nan")
        with self.assertRaises(RuntimeError):
            validate_chunk(payload, "known", [1, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
