"""CPU-only paired FP32/W8A32 clean validation; no fault injection.

Use one writer per output directory. Existing results are never overwritten.
GPU support, fault screening and hardware INT8 claims are outside this module.
"""

from collections import Counter
from datetime import datetime, timezone
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

VERSION = "05a-clean-cpu-v1"
CHECKPOINT_SHA = "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a"
SPLIT_SHA = "143c3273b2bf0cd6393cfe148a74e48fbafb74e770bd3fbdf3f14222355978ee"
GATE_SHA = "dfc04556e731df68af9375c161778e41b87f66601856265703ea8acd69fb64f3"
MODELS = ("fp32_clean", "int8_channel_clean")
PACKAGES = ("torch", "torchvision", "ultralytics", "numpy", "pycocotools", "opencv-python", "Pillow")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_verified(path, expected):
    content = Path(path).read_bytes()
    if digest(content) != expected:
        raise RuntimeError(f"Integrity mismatch: {Path(path).name}")
    return json.loads(content)


def validate_split(split):
    dev = split["image_ids"]["development"]
    held = split["image_ids"]["held_out"]
    if len(dev) != 1000 or len(set(dev)) != 1000:
        raise ValueError("Expected 1000 unique development IDs.")
    if len(held) != 4000 or len(set(held)) != 4000 or set(dev) & set(held):
        raise ValueError("Invalid held-out split.")
    if any(type(i) is not int for i in dev + held):
        raise ValueError("Image IDs must be integers.")
    return dev


def write_record(path, payload):
    """Verified exclusive publication with a crash-visible partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if path.exists() or partial.exists():
        raise FileExistsError(f"Existing record or partial requires inspection: {path.name}")
    envelope = canonical({"payload_sha256": digest(canonical(payload)), "payload": payload})
    with partial.open("xb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
            compressed.write(envelope)
        stream.flush()
        os.fsync(stream.fileno())
    with gzip.open(partial, "rb") as stream:
        if stream.read() != envelope:
            raise RuntimeError("Compressed write verification failed.")
    if path.exists():
        raise FileExistsError("Destination appeared; concurrent writers are unsupported.")
    partial.rename(path)
    return digest(path.read_bytes())


def read_record(path):
    with gzip.open(path, "rb") as stream:
        envelope = json.load(stream)
    payload = envelope["payload"]
    if digest(canonical(payload)) != envelope["payload_sha256"]:
        raise RuntimeError("Record payload checksum mismatch.")
    return payload


def validate_chunk(payload, fingerprint, image_ids):
    if payload["fingerprint"] != fingerprint or payload["image_ids"] != image_ids:
        raise RuntimeError("Chunk configuration or image IDs differ.")
    if payload["status"] != "PAIRED_PREDICTIONS_COMPLETE":
        raise RuntimeError("Chunk is incomplete.")
    if set(payload["predictions"]) != set(MODELS):
        raise RuntimeError("Missing or unexpected model predictions.")
    rows = payload["per_image"]
    if [r["image_id"] for r in rows] != image_ids:
        raise RuntimeError("Per-image execution record differs.")
    for row in rows:
        if len(row["image_sha256"]) != 64:
            raise RuntimeError("Invalid source image checksum.")
    for name in MODELS:
        counts = Counter()
        for prediction in payload["predictions"][name]:
            if prediction["image_id"] not in set(image_ids):
                raise RuntimeError("Unexpected prediction image ID.")
            score = prediction["score"]
            box = prediction["bbox"]
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise RuntimeError("Invalid confidence.")
            if len(box) != 4 or not all(math.isfinite(x) for x in box) or min(box[2:]) <= 0:
                raise RuntimeError("Invalid prediction box.")
            counts[prediction["image_id"]] += 1
        if any(counts[r["image_id"]] != r["counts"][name] for r in rows):
            raise RuntimeError("Per-image prediction count mismatch.")


class CleanSession:
    """Prepare once, smoke-test, then run bounded chunks and evaluate."""

    def __init__(self, project_root, chunk_size=25):
        import cv2
        import numpy as np
        import torch
        from importlib.metadata import version
        from ultralytics import YOLO
        from ultralytics.data.augment import LetterBox

        if chunk_size != 25:
            raise ValueError("Version 1 uses fixed chunks of 25 images.")
        self.cv2, self.np, self.torch = cv2, np, torch
        if version("ultralytics") != "8.4.189":
            raise RuntimeError("This protocol requires Ultralytics 8.4.189.")
        self.root = Path(project_root)
        self.repo = self.root / "code"
        self.out = self.repo / "results/notebook05a/shared_clean_cpu_v1"
        self.chunk_size = chunk_size
        checkpoint = self.root / "models/yolov8s.pt"
        if digest(checkpoint.read_bytes()) != CHECKPOINT_SHA:
            raise RuntimeError("Checkpoint changed.")
        self.split = read_verified(self.repo / "experiments/notebook05/coco2017_split_manifest_v1.json", SPLIT_SHA)
        self.ids = validate_split(self.split)
        self.gate = read_verified(self.repo / "experiments/notebook05/shared_INT8_clean_quality_gate_v1.json", GATE_SHA)
        self.annotation_path = self.root / "datasets/coco2017/annotations/instances_val2017.json"
        annotation = self.annotation_path.read_bytes()
        if digest(annotation) != self.split["annotation_sha256"]:
            raise RuntimeError("Annotation changed.")
        data = json.loads(annotation)
        self.images = {r["id"]: r for r in data["images"]}
        if len(self.images) != 5000 or not set(self.ids) <= set(self.images):
            raise RuntimeError("Annotation image IDs disagree with the split.")
        categories = sorted(data["categories"], key=lambda r: r["id"])
        self.categories = [r["id"] for r in categories]
        torch.manual_seed(20261112)
        torch.use_deterministic_algorithms(True)
        yolo = YOLO(str(checkpoint))
        fp32 = yolo.model.cpu().float().eval()
        if len(categories) != 80 or any(yolo.names[i] != r["name"] for i, r in enumerate(categories)):
            raise RuntimeError("COCO/model class mapping mismatch.")
        int8 = copy.deepcopy(fp32)
        self.quantization = []
        target_keys = set()
        with torch.no_grad():
            for name, module in int8.named_modules():
                if not isinstance(module, torch.nn.Conv2d) or name == "model.22.dfl.conv":
                    continue
                maxima = module.weight.detach().abs().amax(dim=(1, 2, 3), keepdim=True)
                scales = torch.where(maxima > 0, maxima / 127, torch.ones_like(maxima))
                if not torch.isfinite(scales).all() or not (scales > 0).all():
                    raise RuntimeError("Invalid INT8 scale.")
                q = torch.round(module.weight.detach() / scales).clamp(-127, 127).to(torch.int8)
                dequantized = q.float() * scales
                module.weight.copy_(dequantized)
                if not torch.equal(module.weight, dequantized):
                    raise RuntimeError("Quantization verification failed.")
                target_keys.add(name + ".weight")
                self.quantization.append({"layer": name, "shape": list(q.shape),
                                          "scales": scales.flatten().tolist()})
        if len(target_keys) != 63:
            raise RuntimeError("Expected 63 quantized Conv2d layers.")
        reference = fp32.state_dict()
        if any(not torch.equal(t, reference[k]) for k, t in int8.state_dict().items() if k not in target_keys):
            raise RuntimeError("Non-target state changed during quantization.")
        self.models = dict(zip(MODELS, (fp32, int8)))
        self.transform = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True, stride=32)
        with torch.inference_mode():
            for model in self.models.values():
                self._raw(model(torch.zeros(1, 3, 640, 640)))
        self.state_hashes = {name: self._state_hash(model) for name, model in self.models.items()}
        self.config = {
            "version": VERSION, "runner_sha256": digest(Path(__file__).read_bytes()),
            "checkpoint_sha256": CHECKPOINT_SHA, "split_sha256": SPLIT_SHA,
            "annotation_sha256": digest(annotation), "quality_gate_sha256": GATE_SHA,
            "image_ids": self.ids, "chunk_size": chunk_size,
            "environment": {"python": platform.python_version(), "packages": {p: version(p) for p in PACKAGES},
                            "device": "cpu", "torch_threads": torch.get_num_threads()},
            "settings": {"imgsz": 640, "batch": 1, "conf": 0.001, "nms_iou": 0.7,
                         "max_det": 300, "cocoeval_max_dets": [1, 10, 100],
                         "letterbox_auto": False, "scale_fill": False, "scaleup": True,
                         "stride": 32, "color": "RGB", "dtype": "float32",
                         "normalization": "divide_by_255", "deterministic_algorithms": True,
                         "seed": 20261112},
            "model_state_sha256": self.state_hashes,
            "quantization": {"scope": "63_eligible_Conv2d_weights", "axis": 0,
                             "clean_range": [-127, 127], "inference": "dequantized_FP32",
                             "inventory": self.quantization},
        }
        self.fingerprint = digest(canonical(self.config))
        session_file = self.out / "session.json.gz"
        if session_file.exists():
            old = read_record(session_file)
            if old["fingerprint"] != self.fingerprint or old["config"] != self.config:
                raise RuntimeError("Existing session has different code, environment or configuration.")
        else:
            write_record(session_file, {"fingerprint": self.fingerprint, "config": self.config,
                                       "created_at_utc": datetime.now(timezone.utc).isoformat()})
        print("Session prepared on CPU; two synthetic warm-up forwards completed.")
        print("No development image evaluated yet by preparation.")
        print("Fingerprint:", self.fingerprint)

    def _raw(self, result):
        raw = result[0] if isinstance(result, (tuple, list)) else result
        if tuple(raw.shape) != (1, 84, 8400) or not self.torch.isfinite(raw).all():
            raise RuntimeError("Invalid or non-finite raw model output.")
        return raw

    def _state_hash(self, model):
        h = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            a = tensor.detach().cpu().contiguous().numpy()
            h.update(canonical([name, str(a.dtype), list(a.shape)])); h.update(a.tobytes())
        return h.hexdigest()

    def _check_states(self):
        if any(self._state_hash(model) != self.state_hashes[name] for name, model in self.models.items()):
            raise RuntimeError("Model state changed; stop before publishing results.")

    def _image_bytes(self, image_id):
        name = self.images[image_id]["file_name"]
        if Path(name).name != name:
            raise RuntimeError("Unsafe image filename.")
        return (self.root / "datasets/coco2017/images/val2017" / name).read_bytes()

    def _predict(self, image_ids):
        from ultralytics.utils.nms import non_max_suppression
        from ultralytics.utils.ops import scale_boxes

        self._check_states()
        output = {name: [] for name in MODELS}
        rows = []
        start = time.perf_counter()
        with self.torch.inference_mode():
            for image_id in image_ids:
                data = self._image_bytes(image_id)
                image = self.cv2.imdecode(self.np.frombuffer(data, dtype=self.np.uint8), self.cv2.IMREAD_COLOR)
                record = self.images[image_id]
                if image is None or image.shape[:2] != (record["height"], record["width"]):
                    raise RuntimeError(f"Image decoding or dimension failure: {image_id}")
                padded = self.transform(image=image)
                array = self.np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
                tensor = self.torch.from_numpy(array).unsqueeze(0).float() / 255.0
                row = {"image_id": image_id, "image_sha256": digest(data), "counts": {}, "degenerate_boxes": {}}
                for name, model in self.models.items():
                    raw = self._raw(model(tensor))
                    boxes = non_max_suppression(raw.clone(), conf_thres=0.001, iou_thres=0.7, nc=80, max_det=300)[0].clone()
                    boxes[:, :4] = scale_boxes(tensor.shape[2:], boxes[:, :4], image.shape[:2])
                    if not self.torch.isfinite(boxes).all():
                        raise RuntimeError("Non-finite post-NMS output.")
                    count, degenerate = 0, 0
                    for box in boxes:
                        x1, y1, x2, y2, score, cls = box.tolist()
                        if x2 <= x1 or y2 <= y1:
                            degenerate += 1; continue
                        if not (cls == int(cls) and 0 <= cls < 80 and 0 <= score <= 1):
                            raise RuntimeError("Invalid detection class or confidence.")
                        if not (0 <= x1 < x2 <= record["width"] + 1e-4 and 0 <= y1 < y2 <= record["height"] + 1e-4):
                            raise RuntimeError("Detection outside original image bounds.")
                        output[name].append({"image_id": image_id, "category_id": self.categories[int(cls)],
                                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
                        count += 1
                    row["counts"][name] = count; row["degenerate_boxes"][name] = degenerate
                rows.append(row)
        self._check_states()
        payload = {"status": "PAIRED_PREDICTIONS_COMPLETE", "fingerprint": self.fingerprint,
                   "image_ids": image_ids, "predictions": output, "per_image": rows,
                   "wall_seconds": time.perf_counter() - start}
        validate_chunk(payload, self.fingerprint, image_ids)
        return payload

    def smoke(self):
        """Two fixed development images; no AP quality decision."""
        path = self.out / "smoke.json.gz"
        if path.exists():
            payload = read_record(path)
            validate_chunk(payload, self.fingerprint, self.ids[:2])
            print("Existing smoke result verified.")
        else:
            payload = self._predict(self.ids[:2])
            write_record(path, payload)
        print("Smoke images:", payload["image_ids"])
        print("Prediction counts:", {n: len(payload["predictions"][n]) for n in MODELS})
        print("State integrity and finite-output checks: PASS")
        return {"images": payload["image_ids"], "path": str(path)}

    def run(self, max_new_chunks=1):
        """Resume verified chunks; None explicitly allows all remaining chunks."""
        if max_new_chunks is not None and (type(max_new_chunks) is not int or max_new_chunks < 0):
            raise ValueError("Use a non-negative chunk limit or None.")
        new, completed = 0, 0
        for start in range(0, len(self.ids), self.chunk_size):
            image_ids = self.ids[start:start + self.chunk_size]
            path = self.out / f"chunk_{start // self.chunk_size:03d}.json.gz"
            if path.with_name(path.name + ".part").exists():
                raise RuntimeError(f"Partial output requires inspection: {path.name}")
            if path.exists():
                payload = read_record(path)
                validate_chunk(payload, self.fingerprint, image_ids)
                for row in payload["per_image"]:
                    if digest(self._image_bytes(row["image_id"])) != row["image_sha256"]:
                        raise RuntimeError("Source image changed since the saved chunk.")
            elif max_new_chunks is None or new < max_new_chunks:
                payload = self._predict(image_ids)
                write_record(path, payload)
                new += 1
                print(f"Saved paired chunk {start // self.chunk_size + 1}/40 ({len(image_ids)} images)", flush=True)
            else:
                continue
            completed += len(image_ids)
        status = {"completed_images_per_model": completed, "total_images": len(self.ids), "new_chunks": new}
        print(status)
        return status

    def evaluate(self):
        """Joint COCOeval on all 1000 IDs; never average single-image AP."""
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        path = self.out / "evaluation_and_quality_gate.json.gz"
        existing = None
        if path.exists():
            existing = read_record(path)
            if existing["fingerprint"] != self.fingerprint or existing["config"] != self.config:
                raise RuntimeError("Existing evaluation provenance differs.")
        combined = {name: [] for name in MODELS}
        chunk_hashes = {}
        for start in range(0, len(self.ids), self.chunk_size):
            chunk = self.out / f"chunk_{start // self.chunk_size:03d}.json.gz"
            payload = read_record(chunk)
            validate_chunk(payload, self.fingerprint, self.ids[start:start + self.chunk_size])
            for row in payload["per_image"]:
                if digest(self._image_bytes(row["image_id"])) != row["image_sha256"]:
                    raise RuntimeError("Source image changed.")
            chunk_hashes[chunk.name] = digest(chunk.read_bytes())
            for name in MODELS:
                combined[name].extend(payload["predictions"][name])
        if digest(self.annotation_path.read_bytes()) != self.config["annotation_sha256"]:
            raise RuntimeError("Annotation changed before evaluation.")
        if existing is not None:
            if existing["chunk_sha256"] != chunk_hashes:
                raise RuntimeError("Source chunks differ from the existing evaluation.")
            print("Existing evaluation and source chunks verified:", existing["quality_gate_decision"])
            return existing
        gt = COCO(str(self.annotation_path))
        metric_names = ("AP50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                        "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")
        metrics = {}
        for name in MODELS:
            if combined[name]:
                dt = gt.loadRes(copy.deepcopy(combined[name]))
            else:
                dt = COCO(); dt.dataset = {"images": copy.deepcopy(gt.dataset["images"]),
                                          "categories": copy.deepcopy(gt.dataset["categories"]), "annotations": []}
                dt.createIndex()
            evaluator = COCOeval(gt, dt, iouType="bbox")
            evaluator.params.imgIds = list(self.ids)
            evaluator.params.catIds = list(self.categories)
            evaluator.params.maxDets = [1, 10, 100]
            print(f"Evaluating {name} on {len(self.ids)} development images")
            evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
            metrics[name] = dict(zip(metric_names, map(float, evaluator.stats)))
            if any(not math.isfinite(v) for v in metrics[name].values()):
                raise RuntimeError("Non-finite evaluation metric.")
        delta = {k: metrics[MODELS[1]][k] - metrics[MODELS[0]][k] for k in ("AP50_95", "AR100")}
        criteria = self.gate["criteria"]
        passed = (metrics[MODELS[0]]["AP50_95"] >= 0 and metrics[MODELS[1]]["AP50_95"] >= 0
                  and metrics[MODELS[0]]["AR100"] >= 0 and metrics[MODELS[1]]["AR100"] >= 0
                  and delta["AP50_95"] >= criteria["candidate_minus_FP32_AP50_95_minimum"]
                  and delta["AR100"] >= criteria["candidate_minus_FP32_AR100_minimum"])
        report = {"status": "CLEAN_VALIDATION_COMPLETE", "fingerprint": self.fingerprint,
                  "config": self.config, "chunk_sha256": chunk_hashes, "metrics": metrics,
                  "int8_minus_fp32": delta, "quality_gate_decision": "PASS" if passed else "FAIL",
                  "quality_gate": self.gate, "scope": "development_not_final_test",
                  "limitations": ["Engineering point-estimate gate, not statistical equivalence.",
                                  "Logical weight INT8 with FP32 computation; activations are not INT8.",
                                  "Pilot images are included in development; no independent-test claim."]}
        write_record(path, report)
        print("INT8 minus FP32:", delta)
        print("Quality gate:", report["quality_gate_decision"])
        print("Saved:", path)
        print("SHA256:", digest(path.read_bytes()))
        return report


def prepare(project_root):
    return CleanSession(project_root)
