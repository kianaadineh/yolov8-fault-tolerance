"""Resumable frozen 06b end-to-end pilot; independent of campaign output."""
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import platform
import time
import types
from pathlib import Path

PROTOCOL_SHA = "89aa9b56a93fd72c19bceb2e5deff64f06f43e82e47ff1e764ac4e318e3413a8"
DECISIONS_SHA = "89a9a3c3e2b4f8187089325a765b1a715b7059d40ed8446fe1f08e3881fab2c0"

def sha(content):
    return hashlib.sha256(content).hexdigest()

def verified(path, expected):
    content = path.read_bytes()
    if sha(content) != expected:
        raise RuntimeError(f"Checksum mismatch: {path.name}")
    return content

class Pilot:
    def __init__(self, project_root, device="cpu"):
        import torch
        from importlib.metadata import version
        from pycocotools.coco import COCO

        if device not in ("cpu", "cuda:0"):
            raise ValueError("Choose explicitly: cpu or cuda:0.")
        if device == "cuda:0" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no silent device change.")
        self.root = Path(project_root)
        self.repo = self.root / "code"
        self.out = self.repo / "results/notebook06b/end_to_end_pilot_v1"
        self.protocol = json.loads(verified(
            self.repo / "experiments/notebook06b/end_to_end_pilot_protocol_v1.json",
            PROTOCOL_SHA
        ))
        envelope = json.loads(verified(
            self.repo / "results/notebook06b/end_to_end_decisions_v1.json",
            DECISIONS_SHA
        ))
        canonical = lambda x: json.dumps(
            x, sort_keys=True, ensure_ascii=False,
            allow_nan=False, separators=(",", ":")
        ).encode()
        self.decisions = envelope["payload"]
        if sha(canonical(self.decisions)) != envelope["payload_sha256"]:
            raise RuntimeError("Decisions payload mismatch.")
        if self.decisions["protocol_sha256"] != PROTOCOL_SHA:
            raise RuntimeError("Decisions protocol mismatch.")
        if [r["scenario"] for r in self.decisions["rows"]] != self.protocol["scenarios"]:
            raise RuntimeError("Decision scenario mismatch.")

        plan = json.loads(verified(
            self.repo / "experiments/notebook05b/int8_screening_plan_v1.json",
            self.protocol["screening_plan_sha256"]
        ))
        source_path = self.repo / "src/screen05b_campaign_v1.py"
        verified(source_path, plan["source_sha256"])
        spec = importlib.util.spec_from_file_location("verified_screen_e2e", source_path)
        screen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(screen)
        self.c = screen.Campaign(self.root)
        self.c._load_sources()
        torch.set_num_threads(1)
        self.c._model(device)
        if self.c.clean_hash != self.protocol["clean_model_state_sha256"]:
            raise RuntimeError("Clean state mismatch.")
        if self.c.ids != self.protocol["image_ids"]:
            raise RuntimeError("Image IDs mismatch.")
        if self.c.config["settings"] != self.protocol["settings"]:
            raise RuntimeError("Inference settings mismatch.")

        self.c.device = device
        self.c._synchronize = (
            (lambda: torch.cuda.synchronize(device))
            if device == "cuda:0" else (lambda: None)
        )
        self.c._predict = types.MethodType(portable_predict, self.c)
        self.c._cache()
        self.c.category_ids = {r["id"] for r in self.c.categories}
        with contextlib.redirect_stdout(io.StringIO()):
            self.c.gt = COCO(str(self.c.annotation_path))

        self.states = [None]
        self.condition_states = {}
        for row in self.decisions["rows"]:
            scenario = row["scenario"]
            q, _ = self.c.quantized[scenario["layer"]]
            actual = int(q.flatten()[scenario["flat_index"]].item())
            if actual != scenario["q_clean"]:
                raise RuntimeError("Clean target mismatch.")
            for condition, action in row["conditions"].items():
                final = action["q_final"]
                if type(final) is not int or not -128 <= final <= 127:
                    raise RuntimeError("Invalid final integer.")
                state = None if final == actual else (
                    scenario["layer"], scenario["flat_index"], final
                )
                if state not in self.states:
                    self.states.append(state)
                self.condition_states[
                    scenario["scenario_id"] + "/" + condition
                ] = self.states.index(state)
        if len(self.states) != self.decisions["unique_weight_states_including_clean"]:
            raise RuntimeError("Unique state count mismatch.")

        self.session = {
            "protocol_sha256": PROTOCOL_SHA,
            "decisions_sha256": DECISIONS_SHA,
            "module_sha256": sha(Path(__file__).read_bytes()),
            "screen_source_sha256": plan["source_sha256"],
            "device": device,
            "python": platform.python_version(),
            "packages": {p: version(p) for p in (
                "torch", "torchvision", "ultralytics", "numpy",
                "pycocotools", "opencv-python", "Pillow"
            )},
            "gpu": torch.cuda.get_device_name(device) if device == "cuda:0" else None,
            "threads": torch.get_num_threads(),
            "tf32": False,
            "deterministic_algorithms": True,
            "image_ids": self.c.ids,
            "image_sha256": self.c.image_hashes,
            "clean_state": self.c.clean_hash,
            "states": [list(s) if s is not None else None for s in self.states],
            "condition_states": self.condition_states
        }
        self.fingerprint = sha(canonical(self.session))
        self.c._assert_state(self.c.clean_hash)
        print("Prepared:", device, "| unique states:", len(self.states))
        print("No COCO image inference yet.")

    def _read(self, path):
        record = self.c.base.read_record(path)
        if record["fingerprint"] != self.fingerprint:
            raise RuntimeError("Stored execution fingerprint mismatch.")
        return record

    def _execute(self, index):
        import torch
        state = self.states[index]
        self.c._assert_state(self.c.clean_hash)
        target = original = None
        try:
            if state is not None:
                layer, flat, final = state
                target = self.c.modules[layer].weight
                original = target.detach().clone()
                q, scales = self.c.quantized[layer]
                channel = flat // (q.numel() // q.shape[0])
                value = (
                    torch.tensor(final, dtype=torch.float32)
                    * scales.flatten()[channel]
                ).item()
                with torch.no_grad():
                    target.flatten()[flat] = value
                changed = (target != original).flatten().nonzero().flatten().tolist()
                if changed != [flat]:
                    raise RuntimeError("Single-weight isolation failed.")
                with torch.no_grad():
                    target.flatten()[flat] = original.flatten()[flat]
                self.c._assert_state(self.c.clean_hash)
                with torch.no_grad():
                    target.flatten()[flat] = value

            expected_state = self.c._state()
            prediction_path = self.out / f"state_{index:03d}_predictions.json.gz"
            if prediction_path.exists():
                saved = self._read(prediction_path)
                if saved["index"] != index or saved["state_sha256"] != expected_state:
                    raise RuntimeError("Saved prediction state mismatch.")
                predictions = saved["data"]
            else:
                predictions = self.c._predict()
                self.c._assert_state(expected_state)
                self.c._validate_predictions(predictions)
                self.c.base.write_record(prediction_path, {
                    "fingerprint": self.fingerprint, "index": index,
                    "state_sha256": expected_state, "data": predictions
                })
            self.c._validate_predictions(predictions)
            self.c._assert_state(expected_state)
        finally:
            if original is not None:
                with torch.no_grad():
                    target.copy_(original)
            self.c._assert_state(self.c.clean_hash)

        metrics, seconds = self.c._evaluate(predictions["predictions"])
        if any(not math.isfinite(v) for v in metrics.values()) or metrics["AP50_95"] < 0:
            raise RuntimeError("Invalid COCO metrics.")
        result = {
            "fingerprint": self.fingerprint, "index": index,
            "state": list(state) if state is not None else None,
            "state_sha256": expected_state,
            "prediction_sha256": sha(prediction_path.read_bytes()),
            "metrics": metrics, "evaluation_seconds": seconds,
            "exact_clean_restoration": True
        }
        self.c.base.write_record(
            self.out / f"state_{index:03d}_evaluation.json.gz", result
        )
        return result

    def run(self, max_new_states=1):
        import fcntl
        if type(max_new_states) is not int or max_new_states < 0:
            raise ValueError("Use a nonnegative integer state limit.")
        self.out.mkdir(parents=True, exist_ok=True)
        with (self.out / "writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if list(self.out.glob("*.part")) or list(self.out.glob("*_failure.json.gz")):
                raise RuntimeError("Partial/failure records require review.")
            session_path = self.out / "session.json.gz"
            if session_path.exists():
                if self.c.base.read_record(session_path) != self.session:
                    raise RuntimeError("Resume session differs; no overwrite.")
            else:
                self.c.base.write_record(session_path, self.session)

            completed, new = 0, 0
            for index in range(len(self.states)):
                path = self.out / f"state_{index:03d}_evaluation.json.gz"
                if path.exists():
                    result = self._read(path)
                    expected = list(self.states[index]) if self.states[index] is not None else None
                    if result["index"] != index or result["state"] != expected:
                        raise RuntimeError("Saved evaluation state mismatch.")
                    prediction_path = self.out / f"state_{index:03d}_predictions.json.gz"
                    if result["prediction_sha256"] != sha(prediction_path.read_bytes()):
                        raise RuntimeError("Saved prediction checksum mismatch.")
                    prediction = self._read(prediction_path)
                    if prediction["state_sha256"] != result["state_sha256"]:
                        raise RuntimeError("Prediction/evaluation state mismatch.")
                    self.c._validate_predictions(prediction["data"])
                    completed += 1
                    continue
                if new >= max_new_states:
                    break
                try:
                    result = self._execute(index)
                except BaseException as exc:
                    failure = self.out / f"state_{index:03d}_failure.json.gz"
                    if not failure.exists():
                        self.c.base.write_record(failure, {
                            "fingerprint": self.fingerprint, "index": index,
                            "exception": type(exc).__name__, "message": str(exc),
                            "clean_state_restored": self.c._state() == self.c.clean_hash
                        })
                    raise
                new += 1
                completed += 1
                print(
                    f"State {index:03d}: AP50:95={result['metrics']['AP50_95']:.9f} | saved",
                    flush=True
                )
            self.c._assert_state(self.c.clean_hash)
            print(f"Completed states: {completed}/{len(self.states)} | new: {new}")
            print("Output:", self.out)
            print("05b campaign output unchanged.")
            return {"completed_states": completed, "new_states": new}


def portable_predict(self):
    import cv2
    import numpy as np
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    torch = self.torch
    transform = LetterBox(new_shape=(640,640), auto=False, scale_fill=False, scaleup=True, stride=32)
    predictions, per_image = [], []
    self._synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for image_id in self.ids:
            image = cv2.imdecode(np.frombuffer(self.cache[image_id],dtype=np.uint8),cv2.IMREAD_COLOR)
            record = self.images[image_id]
            if image is None or image.shape[:2] != (record['height'],record['width']):
                raise RuntimeError('Image decode/dimensions invalid.')
            padded = transform(image=image)
            array = np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1))
            tensor = (torch.from_numpy(array).unsqueeze(0).float()/255).to(self.device)
            result = self.model(tensor)
            raw = (result[0] if isinstance(result,(tuple,list)) else result).detach().cpu()
            if tuple(raw.shape) != (1,84,8400) or not torch.isfinite(raw).all():
                raise RuntimeError('Invalid raw output.')
            boxes = non_max_suppression(raw.clone(), conf_thres=.001,iou_thres=.7,nc=80,max_det=300)[0].clone()
            boxes[:,:4] = scale_boxes(tensor.shape[2:],boxes[:,:4],image.shape[:2])
            if not torch.isfinite(boxes).all():
                raise RuntimeError('Non-finite post-NMS output.')
            count, degenerate = 0, 0
            for box in boxes:
                x1,y1,x2,y2,score,cls = box.tolist()
                if x2 <= x1 or y2 <= y1:
                    degenerate += 1
                    continue
                if not (cls == int(cls) and 0<=cls<80 and 0<=score<=1
                        and 0<=x1<x2<=record['width']+1e-4 and 0<=y1<y2<=record['height']+1e-4):
                    raise RuntimeError('Invalid detection geometry/class/score.')
                predictions.append({'image_id':image_id,'category_id':self.categories[int(cls)]['id'],
                                    'bbox':[x1,y1,x2-x1,y2-y1],'score':score})
                count += 1
            per_image.append({'image_id':image_id,'count':count,'degenerate':degenerate})
    self._synchronize()
    return {'image_ids':self.ids,'predictions':predictions,'per_image':per_image,
            'pipeline_seconds':time.perf_counter()-started}
