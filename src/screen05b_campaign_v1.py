"""Offline-capable, resumable logical INT8 weight-fault screening.

No downloads, package installations, Git operations, or network requests.
Plan creation is separate from execution. Historical files are read-only.
"""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

VERSION = '1.0'
SEED = 20261114
BOOTSTRAP_SEED = 20261115
WEIGHTS_PER_BIT = 10
BASE_SHA = 'f1069948c16a8899d422930b6dfb2c4ce0f900e5e27e358e0761d588b11492b3'
IMAGE_MANIFEST_SHA = 'f2130cb0b0d505d641c8604dcb075cdcd0830092e5d8b31f594a45dcf1074aeb'
BASELINE_REPORT_SHA = '47861220401cc36ea2c6c19f427d351dcbb5ed39a9067fa79e0abbb175f889d7'
CHECKPOINT_SHA = '1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a'
METRICS = ('AP50_95','AP50','AP75','AP_small','AP_medium','AP_large',
           'AR1','AR10','AR100','AR_small','AR_medium','AR_large')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def digest(content):
    return hashlib.sha256(content).hexdigest()


def verified(path, expected):
    content = Path(path).read_bytes()
    if digest(content) != expected:
        raise RuntimeError(f'File identity mismatch: {Path(path).name}')
    return content


def bitflip(q, bit):
    if type(q) is not int or not -128 <= q <= 127 or type(bit) is not int or not 0 <= bit < 8:
        raise ValueError('Invalid INT8 value or bit.')
    unsigned = (q & 255) ^ (1 << bit)
    return unsigned - 256 if unsigned >= 128 else unsigned


def stratified_loss_summary(rows, draws=2000):
    """Exploratory fault-only stratified bootstrap, conditional on fixed images."""
    groups = {b: [float(r['loss']) for r in rows if r['bit'] == b] for b in range(8)}
    if any(not g for g in groups.values()) or len({len(g) for g in groups.values()}) != 1:
        raise ValueError('Summary requires complete balanced bit strata.')
    if any(not math.isfinite(x) or x < 0 for g in groups.values() for x in g):
        raise ValueError('Losses must be finite and nonnegative.')
    per_bit = {str(b): {'n': len(g), 'mean_loss': sum(g)/len(g), 'max_observed_loss': max(g)}
               for b, g in groups.items()}
    mean = sum(x['mean_loss'] for x in per_bit.values()) / 8
    rng = random.Random(BOOTSTRAP_SEED)
    bootstrap = sorted(sum(sum(rng.choice(g) for _ in g)/len(g) for g in groups.values()) / 8
                       for _ in range(draws))
    return {'equal_bit_mean_nonnegative_AP_loss': mean,
            'exploratory_fault_only_percentile_95_interval': [bootstrap[int(.025*draws)], bootstrap[min(draws-1, int(.975*draws))]],
            'max_observed_loss': max(x for g in groups.values() for x in g),
            'per_bit': per_bit,
            'interval_limitations': 'Fixed images; small-sample percentile approximation; no simultaneous or image uncertainty coverage.'}


def make_locations(inventory):
    """Stable per-layer/bit substreams; all records shuffled before execution."""
    scenarios = []
    for layer in inventory:
        for bit in range(8):
            sub_seed = int(digest(canonical([SEED, layer['name'], bit])), 16)
            indices = random.Random(sub_seed).sample(range(layer['weight_count']), WEIGHTS_PER_BIT)
            for replicate, index in enumerate(indices, 1):
                scenarios.append({'layer': layer['name'], 'bit': bit, 'replicate': replicate,
                                  'flat_index': index})
    random.Random(SEED).shuffle(scenarios)
    for n, scenario in enumerate(scenarios, 1):
        scenario['scenario_id'] = f'S05B-{n:05d}'
    return scenarios


class Campaign:
    def __init__(self, project_root):
        self.root = Path(project_root)
        self.repo = self.root / 'code'
        base_path = self.repo / 'src/baseline05a_runner_v1.py'
        verified(base_path, BASE_SHA)
        spec = importlib.util.spec_from_file_location('_screen_verified_base', base_path)
        self.base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.base)
        self.source_sha = digest(Path(__file__).read_bytes())
        self.plan_path = self.repo / 'experiments/notebook05b/int8_screening_plan_v1.json'
        self.out = self.repo / 'results/notebook05b/int8_screening_v1'
        self.public = self.repo / 'docs/research_reports/notebook05b_screening_v1'
        self._summary_cache = {}

    def _load_sources(self):
        self.published = json.loads(verified(self.repo / 'docs/research_reports/05a_shared_INT8_clean_evaluation_v1.json', BASELINE_REPORT_SHA))
        self.config = self.published['evaluation_record']['payload']['config']
        self.manifest = json.loads(verified(self.repo / 'experiments/notebook05b/screening_image_manifest_v1.json', IMAGE_MANIFEST_SHA))
        self.ids = self.manifest['selection']['image_ids']
        split = json.loads(verified(self.repo / 'experiments/notebook05/coco2017_split_manifest_v1.json', self.config['split_sha256']))
        if (len(self.ids) != 200 or len(set(self.ids)) != 200
                or any(type(i) is not int for i in self.ids)
                or not set(self.ids) <= set(split['image_ids']['development'])
                or set(self.ids) & set(split['image_ids']['held_out'])):
            raise RuntimeError('Invalid frozen development subset.')
        if digest(canonical(self.ids)) != self.manifest['selection']['image_id_list_sha256']:
            raise RuntimeError('Subset ID hash mismatch.')
        self.annotation_path = self.root / 'datasets/coco2017/annotations/instances_val2017.json'
        data = json.loads(verified(self.annotation_path, self.config['annotation_sha256']))
        self.images = {r['id']: r for r in data['images']}
        self.categories = sorted(data['categories'], key=lambda r: r['id'])
        self.checkpoint = self.root / 'models/yolov8s.pt'
        verified(self.checkpoint, CHECKPOINT_SHA)

    def _model(self, device):
        import torch
        from ultralytics import YOLO
        from importlib.metadata import version
        if version('ultralytics') != '8.4.189':
            raise RuntimeError('Requires Ultralytics 8.4.189.')
        self.torch = torch
        torch.manual_seed(20261112)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yolo = YOLO(str(self.checkpoint))
        self.model = yolo.model.cpu().float().eval()
        if len(self.categories) != 80 or any(yolo.names[i] != r['name'] for i, r in enumerate(self.categories)):
            raise RuntimeError('Category mapping mismatch.')
        self.quantized, self.inventory = {}, []
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if not isinstance(module, torch.nn.Conv2d) or name == 'model.22.dfl.conv':
                    continue
                maxima = module.weight.abs().amax(dim=(1,2,3), keepdim=True)
                scales = torch.where(maxima > 0, maxima / 127, torch.ones_like(maxima))
                if not torch.isfinite(scales).all() or not (scales > 0).all():
                    raise RuntimeError('Invalid quantization scales.')
                q = torch.round(module.weight / scales).clamp(-127,127).to(torch.int8)
                module.weight.copy_(q.float() * scales)
                self.quantized[name] = (q.clone(), scales.clone())
                self.inventory.append({'name': name, 'weight_shape': list(q.shape), 'weight_count': q.numel()})
        if len(self.inventory) != 63:
            raise RuntimeError('Expected 63 eligible layers.')
        with torch.inference_mode():
            self.model(torch.zeros(1,3,640,640))
        self.clean_hash = self.config['model_state_sha256']['int8_channel_clean']
        self._assert_state(self.clean_hash)
        self.model = self.model.to(device)
        self._assert_state(self.clean_hash)
        self.modules = dict(self.model.named_modules())

    def _state(self):
        h = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            a = tensor.detach().cpu().contiguous().numpy()
            h.update(canonical([name, str(a.dtype), list(a.shape)]))
            h.update(a.tobytes())
        return h.hexdigest()

    def _assert_state(self, expected):
        if self._state() != expected:
            raise RuntimeError('Model state integrity check failed.')

    def _enrich(self, location):
        q, scales = self.quantized[location['layer']]
        flat = location['flat_index']
        channel = flat // (q.numel() // q.shape[0])
        q_clean = int(q.flatten()[flat])
        q_faulty = bitflip(q_clean, location['bit'])
        scale = scales.flatten()[channel]
        clean = (self.torch.tensor(q_clean, dtype=self.torch.float32) * scale).item()
        faulty = (self.torch.tensor(q_faulty, dtype=self.torch.float32) * scale).item()
        if clean == faulty:
            raise RuntimeError('Represented fault does not change weight.')
        return dict(location, q_clean=q_clean, q_faulty=q_faulty, scale=scale.item(),
                    clean_weight=clean, faulty_weight=faulty)

    @staticmethod
    def _exclusive_json(path, payload):
        content = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8') + b'\n'
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + '.part')
        if path.exists() or partial.exists():
            raise FileExistsError(f'Existing file/partial requires review: {path}')
        with partial.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if partial.read_bytes() != content:
            raise RuntimeError('JSON verification failed.')
        partial.rename(path)
        return digest(content)

    def create_plan(self):
        """Freeze the bounded exploratory design. Does not execute real images."""
        if self.plan_path.exists() or self.plan_path.with_name(self.plan_path.name+'.part').exists():
            raise FileExistsError('Plan already exists; inspect it rather than recreate.')
        self._load_sources()
        self._model('cpu')
        scenarios = [self._enrich(r) for r in make_locations(self.inventory)]
        plan = {
            'version': VERSION, 'status': 'FIXED_BUDGET_EXPLORATORY_SCREENING',
            'source_sha256': self.source_sha, 'baseline_runner_sha256': BASE_SHA,
            'baseline_report_sha256': BASELINE_REPORT_SHA, 'image_manifest_sha256': IMAGE_MANIFEST_SHA,
            'checkpoint_sha256': CHECKPOINT_SHA, 'clean_model_state_sha256': self.clean_hash,
            'image_ids': self.ids, 'image_id_list_sha256': digest(canonical(self.ids)),
            'inventory': self.inventory, 'scenario_count': len(scenarios), 'scenarios': scenarios,
            'sampling': {'seed': SEED, 'weights_per_layer_bit': WEIGHTS_PER_BIT,
                         'weights': 'uniform_without_replacement_per_layer_bit',
                         'bit_allocation': 'equal_all_8_bits', 'between_bit_weight_reuse': True,
                         'execution_order': 'seeded_shuffled_fixed_order', 'power_validated': False},
            'fault': {'operation': 'single_logical_INT8_weight_bit_XOR', 'persistence': 'all_200_images',
                      'clean_range': [-127,127], 'fault_range': [-128,127],
                      'excluded': ['biases','DFL_projection','scales','activations','other_state']},
            'primary': 'equal_bit_mean_of_max_0_clean_AP50_95_minus_faulty_AP50_95',
            'secondary': ['signed_AP_change','AP50','AP75','AR100','per_bit_loss','observed_max_loss'],
            'practical_loss_flags': {'AP50_95_absolute': [0.001,0.01],
                                    'interpretation': 'descriptive engineering thresholds: 0.1 and 1 AP percentage points, not safety standards'},
            'stopping': 'all_5040_scenarios_or_recorded_resource_interruption; never stop for preferred ranking',
            'failure_policy': 'save_failure_and_stop; no silent exclusion, zero substitution or complete campaign claim',
            'uncertainty': {'fault_bootstrap': '2000_stratified_percentile_draws_per_layer',
                            'seed': BOOTSTRAP_SEED, 'conditional_on_fixed_images': True,
                            'image_uncertainty_estimated': False, 'simultaneous_intervals': False},
            'confirmation': 'REQUIRED_SEPARATE_PROTOCOL_fresh_faults_more_development_images_and_low_risk_controls',
            'selection_for_protection': 'NOT_AUTHORIZED_BY_SCREENING_RANKING_ALONE',
            'held_out_use': False, 'settings': self.config['settings'],
            'limitations': ['No rare-fault discovery guarantee or validated ranking precision.',
                            'Fixed 200 images cover 76 classes; four classes lack valid non-crowd boxes.',
                            'Equal layer/bit sampling is not uniform whole-network weight sampling.',
                            'Not hardware fault probability, native INT8 execution or protection effectiveness.',
                            'Engineering sign-bit result and historical INT4 results observed before this protocol.'],
        }
        plan_sha = self._exclusive_json(self.plan_path, plan)
        self._exclusive_json(self.plan_path.with_suffix('.sha256.json'), {'sha256':plan_sha})
        print('Saved plan:', self.plan_path)
        print('Plan SHA256:', plan_sha)
        print('Scenarios:', len(scenarios), '| no screening inference performed')
        return plan

    def _load_plan(self):
        self.plan_bytes = self.plan_path.read_bytes()
        recorded_sha = json.loads(self.plan_path.with_suffix('.sha256.json').read_bytes())['sha256']
        if digest(self.plan_bytes) != recorded_sha:
            raise RuntimeError('Frozen plan checksum mismatch.')
        self.plan = json.loads(self.plan_bytes)
        if (self.plan['source_sha256'] != self.source_sha or self.plan['image_manifest_sha256'] != IMAGE_MANIFEST_SHA
                or self.plan['scenario_count'] != 5040 or len(self.plan['scenarios']) != 5040
                or self.plan['image_ids'] != self.ids):
            raise RuntimeError('Plan/source identity or size mismatch.')
        expected = [self._enrich(r) for r in make_locations(self.inventory)]
        if self.plan['scenarios'] != expected or self.plan['inventory'] != self.inventory:
            raise RuntimeError('Plan scenarios do not reproduce from the declared sampling.')
        self.plan_sha = digest(self.plan_bytes)

    def _cache(self):
        self.cache, self.image_hashes = {}, {}
        for image_id in self.ids:
            filename = self.images[image_id]['file_name']
            if Path(filename).name != filename:
                raise RuntimeError('Unsafe image filename.')
            content = (self.root/'datasets/coco2017/images/val2017'/filename).read_bytes()
            self.cache[image_id] = content
            self.image_hashes[str(image_id)] = digest(content)

    def _predict(self):
        import cv2
        import numpy as np
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils.nms import non_max_suppression
        from ultralytics.utils.ops import scale_boxes
        torch = self.torch
        transform = LetterBox(new_shape=(640,640), auto=False, scale_fill=False, scaleup=True, stride=32)
        predictions, per_image = [], []
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for image_id in self.ids:
                image = cv2.imdecode(np.frombuffer(self.cache[image_id],dtype=np.uint8),cv2.IMREAD_COLOR)
                record = self.images[image_id]
                if image is None or image.shape[:2] != (record['height'],record['width']):
                    raise RuntimeError('Image decode/dimensions invalid.')
                padded = transform(image=image)
                array = np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1))
                tensor = (torch.from_numpy(array).unsqueeze(0).float()/255).to('cuda:0')
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
        torch.cuda.synchronize()
        return {'image_ids':self.ids,'predictions':predictions,'per_image':per_image,
                'pipeline_seconds':time.perf_counter()-started}

    def _validate_predictions(self, payload):
        if payload['image_ids'] != self.ids or [r['image_id'] for r in payload['per_image']] != self.ids:
            raise RuntimeError('Prediction image execution record mismatch.')
        counts = {i:0 for i in self.ids}
        for record in payload['predictions']:
            if record['image_id'] not in counts or record['category_id'] not in self.category_ids:
                raise RuntimeError('Prediction reference mismatch.')
            values = record['bbox'] + [record['score']]
            if len(record['bbox']) != 4 or not all(math.isfinite(x) for x in values):
                raise RuntimeError('Invalid prediction values.')
            if record['bbox'][2]<=0 or record['bbox'][3]<=0 or not 0<=record['score']<=1:
                raise RuntimeError('Invalid prediction geometry or confidence.')
            counts[record['image_id']] += 1
        if any(r['count'] != counts[r['image_id']] or r['count']+r['degenerate']>300 for r in payload['per_image']):
            raise RuntimeError('Per-image prediction counts disagree.')

    def _evaluate(self, predictions):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import io
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            if predictions:
                dt = self.gt.loadRes(copy.deepcopy(predictions))
            else:
                dt = COCO()
                dt.dataset = {'images': self.gt.dataset['images'], 'categories':self.gt.dataset['categories'],'annotations':[]}
                dt.createIndex()
            ev = COCOeval(self.gt,dt,'bbox')
            ev.params.imgIds = self.ids
            ev.params.maxDets = [1,10,100]
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        return dict(zip(METRICS,map(float,ev.stats))), time.perf_counter()-started

    def _record(self, kind, **extra):
        return dict(kind=kind, session_fingerprint=self.fingerprint, plan_sha256=self.plan_sha, **extra)

    def _read(self, path, kind):
        r = self.base.read_record(path)
        if r['kind'] != kind or r['session_fingerprint'] != self.fingerprint or r['plan_sha256'] != self.plan_sha:
            raise RuntimeError('Stored result provenance mismatch.')
        return r

    def _prediction(self, path, scenario, expected_state):
        if path.exists():
            record = self._read(path,'predictions')
            if record['scenario'] != scenario or record['model_state_sha256'] != expected_state:
                raise RuntimeError('Prediction scenario/state mismatch.')
            payload = record['data']
        else:
            payload = self._predict()
            self._assert_state(expected_state)
            self.base.write_record(path,self._record('predictions',scenario=scenario,model_state_sha256=expected_state,data=payload))
        self._validate_predictions(payload)
        return payload

    def run(self, max_new_scenarios=None, max_wall_seconds=14400):
        """Single writer; resumable after a clean resource interruption.

        Stops after a scenario when the wall budget expires. Actual failures
        save a failure record and block further execution until reviewed.
        """
        if max_new_scenarios is not None and (type(max_new_scenarios) is not int or max_new_scenarios<0):
            raise ValueError('max_new_scenarios must be nonnegative or None.')
        if not math.isfinite(max_wall_seconds) or max_wall_seconds<=0:
            raise ValueError('Positive wall budget required.')
        import fcntl
        self.out.mkdir(parents=True,exist_ok=True)
        with (self.out/'writer.lock').open('a') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another campaign writer is active.') from None
            return self._run_locked(max_new_scenarios,max_wall_seconds)

    def _run_locked(self,max_new,max_seconds):
        import platform
        import io
        from importlib.metadata import version
        import torch
        from pycocotools.coco import COCO
        started = time.perf_counter()
        if list(self.out.rglob('*.part')) or list(self.out.glob('*_failure.json.gz')):
            raise RuntimeError('Partial/failure records require review; no automatic deletion.')
        if not torch.cuda.is_available():
            raise RuntimeError('This version requires CUDA; no silent CPU fallback.')
        self._load_sources()
        self._model('cuda:0')
        self._load_plan()
        self._cache()
        self.category_ids = {r['id'] for r in self.categories}
        environment = {'python':platform.python_version(), 'packages':{p:version(p) for p in
                       ('torch','torchvision','ultralytics','numpy','pycocotools','opencv-python','Pillow')},
                       'gpu':torch.cuda.get_device_name(0),'torch_threads':torch.get_num_threads(),
                       'torch_cuda_build':torch.version.cuda,'tf32':False,'deterministic_algorithms':True}
        session = {'plan_sha256':self.plan_sha,'source_sha256':self.source_sha,'environment':environment,
                   'image_sha256':self.image_hashes,'clean_state':self.clean_hash}
        self.fingerprint = digest(canonical(session))
        session_path = self.out/'session.json.gz'
        if session_path.exists():
            previous = self.base.read_record(session_path)
            if previous != session:
                raise RuntimeError('Resume environment/source/image identity changed; review before continuing.')
        else:
            self.base.write_record(session_path,session)
        with torch.inference_mode():
            dummy = torch.zeros(1,3,640,640,device='cuda:0')
            for _ in range(5):
                self.model(dummy)
            torch.cuda.synchronize()
        self._assert_state(self.clean_hash)
        with contextlib.redirect_stdout(io.StringIO()):
            self.gt = COCO(str(self.annotation_path))
        clean_path = self.out/'clean_predictions.json.gz'
        clean = self._prediction(clean_path,None,self.clean_hash)
        clean_pred_sha = digest(clean_path.read_bytes())
        baseline_path = self.out/'clean_evaluation.json.gz'
        if baseline_path.exists():
            baseline = self._read(baseline_path,'baseline')
            if baseline['prediction_sha256'] != clean_pred_sha:
                raise RuntimeError('Clean prediction/evaluation identity mismatch.')
        else:
            metrics, seconds = self._evaluate(clean['predictions'])
            baseline = self._record('baseline',prediction_sha256=clean_pred_sha,metrics=metrics,evaluation_seconds=seconds)
            self.base.write_record(baseline_path,baseline)
        self.clean_metrics = baseline['metrics']
        finished, new_count = [], 0
        for scenario in self.plan['scenarios']:
            sid = scenario['scenario_id']
            prediction_path = self.out/f'{sid}_predictions.json.gz'
            evaluation_path = self.out/f'{sid}_evaluation.json.gz'
            if evaluation_path.exists():
                record = self._read(evaluation_path,'scenario_evaluation')
                if record['scenario'] != scenario or record['prediction_sha256'] != digest(prediction_path.read_bytes()):
                    raise RuntimeError('Completed scenario artifact identity mismatch.')
                prediction = self._read(prediction_path,'predictions')
                if prediction['scenario'] != scenario:
                    raise RuntimeError('Completed prediction scenario mismatch.')
                self._validate_predictions(prediction['data'])
                finished.append(record)
                continue
            if (max_new is not None and new_count>=max_new) or time.perf_counter()-started>=max_seconds:
                break
            scenario_started = time.perf_counter()
            self._assert_state(self.clean_hash)
            target = self.modules[scenario['layer']].weight
            flat = scenario['flat_index']
            original = target.detach().clone()
            if target.flatten()[flat].item() != scenario['clean_weight']:
                raise RuntimeError('Planned clean weight differs from reconstructed baseline.')
            try:
                try:
                    with torch.no_grad():
                        target.flatten()[flat] = scenario['faulty_weight']
                    changed = (target != original).flatten().nonzero().flatten().tolist()
                    if changed != [flat]:
                        raise RuntimeError('Single-weight isolation failed.')
                    # Prove all other model state remains clean by masking this one scalar.
                    with torch.no_grad():
                        target.flatten()[flat] = original.flatten()[flat]
                    self._assert_state(self.clean_hash)
                    with torch.no_grad():
                        target.flatten()[flat] = scenario['faulty_weight']
                    faulty_hash = self._state()
                    payload = self._prediction(prediction_path,scenario,faulty_hash)
                    self._assert_state(faulty_hash)
                finally:
                    with torch.no_grad():
                        target.copy_(original)
                    self._assert_state(self.clean_hash)
                metrics, eval_seconds = self._evaluate(payload['predictions'])
                if any(not math.isfinite(v) for v in metrics.values()) or metrics['AP50_95']<0:
                    raise RuntimeError('Unavailable/invalid primary scenario metrics.')
                delta = metrics['AP50_95']-self.clean_metrics['AP50_95']
                record = self._record('scenario_evaluation',scenario=scenario,metrics=metrics,
                           clean_metrics=self.clean_metrics,AP50_95_change=delta,loss=max(0.,-delta),
                           prediction_sha256=digest(prediction_path.read_bytes()),
                           single_weight_isolation=True,exact_restoration=True,
                           timing_seconds={'pipeline':payload['pipeline_seconds'],'cocoeval':eval_seconds,
                                           'scenario_total_before_report':time.perf_counter()-scenario_started})
                self.base.write_record(evaluation_path,record)
            except BaseException as exc:
                # Preserve failures, including an interrupted injection, without reporting success.
                failure_path = self.out/f'{sid}_failure.json.gz'
                if not failure_path.exists() and not failure_path.with_name(failure_path.name+'.part').exists():
                    self.base.write_record(failure_path,self._record('failure',scenario=scenario,
                        exception_type=type(exc).__name__,message=str(exc),clean_state_restored=(self._state()==self.clean_hash)))
                raise
            finished.append(record)
            new_count += 1
            print(f'{len(finished)}/5040 | {sid} | AP change={delta:+.8f} | saved',flush=True)
            if len(finished) % 50 == 0:
                self._write_progress(finished)
        summary = self._write_progress(finished)
        print('Completed:',len(finished),'/5040 | new:',new_count)
        print('Invocation wall seconds:', round(time.perf_counter()-started,3))
        print('Summary:',self.public/f'progress_{len(finished):05d}.json')
        print('No held-out evaluation. No protection or final ranking claim.')
        return summary

    def _write_progress(self,finished):
        groups = {r['name']:[] for r in self.plan['inventory']}
        for r in finished:
            groups[r['scenario']['layer']].append({'bit':r['scenario']['bit'],'loss':r['loss'],
                                                   'delta':r['AP50_95_change']})
        layers = []
        for name, rows in groups.items():
            item = {'layer':name,'completed':len(rows),'planned':80,'complete':len(rows)==80}
            if len(rows)==80:
                if name not in self._summary_cache:
                    self._summary_cache[name] = stratified_loss_summary(rows)
                item.update(self._summary_cache[name])
                item['mean_signed_AP_change'] = sum(r['delta'] for r in rows)/len(rows)
                item['counts_loss_at_least'] = {str(t):sum(r['loss']>=t for r in rows) for t in (.001,.01)}
            layers.append(item)
        complete = len(finished)==5040
        ranking = sorted((r for r in layers if r['complete']),
                         key=lambda r:(-r['equal_bit_mean_nonnegative_AP_loss'],r['layer'])) if complete else []
        payload = self._record('progress',status='SCREENING_COMPLETE_CONFIRMATION_REQUIRED' if complete else 'PARTIAL_SCREENING_NO_RANKING',
            completed_scenarios=len(finished),planned_scenarios=5040,layers=layers,
            exploratory_ranking=ranking,clean_metrics=self.clean_metrics,
            limitations=self.plan['limitations']+[self.plan['confirmation'],
                'Fault bootstrap intervals are exploratory, conditional on these images and unadjusted for 63 comparisons.',
                'No image-sampling uncertainty, verified rank stability or rare-tail precision is established.'])
        path = self.public/f'progress_{len(finished):05d}.json'
        if path.exists():
            if json.loads(path.read_bytes()) != payload:
                raise RuntimeError('Existing progress snapshot differs.')
        else:
            self._exclusive_json(path,payload)
        if complete:
            self._publish_final(payload)
        return payload

    def _publish_final(self, payload):
        import csv
        import io
        rows = payload['exploratory_ranking']
        text = ['# Logical INT8 weight-fault screening', '',
                'Status: complete exploratory screening; independent confirmation required.',
                f'Completed scenarios: {payload["completed_scenarios"]}',
                f'Plan SHA256: {self.plan_sha}',
                f'Session fingerprint: {self.fingerprint}', '',
                'The intervals below concern fault sampling on 200 fixed development images.',
                'They do not include image uncertainty or simultaneous-comparison correction.',
                'Equal bit allocation is an experimental distribution, not a hardware fault model.',
                '', '| Rank | Layer | Mean nonnegative AP loss | Exploratory interval | Observed max loss |',
                '| ---: | --- | ---: | --- | ---: |']
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(['rank','layer','mean_nonnegative_AP_loss','interval_low','interval_high','observed_max_loss','mean_signed_AP_change'])
        for rank, row in enumerate(rows, 1):
            lo, hi = row['exploratory_fault_only_percentile_95_interval']
            mean = row['equal_bit_mean_nonnegative_AP_loss']
            maximum = row['max_observed_loss']
            text.append(f'| {rank} | {row["layer"]} | {mean:.8f} | [{lo:.8f}, {hi:.8f}] | {maximum:.8f} |')
            writer.writerow([rank,row['layer'],mean,lo,hi,maximum,row['mean_signed_AP_change']])
        text += ['', '## Limits', '', *['- '+v for v in payload['limitations']],
                 '', 'No layer is declared safe or definitively sensitive by this screening alone.',
                 'The observed maximum is not the population worst case.', '']
        for name, content in [('screening_report_v1.md','\n'.join(text).encode('utf-8')),
                              ('screening_ranking_v1.csv',stream.getvalue().encode('utf-8'))]:
            path = self.public/name
            if path.exists():
                if path.read_bytes() != content:
                    raise RuntimeError('Existing final report differs.')
                continue
            partial = path.with_name(path.name+'.part')
            with partial.open('xb') as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if partial.read_bytes()!=content:
                raise RuntimeError('Final report verification failed.')
            partial.rename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('plan','run'))
    parser.add_argument('--project-root',required=True)
    parser.add_argument('--max-new',type=int,default=None)
    parser.add_argument('--max-seconds',type=float,default=14400)
    args = parser.parse_args()
    campaign = Campaign(args.project_root)
    if args.command=='plan':
        campaign.create_plan()
    else:
        campaign.run(max_new_scenarios=args.max_new,max_wall_seconds=args.max_seconds)


if __name__=='__main__':
    main()
