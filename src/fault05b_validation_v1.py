"""One engineering validation fault, not a screening or ranking campaign.

Requires the verified baseline05a_runner_v1 module for atomic record storage.
Imports torch/Ultralytics only when run() is called. Never edits the checkpoint.
"""
import hashlib
import json
import time
from pathlib import Path

BASE_SHA = 'f1069948c16a8899d422930b6dfb2c4ce0f900e5e27e358e0761d588b11492b3'
REPORT_SHA = '47861220401cc36ea2c6c19f427d351dcbb5ed39a9067fa79e0abbb175f889d7'
MANIFEST_SHA = 'f2130cb0b0d505d641c8604dcb075cdcd0830092e5d8b31f594a45dcf1074aeb'
CHECKPOINT_SHA = '1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a'
METRICS = ('AP50_95', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
           'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large')


def flip_int8(q, bit):
    if type(q) is not int or not -128 <= q <= 127:
        raise ValueError('Expected a signed INT8 integer.')
    if type(bit) is not int or not 0 <= bit <= 7:
        raise ValueError('Expected a bit index in [0, 7].')
    unsigned = (q & 255) ^ (1 << bit)
    return unsigned - 256 if unsigned >= 128 else unsigned


def verify_subset(ids, development, held_out):
    if (len(ids) != 200 or len(set(ids)) != 200
            or any(type(i) is not int for i in ids)
            or not set(ids) <= set(development) or set(ids) & set(held_out)):
        raise ValueError('Invalid fixed development subset.')


def verified_bytes(path, expected):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError(f'SHA256 mismatch: {Path(path).name}')
    return content


def run(project_root):
    """Run clean/faulted pairs on 200 images, restore in finally, save results.

    Fixed engineering scenario: model.0.conv, flat index 0, sign bit 7.
    Results must never be treated as randomly sampled screening evidence.
    No reuse of output destinations; partial outputs require explicit review.
    """
    import importlib.util
    import platform
    from importlib.metadata import version
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    started_total = time.perf_counter()
    root = Path(project_root)
    repo = root / 'code'
    out = repo / 'results/notebook05b/engineering_fault_validation_v1'
    if out.exists():
        raise FileExistsError('Validation directory exists; inspect before rerunning.')
    if version('ultralytics') != '8.4.189' or not torch.cuda.is_available():
        raise RuntimeError('Requires Ultralytics 8.4.189 and a working CUDA device.')
    base_path = repo / 'src/baseline05a_runner_v1.py'
    verified_bytes(base_path, BASE_SHA)
    spec = importlib.util.spec_from_file_location('_verified_base05a', base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    report_bytes = verified_bytes(repo / 'docs/research_reports/05a_shared_INT8_clean_evaluation_v1.json', REPORT_SHA)
    config = json.loads(report_bytes)['evaluation_record']['payload']['config']
    manifest_bytes = verified_bytes(repo / 'experiments/notebook05b/screening_image_manifest_v1.json', MANIFEST_SHA)
    manifest = json.loads(manifest_bytes)
    split_bytes = verified_bytes(repo / 'experiments/notebook05/coco2017_split_manifest_v1.json', config['split_sha256'])
    split = json.loads(split_bytes)
    ids = manifest['selection']['image_ids']
    verify_subset(ids, split['image_ids']['development'], split['image_ids']['held_out'])
    if base.digest(base.canonical(ids)) != manifest['selection']['image_id_list_sha256']:
        raise RuntimeError('Subset ID checksum mismatch.')
    annotation_path = root / 'datasets/coco2017/annotations/instances_val2017.json'
    annotation_bytes = verified_bytes(annotation_path, config['annotation_sha256'])
    coco_data = json.loads(annotation_bytes)
    images = {r['id']: r for r in coco_data['images']}
    categories = sorted(coco_data['categories'], key=lambda r: r['id'])
    checkpoint = root / 'models/yolov8s.pt'
    verified_bytes(checkpoint, CHECKPOINT_SHA)
    torch.manual_seed(20261112)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yolo = YOLO(str(checkpoint))
    model = yolo.model.cpu().float().eval()
    if len(categories) != 80 or any(yolo.names[i] != r['name'] for i, r in enumerate(categories)):
        raise RuntimeError('Model/COCO category mapping mismatch.')

    def state_hash():
        h = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            array = tensor.detach().cpu().contiguous().numpy()
            h.update(base.canonical([name, str(array.dtype), list(array.shape)]))
            h.update(array.tobytes())
        return h.hexdigest()

    target_name = 'model.0.conv'
    bit = 7
    q_clean = q_faulty = None
    with torch.no_grad():
        count = 0
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Conv2d) or name == 'model.22.dfl.conv':
                continue
            maxima = module.weight.abs().amax(dim=(1, 2, 3), keepdim=True)
            scales = torch.where(maxima > 0, maxima / 127, torch.ones_like(maxima))
            q = torch.round(module.weight / scales).clamp(-127, 127).to(torch.int8)
            if name == target_name:
                q_clean = int(q.flatten()[0].item())
                q_faulty = flip_int8(q_clean, bit)
                scale = scales.flatten()[0].clone()
                faulty_value = (torch.tensor(q_faulty, dtype=torch.float32) * scale).item()
                scale_value = scale.item()
            module.weight.copy_(q.float() * scales)
            count += 1
        if count != 63 or q_clean is None:
            raise RuntimeError('Quantization scope mismatch.')
    with torch.inference_mode():
        model(torch.zeros(1, 3, 640, 640))
    clean_hash = config['model_state_sha256']['int8_channel_clean']
    if state_hash() != clean_hash:
        raise RuntimeError('Shared clean state differs from 05a.')
    model = model.to('cuda:0')
    target = dict(model.named_modules())[target_name].weight
    original_target = target.detach().clone()
    clean_value = target.flatten()[0].item()
    if clean_value == faulty_value:
        raise RuntimeError('Fault does not change the represented weight.')
    if state_hash() != clean_hash:
        raise RuntimeError('GPU transfer changed state.')
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 640, 640, device='cuda:0')
        for _ in range(5):
            model(dummy)
        torch.cuda.synchronize()
    if state_hash() != clean_hash:
        raise RuntimeError('Warmup changed state.')
    cache_start = time.perf_counter()
    cache, image_hashes = {}, {}
    for image_id in ids:
        filename = images[image_id]['file_name']
        if Path(filename).name != filename:
            raise RuntimeError('Unsafe image filename.')
        data = (root / 'datasets/coco2017/images/val2017' / filename).read_bytes()
        cache[image_id] = data
        image_hashes[str(image_id)] = base.digest(data)
    cache_seconds = time.perf_counter() - cache_start
    transform = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True, stride=32)

    def predict(label):
        predictions, rows = [], []
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for image_id in ids:
                image = cv2.imdecode(np.frombuffer(cache[image_id], dtype=np.uint8), cv2.IMREAD_COLOR)
                record = images[image_id]
                if image is None or image.shape[:2] != (record['height'], record['width']):
                    raise RuntimeError(f'Image decode/dimension failure: {image_id}')
                padded = transform(image=image)
                array = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
                tensor = (torch.from_numpy(array).unsqueeze(0).float() / 255).to('cuda:0')
                result = model(tensor)
                raw = result[0] if isinstance(result, (tuple, list)) else result
                raw = raw.detach().cpu()
                if tuple(raw.shape) != (1, 84, 8400) or not torch.isfinite(raw).all():
                    raise RuntimeError('Invalid raw predictions; scenario incomplete.')
                boxes = non_max_suppression(raw.clone(), conf_thres=0.001, iou_thres=0.7, nc=80, max_det=300)[0].clone()
                boxes[:, :4] = scale_boxes(tensor.shape[2:], boxes[:, :4], image.shape[:2])
                if not torch.isfinite(boxes).all():
                    raise RuntimeError('Invalid post-NMS predictions; scenario incomplete.')
                retained, degenerate = 0, 0
                for box in boxes:
                    x1, y1, x2, y2, score, cls = box.tolist()
                    if x2 <= x1 or y2 <= y1:
                        degenerate += 1
                        continue
                    if not (cls == int(cls) and 0 <= cls < 80 and 0 <= score <= 1):
                        raise RuntimeError('Invalid detection class/score.')
                    if not (0 <= x1 < x2 <= record['width'] + 1e-4 and 0 <= y1 < y2 <= record['height'] + 1e-4):
                        raise RuntimeError('Detection outside image bounds.')
                    predictions.append({'image_id': image_id, 'category_id': categories[int(cls)]['id'],
                                        'bbox': [x1, y1, x2-x1, y2-y1], 'score': score})
                    retained += 1
                rows.append({'image_id': image_id, 'detections': retained, 'degenerate_boxes': degenerate})
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        print(f'{label}: {len(ids)} images | {len(predictions)} predictions | {elapsed:.3f} pipeline seconds')
        return {'image_ids': ids, 'predictions': predictions, 'per_image': rows, 'pipeline_wall_seconds': elapsed}

    out.mkdir(parents=True, exist_ok=False)
    clean = predict('INT8 clean')
    if state_hash() != clean_hash:
        raise RuntimeError('Clean inference changed model state.')
    base.write_record(out / 'clean_predictions.json.gz', clean)
    injection_started = time.perf_counter()
    try:
        with torch.no_grad():
            target.flatten()[0] = faulty_value
        differences = (target != original_target).nonzero()
        if differences.shape[0] != 1 or differences[0].tolist() != [0, 0, 0, 0]:
            raise RuntimeError('Single-weight isolation failed.')
        with torch.no_grad():
            probe = target.flatten()[0].clone()
            target.flatten()[0] = original_target.flatten()[0]
        try:
            if state_hash() != clean_hash:
                raise RuntimeError('Non-target model state changed.')
        finally:
            with torch.no_grad():
                target.flatten()[0] = probe
        fault_hash = state_hash()
        injection_seconds = time.perf_counter() - injection_started
        faulty = predict('INT8 sign-bit engineering fault')
        if state_hash() != fault_hash:
            raise RuntimeError('Faulted inference changed model state.')
        base.write_record(out / 'faulty_predictions.json.gz', faulty)
    finally:
        with torch.no_grad():
            target.copy_(original_target)
        if state_hash() != clean_hash:
            raise RuntimeError('CRITICAL: exact clean restoration failed.')
        print('Exact shared clean restoration: PASS')
    evaluation_started = time.perf_counter()
    gt = COCO(str(annotation_path))
    metrics = {}
    for label, payload in (('clean', clean), ('faulty', faulty)):
        if payload['predictions']:
            dt = gt.loadRes(payload['predictions'])
        else:
            dt = COCO()
            dt.dataset = {'images': gt.dataset['images'], 'categories': gt.dataset['categories'], 'annotations': []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, 'bbox')
        evaluator.params.imgIds = ids
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        metrics[label] = dict(zip(METRICS, map(float, evaluator.stats)))
    evaluation_seconds = time.perf_counter() - evaluation_started
    if state_hash() != clean_hash:
        raise RuntimeError('Final model state check failed.')
    artifacts = {p.name: base.digest(p.read_bytes()) for p in sorted(out.glob('*_predictions.json.gz'))}
    result = {
        'status': 'ENGINEERING_VALIDATION_COMPLETE_NOT_SCREENING_EVIDENCE',
        'scenario': {'layer': target_name, 'weight_index': [0, 0, 0, 0], 'bit': bit,
                     'q_clean': q_clean, 'q_faulty': q_faulty, 'scale': scale_value,
                     'clean_weight': clean_value, 'faulty_weight': faulty_value,
                     'selection': 'fixed_convenience_scenario_not_random', 'persistence': 'all_200_images'},
        'source_sha256': {'runner': base.digest(Path(__file__).read_bytes()), 'baseline_runner': BASE_SHA,
                          'image_manifest': MANIFEST_SHA, 'baseline_report': REPORT_SHA, 'checkpoint': CHECKPOINT_SHA},
        'environment': {'python': platform.python_version(), 'packages': {p: version(p) for p in
                        ('torch', 'torchvision', 'ultralytics', 'numpy', 'pycocotools', 'opencv-python')},
                        'gpu': torch.cuda.get_device_name(0), 'torch_threads': torch.get_num_threads(),
                        'deterministic_algorithms': True, 'cudnn_benchmark': False, 'tf32': False},
        'settings': config['settings'], 'inference_device': 'cuda:0', 'nms_device': 'cpu',
        'image_ids': ids, 'image_sha256': image_hashes, 'prediction_artifact_sha256': artifacts,
        'clean_model_state_sha256': clean_hash, 'fault_model_state_sha256': fault_hash,
        'exact_restoration_verified': True, 'single_weight_isolation_verified': True,
        'metrics': metrics,
        'faulty_minus_clean': {k: metrics['faulty'][k]-metrics['clean'][k] for k in METRICS},
        'nonnegative_AP50_95_loss': max(0., metrics['clean']['AP50_95']-metrics['faulty']['AP50_95']),
        'timing_seconds': {'drive_read_and_hash': cache_seconds, 'clean_pipeline': clean['pipeline_wall_seconds'],
                           'faulty_pipeline': faulty['pipeline_wall_seconds'], 'injection_and_checks': injection_seconds,
                           'paired_cocoeval': evaluation_seconds, 'total_before_final_report_write': time.perf_counter()-started_total},
        'limitations': ['One convenience fault; not a representative runtime or damage sample.',
                        'Subset AP is not directly comparable with full development AP.',
                        'No held-out evaluation or hardware fault-rate inference.',
                        'Logical INT8 weights with FP32 computation; scales and activations are outside fault scope.',
                        'Failures abort without a completed report; partial artifacts must be reviewed.'],
    }
    base.write_record(out / 'validation_report.json.gz', result)
    print('Saved:', out / 'validation_report.json.gz')
    print('Report SHA256:', base.digest((out / 'validation_report.json.gz').read_bytes()))
    print('No campaign started. No commit or push.')
    return result
