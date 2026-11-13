"""Fixed-protocol masked-bit feasibility comparison, not a fault repair system.

No Torch, CUDA, model inference, images, network requests or Git operations.
No automatic hyperparameter selection or use of test labels during fitting.
"""
import contextlib
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path

PROTOCOL_SHA = 'd851b7a91569ffa155d3e85ffd7afa1fb80e0df851857979cde7a51e392bceec'
DATASET_SHA = '6e2a4c366321b392f59a330b2cc2810a965897feb9d72e34e98b492666a1df0e'
MANIFEST_SHA = '515632610a2503bce69fd99b21391d37968d217457c54f39d56f73143ea307fe'
METHODS = ('A','B','C','D')
PARTS = ('train','calibration','test')
CLIP = 1e-6
BOOTSTRAP_SEED = 20261117
BOOTSTRAP_DRAWS = 2000


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',',':')).encode('utf-8')


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def verified(path, expected):
    content = Path(path).read_bytes()
    if sha256(content) != expected:
        raise RuntimeError(f'Source SHA256 mismatch: {Path(path).name}')
    return content


def write_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    partial = path.with_name(path.name+'.part')
    if path.exists() or partial.exists():
        raise FileExistsError(f'Existing file/partial requires review: {path.name}')
    with partial.open('xb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    if partial.read_bytes() != content:
        raise RuntimeError('Write verification failed.')
    if path.exists():
        raise FileExistsError('Concurrent publication is unsupported.')
    partial.rename(path)
    return sha256(content)


def write_json(path, payload):
    envelope = {'payload_sha256':sha256(canonical(payload)),'payload':payload}
    return write_bytes(path,json.dumps(envelope,indent=2,ensure_ascii=False,allow_nan=False).encode('utf-8')+b'\n')


def read_json(path):
    envelope = json.loads(Path(path).read_bytes())
    if sha256(canonical(envelope['payload'])) != envelope['payload_sha256']:
        raise RuntimeError('Record payload checksum mismatch.')
    return envelope['payload']


def masked_key(bits, bit):
    import numpy as np
    if type(bit) is not int or not 0<=bit<8:
        raise ValueError('Invalid target bit.')
    bits = np.asarray(bits)
    if bits.ndim!=2 or bits.shape[1]!=8 or not np.isin(bits,[0,1]).all():
        raise ValueError('Expected N by 8 binary labels.')
    columns = np.arange(8)!=bit
    return (bits[:,columns].astype(np.int16)*(1<<np.arange(8)[columns])).sum(axis=1)


def features(partition, bit, with_context, n_layers=63):
    """Explicit allowlist: no channel, address or target-bit features."""
    import numpy as np
    masked_key(partition['bits'],bit)
    own = partition['bits'][:,np.arange(8)!=bit].astype(np.float32)
    layer = np.asarray(partition['layer'])
    if layer.ndim!=1 or len(layer)!=len(own) or not np.issubdtype(layer.dtype,np.integer):
        raise ValueError('Invalid layer metadata.')
    if np.any(layer<0) or np.any(layer>=n_layers):
        raise ValueError('Layer outside allowed range.')
    one_hot = np.eye(n_layers,dtype=np.float32)[layer]
    blocks = [own,one_hot]
    if with_context:
        context = np.asarray(partition['context'],dtype=np.float32)
        if context.shape!=(len(own),13) or not np.isfinite(context).all():
            raise ValueError('Invalid context features.')
        blocks.append(context)
    return np.ascontiguousarray(np.concatenate(blocks,axis=1))


def frequency_fit(train, bit, n_layers=63):
    import numpy as np
    layer = train['layer']
    labels = train['bits'][:,bit]
    keys = masked_key(train['bits'],bit)
    total_a = np.bincount(layer,minlength=n_layers)
    ones_a = np.bincount(layer,weights=labels,minlength=n_layers)
    total_b = np.zeros((n_layers,256),dtype=np.int64)
    ones_b = np.zeros((n_layers,256),dtype=np.int64)
    np.add.at(total_b,(layer,keys),1)
    np.add.at(ones_b,(layer,keys),labels)
    return {'A':(ones_a+1)/(total_a+2),'B':(ones_b+1)/(total_b+2),
            'conditional_training_counts':total_b}


def frequency_predict(model, partition, bit):
    keys = masked_key(partition['bits'],bit)
    layer = partition['layer']
    return {'A':model['A'][layer],'B':model['B'][layer,keys]}


def bernoulli_losses(y, probabilities):
    import numpy as np
    y = np.asarray(y)
    p = np.asarray(probabilities,dtype=np.float64)
    if y.shape!=p.shape or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):
        raise ValueError('Invalid labels or probabilities.')
    p = np.clip(p,CLIP,1-CLIP)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def metrics(y, probabilities):
    import numpy as np
    y = np.asarray(y)
    p = np.asarray(probabilities,dtype=np.float64)
    loss = bernoulli_losses(y,p)
    prediction = p>=.5
    has_zero, has_one = bool(np.any(y==0)), bool(np.any(y==1))
    balanced = (float(np.mean(prediction[y==1]))+float(np.mean(~prediction[y==0])))/2 if has_zero and has_one else None
    return {'n':int(len(y)),'ones':int(np.sum(y)),'one_prevalence':float(np.mean(y)),
            'log_loss':float(np.mean(loss)),'brier_score':float(np.mean((p-y)**2)),
            'accuracy':float(np.mean(prediction==y)),'balanced_accuracy':balanced,
            'constant_label_group':not(has_zero and has_one)}


def paired_channel_bootstrap(delta, layers, channels, draws=BOOTSTRAP_DRAWS):
    """Stratify by layer, resample test output channels, keep their rows together.

    Fixed fitted predictors. No repeated training or coverage guarantee.
    Bits share bootstrap draws so their paired dependence is preserved.
    """
    import numpy as np
    delta = np.asarray(delta,dtype=np.float64)
    if delta.ndim==1:
        delta = delta[:,None]
    layers, channels = np.asarray(layers),np.asarray(channels)
    if len(delta)==0 or len(layers)!=len(delta) or len(channels)!=len(delta) or not np.isfinite(delta).all():
        raise ValueError('Invalid paired bootstrap inputs.')
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    numerators = np.zeros((draws,delta.shape[1]))
    denominators = np.zeros(draws)
    cluster_count = 0
    for layer in np.unique(layers):
        selected = layers==layer
        unique, inverse = np.unique(channels[selected],return_inverse=True)
        k = len(unique)
        cluster_count += k
        totals = np.zeros((k,delta.shape[1]))
        np.add.at(totals,inverse,delta[selected])
        counts = np.bincount(inverse,minlength=k)
        sample = rng.integers(0,k,size=(draws,k))
        numerators += totals[sample].sum(axis=1)
        denominators += counts[sample].sum(axis=1)
    means = numerators/denominators[:,None]
    interval = np.quantile(means,[.025,.975],axis=0)
    macro = means.mean(axis=1)
    return {'point_difference_by_bit':delta.mean(axis=0).tolist(),
            'exploratory_percentile_95_interval_by_bit':interval.T.tolist(),
            'mean_over_bits':float(delta.mean()),
            'exploratory_percentile_95_interval_mean_over_bits':np.quantile(macro,[.025,.975]).tolist(),
            'test_output_channel_clusters':cluster_count,'draws':draws,'seed':BOOTSTRAP_SEED,
            'limitations':['Fixed predictors and checkpoint; not training uncertainty.',
                           'Approximate cluster bootstrap; unadjusted multiple comparisons.',
                           'No fault detection, repair or detection-AP guarantee.']}


def validate_dataset(data, manifest):
    import numpy as np
    if set(data)!=set(PARTS):
        raise ValueError('Unexpected dataset partitions.')
    group_sets = {}
    for part in PARTS:
        arrays = data[part]
        if set(arrays)!= {'bits','context','layer','channel','flat_index'}:
            raise ValueError('Unexpected dataset fields.')
        n = manifest['partition_counts'][part]
        if arrays['bits'].shape!=(n,8) or arrays['context'].shape!=(n,13):
            raise ValueError('Dataset shape mismatch.')
        if not np.isin(arrays['bits'],[0,1]).all() or not np.isfinite(arrays['context']).all():
            raise ValueError('Invalid labels/context.')
        for name in ('layer','channel','flat_index'):
            if arrays[name].shape!=(n,) or not np.issubdtype(arrays[name].dtype,np.integer):
                raise ValueError('Invalid provenance array.')
        groups = set(zip(map(int,arrays['layer']),map(int,arrays['channel'])))
        group_sets[part] = groups
        pairs = list(zip(map(int,arrays['layer']),map(int,arrays['flat_index'])))
        if len(set(pairs))!=n:
            raise ValueError('Repeated target location in a partition.')
        for row in manifest['inventory']:
            layer = row['layer_id']
            observed = set(map(int,arrays['channel'][arrays['layer']==layer]))
            if not observed <= set(row['channels'][part]):
                raise ValueError('Target channel is outside its declared split.')
        if set(map(int,np.unique(arrays['layer'])))!=set(range(63)):
            raise ValueError('Every partition must include all 63 layers.')
    if any(group_sets[a]&group_sets[b] for a,b in [('train','calibration'),('train','test'),('calibration','test')]):
        raise ValueError('Output channel leakage between partitions.')
    for row in manifest['inventory']:
        groups = row['channels']
        if any(set(groups[a])&set(groups[b]) for a,b in [('train','calibration'),('train','test'),('calibration','test')]):
            raise ValueError('Manifest channel groups overlap.')


def fit_models(train, bit, protocol):
    from sklearn.ensemble import HistGradientBoostingClassifier
    models = {'frequencies':frequency_fit(train,bit)}
    for method, context in [('D',False),('C',True)]:
        settings = dict(protocol['predictor_'+method])
        settings.pop('family')
        settings.pop('hyperparameter_search')
        estimator = HistGradientBoostingClassifier(**settings,early_stopping=False,
                                                   random_state=protocol['seed']+bit)
        estimator.fit(features(train,bit,context),train['bits'][:,bit])
        models[method] = estimator
    return models


def predict_models(models, partition, bit):
    import numpy as np
    result = frequency_predict(models['frequencies'],partition,bit)
    for method, context in [('D',False),('C',True)]:
        estimator = models[method]
        if len(estimator.classes_)==1:
            result[method] = np.full(len(partition['bits']), float(estimator.classes_[0]))
            continue
        probabilities = estimator.predict_proba(features(partition,bit,context))
        ones = np.flatnonzero(estimator.classes_==1)
        result[method] = probabilities[:,int(ones[0])] if len(ones) else np.zeros(len(probabilities))
    return result


def run(project_root):
    """Fit one frozen bit at a time; save verified bit checkpoints; resume safely."""
    import fcntl
    import platform
    from importlib.metadata import version
    import numpy as np
    import joblib
    from threadpoolctl import threadpool_limits
    root = Path(project_root)
    repo = root/'code'
    protocol_path = repo/'experiments/notebook06a/masked_bit_feasibility_protocol_v2.json'
    protocol = json.loads(verified(protocol_path,PROTOCOL_SHA))
    source = repo/'results/notebook06a/masked_bit_dataset_v1'
    manifest = json.loads(verified(source/'manifest.json',MANIFEST_SHA))
    if manifest['protocol_sha256']!=PROTOCOL_SHA or manifest['dataset_sha256']!=DATASET_SHA:
        raise RuntimeError('Dataset manifest provenance mismatch.')
    raw_dataset = verified(source/'dataset.npz',DATASET_SHA)
    with np.load(io.BytesIO(raw_dataset),allow_pickle=False) as archive:
        expected = {f'{part}_{key}' for part in PARTS for key in ('bits','context','layer','channel','flat_index')}
        if set(archive.files)!=expected:
            raise RuntimeError('Unexpected NPZ entries.')
        data = {part:{key:archive[f'{part}_{key}'].copy() for key in ('bits','context','layer','channel','flat_index')} for part in PARTS}
    validate_dataset(data,manifest)
    if version('scikit-learn')!='1.6.1' or version('numpy')!='2.4.6':
        raise RuntimeError('Research run requires scikit-learn 1.6.1 and NumPy 2.4.6.')
    environment = {'python':platform.python_version(),'packages':{p:version(p) for p in
                   ('numpy','scikit-learn','scipy','joblib','threadpoolctl')},'thread_limit':1}
    session = {'source_sha256':sha256(Path(__file__).read_bytes()),'protocol_sha256':PROTOCOL_SHA,
               'dataset_sha256':DATASET_SHA,'manifest_sha256':MANIFEST_SHA,'environment':environment,
               'implementation':{'layer_encoding':'one_hot_63','fixed_iterations':100,'early_stopping':False,
                   'random_state':'protocol_seed_plus_bit','probability_calibration':'none',
                   'calibration_partition':'evaluated_but_not_used_to_fit_or_select',
                   'bootstrap_seed':BOOTSTRAP_SEED,'bootstrap_draws':BOOTSTRAP_DRAWS,
                   'global_averaging':'target_row_weighted_then_equal_mean_over_8_bits',
                   'probability_clipping':'log_loss_only_Brier_uses_native_probabilities'}}
    fingerprint = sha256(canonical(session))
    out = repo/'results/notebook06a/masked_bit_comparison_v1'
    public = repo/'docs/research_reports/notebook06a_feasibility_v1'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'writer.lock').open('a') as lock, threadpool_limits(limits=1):
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another writer is active.') from None
        if list(out.rglob('*.part')):
            raise RuntimeError('Partial artifacts require review; none were deleted.')
        session_path = out/'session.json'
        if session_path.exists():
            if read_json(session_path)!=session:
                raise RuntimeError('Resume configuration differs.')
        else:
            write_json(session_path,session)
        bit_results, predictions_by_bit = [], []
        for bit in range(8):
            result_path = out/f'bit_{bit}_result.json'
            prediction_path = out/f'bit_{bit}_probabilities.npz'
            model_path = out/f'bit_{bit}_models.joblib'
            if result_path.exists():
                record = read_json(result_path)
                if record['session_fingerprint']!=fingerprint or record['bit']!=bit:
                    raise RuntimeError('Completed bit provenance mismatch.')
                prediction_bytes = verified(prediction_path,record['prediction_sha256'])
                verified(model_path,record['models_sha256'])
                print(f'Bit {bit}: verified saved comparison',flush=True)
            else:
                if prediction_path.exists() or model_path.exists():
                    raise RuntimeError('Unfinished bit artifacts require review before retry.')
                started = time.perf_counter()
                models = fit_models(data['train'],bit,protocol)
                fit_seconds = time.perf_counter()-started
                probabilities = {}
                metrics_result = {}
                prediction_started = time.perf_counter()
                for part in ('calibration','test'):
                    predicted = predict_models(models,data[part],bit)
                    labels = data[part]['bits'][:,bit]
                    for method in METHODS:
                        probabilities[f'{part}_{method}'] = predicted[method].astype(np.float64)
                        metrics_result[f'{part}_{method}'] = metrics(labels,predicted[method])
                prediction_seconds = time.perf_counter()-prediction_started
                prediction_buffer = io.BytesIO()
                np.savez_compressed(prediction_buffer,**probabilities)
                prediction_bytes = prediction_buffer.getvalue()
                prediction_sha = write_bytes(prediction_path,prediction_bytes)
                model_buffer = io.BytesIO()
                joblib.dump(models,model_buffer,compress=3)
                models_sha = write_bytes(model_path,model_buffer.getvalue())
                record = {'session_fingerprint':fingerprint,'bit':bit,'metrics':metrics_result,
                          'prediction_sha256':prediction_sha,'models_sha256':models_sha,
                          'models_storage_bytes':model_path.stat().st_size,
                          'fit_wall_seconds':fit_seconds,'prediction_and_metric_seconds':prediction_seconds,
                          'conditional_B_unseen_test_context_fraction':float(np.mean(
                              models['frequencies']['conditional_training_counts'][data['test']['layer'],
                                  masked_key(data['test']['bits'],bit)]==0))}
                write_json(result_path,record)
                print(f'Bit {bit}: A/B/C/D saved | fit={fit_seconds:.2f}s',flush=True)
            with np.load(io.BytesIO(prediction_bytes),allow_pickle=False) as archive:
                probabilities = {key:archive[key].copy() for key in archive.files}
            if set(probabilities)!={f'{part}_{method}' for part in ('calibration','test') for method in METHODS}:
                raise RuntimeError('Saved probability schema mismatch.')
            for part in ('calibration','test'):
                labels = data[part]['bits'][:,bit]
                for method in METHODS:
                    if metrics(labels,probabilities[f'{part}_{method}'])!=record['metrics'][f'{part}_{method}']:
                        raise RuntimeError('Saved probabilities do not reproduce metrics.')
            bit_results.append(record)
            predictions_by_bit.append(probabilities)
        per_layer_rows = []
        for bit, probabilities in enumerate(predictions_by_bit):
            for part in ('calibration','test'):
                for layer in range(63):
                    mask = data[part]['layer']==layer
                    for method in METHODS:
                        per_layer_rows.append({'partition':part,'layer_id':layer,'layer':manifest['inventory'][layer]['layer'],
                            'bit':bit,'method':method,**metrics(data[part]['bits'][mask,bit],probabilities[f'{part}_{method}'][mask])})
        comparisons = {}
        for reference in ('B','D'):
            delta = np.column_stack([
                bernoulli_losses(data['test']['bits'][:,bit],p['test_C'])-
                bernoulli_losses(data['test']['bits'][:,bit],p['test_'+reference])
                for bit,p in enumerate(predictions_by_bit)])
            comparisons['C_minus_'+reference] = paired_channel_bootstrap(delta,data['test']['layer'],data['test']['channel'])
        summary = {'status':'MASKED_BIT_FEASIBILITY_COMPLETE_NOT_FAULT_REPAIR',
                   'session_fingerprint':fingerprint,'session':session,'targets_by_partition':manifest['partition_counts'],
                   'bits':bit_results,'paired_test_log_loss_comparisons':comparisons,
                   'interpretation':'Negative C-minus-D means lower test log loss with context in this probe; not proof of fault repair.',
                   'models_total_storage_bytes':sum(r['models_storage_bytes'] for r in bit_results),
                   'limitations':protocol['limitations']+[
                       'Known masked-bit position; no fault-localization or correction experiment.',
                       'Model fitting uses training targets only; calibration labels are not used to tune methods.',
                       'Bootstrap conditions on one checkpoint, sampled targets and fitted predictors.',
                       'Intervals are exploratory, not simultaneous; inspect all bits and constant-label groups.',
                       'Per-layer results are descriptive; this run does not measure detection AP recovery.',
                       'Joblib files are executable serialization: load only trusted locally produced files.']}
        public.mkdir(parents=True,exist_ok=True)
        _publish(public/'comparison_summary_v1.json',json.dumps(summary,indent=2,allow_nan=False).encode('utf-8')+b'\n')
        import csv
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream,fieldnames=list(per_layer_rows[0]))
        writer.writeheader()
        writer.writerows(per_layer_rows)
        _publish(public/'per_layer_bit_metrics_v1.csv',stream.getvalue().encode('utf-8'))
        lines = ['# Contextual masked-bit feasibility', '',
                 'This is bit completion, not fault detection, repair or AP recovery.',
                 'All methods are fixed before test evaluation. Context is clean in this probe.',
                 'C/D use the same learner and hyperparameters; D has no neighbor features.',
                 'Calibration targets were evaluated but not used to fit or select methods.', '',
                 '| Bit | A test log loss | B test log loss | D test log loss | C test log loss | C minus D |',
                 '| ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in bit_results:
            losses = {method:r['metrics']['test_'+method]['log_loss'] for method in METHODS}
            lines.append(f'| {r["bit"]} | {losses["A"]:.7f} | {losses["B"]:.7f} | {losses["D"]:.7f} | {losses["C"]:.7f} | {losses["C"]-losses["D"]:+.7f} |')
        lines += ['', '## Paired exploratory channel-bootstrap comparisons', '']
        for name, comparison in comparisons.items():
            lines.append(f'- {name}: mean log-loss difference {comparison["mean_over_bits"]:+.8f}; exploratory interval {comparison["exploratory_percentile_95_interval_mean_over_bits"]}.')
        lines += ['', 'No novelty, repair-success or deployment-safety claim follows from this report.',
                  '', '## Limitations', '', *['- '+v for v in summary['limitations']], '']
        _publish(public/'feasibility_report_v1.md','\n'.join(lines).encode('utf-8'))
        print('Saved report:',public/'feasibility_report_v1.md')
        print('No COCO images, network inference, fault repair, commit or push.')
        return summary


def _publish(path, content):
    if path.exists():
        if path.read_bytes()!=content:
            raise RuntimeError('Existing publication differs; do not overwrite.')
    else:
        write_bytes(path,content)
