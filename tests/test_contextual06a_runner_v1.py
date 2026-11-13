"""Synthetic controls only; no research dataset or held-out labels are loaded."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from threadpoolctl import threadpool_limits

spec=importlib.util.spec_from_file_location('contextual06a_runner_v1',Path(__file__).resolve().parents[1]/'src/contextual06a_runner_v1.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def partition(n=128):
    rng=np.random.default_rng(77)
    return {'bits':rng.integers(0,2,size=(n,8),dtype=np.uint8),
            'context':rng.normal(size=(n,13)).astype(np.float32),
            'layer':rng.integers(0,3,size=n,dtype=np.int16),
            'channel':np.arange(n,dtype=np.int32),
            'flat_index':np.arange(n,dtype=np.int64)}


class FeasibilityControls(unittest.TestCase):
    def test_target_flip_does_not_change_features_or_key(self):
        data=partition()
        for bit in range(8):
            changed={key:value.copy() for key,value in data.items()}
            changed['bits'][:,bit]^=1
            np.testing.assert_array_equal(m.masked_key(data['bits'],bit),m.masked_key(changed['bits'],bit))
            for context in (False,True):
                np.testing.assert_array_equal(m.features(data,bit,context),m.features(changed,bit,context))

    def test_addresses_and_channel_identity_are_not_features(self):
        data=partition()
        changed={key:value.copy() for key,value in data.items()}
        changed['channel']+=10000
        changed['flat_index']+=9999
        for context in (False,True):
            np.testing.assert_array_equal(m.features(data,3,context),m.features(changed,3,context))

    def test_context_is_only_extra_feature_block(self):
        data=partition()
        c=m.features(data,0,True)
        d=m.features(data,0,False)
        self.assertEqual(d.shape,(128,70))
        self.assertEqual(c.shape,(128,83))
        np.testing.assert_array_equal(c[:,:70],d)
        np.testing.assert_array_equal(c[:,70:],data['context'])

    def test_invalid_feature_labels_rejected(self):
        data=partition()
        data['bits'][0,0]=2
        with self.assertRaises(ValueError):
            m.features(data,0,True)

    def test_frequency_predictions_ignore_hidden_target_bit(self):
        train=partition()
        for bit in range(8):
            model=m.frequency_fit(train,bit)
            changed={key:value.copy() for key,value in train.items()}
            changed['bits'][:,bit]^=1
            first=m.frequency_predict(model,train,bit)
            second=m.frequency_predict(model,changed,bit)
            for method in ('A','B'):
                np.testing.assert_array_equal(first[method],second[method])

    def test_unobserved_frequency_context_is_smoothed(self):
        train=partition(2)
        train['bits'][:]=0
        train['layer'][:]=0
        test=partition(1)
        test['bits'][:]=1
        test['layer'][:]=0
        p=m.frequency_predict(m.frequency_fit(train,0),test,0)
        self.assertEqual(p['A'][0],.25)
        self.assertEqual(p['B'][0],.5)

    def test_constant_label_group_has_no_fake_balanced_accuracy(self):
        result=m.metrics(np.zeros(10),np.zeros(10))
        self.assertIsNone(result['balanced_accuracy'])
        self.assertTrue(result['constant_label_group'])
        self.assertTrue(np.isfinite(result['log_loss']))
        self.assertEqual(result['brier_score'],0.)

    def test_nonfinite_or_out_of_range_probability_rejected(self):
        for p in (np.array([np.nan]),np.array([1.1]),np.array([-.1])):
            with self.assertRaises(ValueError):
                m.metrics(np.array([0]),p)

    def test_probability_calculation_matches_hand_value(self):
        expected=-np.log(.75)
        result=m.metrics(np.array([0,1]),np.array([.25,.75]))
        self.assertAlmostEqual(result['log_loss'],expected)
        self.assertAlmostEqual(result['brier_score'],.0625)

    def test_identical_methods_have_zero_paired_interval(self):
        result=m.paired_channel_bootstrap(np.zeros((12,8)),np.repeat([0,1],6),np.tile([0,0,1,1,2,2],2),draws=30)
        self.assertEqual(result['mean_over_bits'],0.)
        self.assertEqual(result['exploratory_percentile_95_interval_mean_over_bits'],[0.,0.])
        self.assertEqual(result['test_output_channel_clusters'],6)

    def test_bootstrap_keeps_cluster_dependence(self):
        delta=np.repeat([-1.,1.],20)
        result=m.paired_channel_bootstrap(delta,np.zeros(40),np.repeat([0,1],20),draws=100)
        self.assertEqual(result['exploratory_percentile_95_interval_by_bit'][0],[-1.,1.])

    def test_exclusive_records_and_checksum_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'record.json'
            m.write_json(path,{'test':1})
            self.assertEqual(m.read_json(path),{'test':1})
            with self.assertRaises(FileExistsError):
                m.write_json(path,{'test':2})
            path.write_bytes(path.read_bytes().replace(b'"test": 1',b'"test": 2'))
            with self.assertRaises(RuntimeError):
                m.read_json(path)
            partial=Path(directory)/'other.part'
            partial.write_bytes(b'evidence')
            with self.assertRaises(FileExistsError):
                m.write_bytes(Path(directory)/'other',b'new')
            self.assertEqual(partial.read_bytes(),b'evidence')

    def test_matched_predictors_fit_and_serialize_on_synthetic_data(self):
        import io
        import joblib
        data=partition(400)
        settings={'family':'HistGradientBoostingClassifier_one_model_per_bit',
                  'max_iter':5,'max_leaf_nodes':3,'learning_rate':.1,'min_samples_leaf':10,
                  'l2_regularization':1.,'hyperparameter_search':False}
        protocol={'seed':20261116,'predictor_C':settings,'predictor_D':settings}
        with threadpool_limits(limits=1):
            models=m.fit_models(data,2,protocol)
            self.assertFalse(models['C'].early_stopping)
            self.assertEqual(models['C'].random_state,models['D'].random_state)
            buffer=io.BytesIO()
            joblib.dump(models,buffer)
            loaded=joblib.load(io.BytesIO(buffer.getvalue()))
            predicted=m.predict_models(loaded,data,2)
            self.assertEqual(set(predicted),{'A','B','C','D'})
            for values in predicted.values():
                self.assertEqual(len(values),400)
                self.assertTrue(np.isfinite(values).all())


if __name__=='__main__':
    unittest.main()
