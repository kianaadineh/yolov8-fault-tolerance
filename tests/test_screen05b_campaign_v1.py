"""CPU-only control tests. They do not validate CUDA inference or COCOeval."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('screen05b_campaign_v1',Path(__file__).resolve().parents[1]/'src/screen05b_campaign_v1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CampaignControlTests(unittest.TestCase):
    def test_signed_bit_boundaries(self):
        self.assertEqual(m.bitflip(0,7),-128)
        self.assertEqual(m.bitflip(-128,7),0)
        self.assertEqual(m.bitflip(127,7),-1)
        for q in range(-128,128):
            for b in range(8):
                self.assertEqual(m.bitflip(m.bitflip(q,b),b),q)
                self.assertNotEqual(m.bitflip(q,b),q)

    def test_invalid_bit_values(self):
        for q, b in [(True,0),(0,True),(128,0),(0,8),(0,1.0)]:
            with self.assertRaises(ValueError):
                m.bitflip(q,b)

    def test_sampling_is_balanced_and_without_replacement(self):
        inventory=[{'name':f'layer.{i}','weight_count':100+i} for i in range(63)]
        scenarios=m.make_locations(inventory)
        self.assertEqual(len(scenarios),5040)
        self.assertEqual(len({r['scenario_id'] for r in scenarios}),5040)
        for layer in inventory:
            for b in range(8):
                indices=[r['flat_index'] for r in scenarios if r['layer']==layer['name'] and r['bit']==b]
                self.assertEqual(len(indices),10)
                self.assertEqual(len(set(indices)),10)
                self.assertTrue(all(0<=i<layer['weight_count'] for i in indices))

    def test_sampling_reproducibility_and_layer_stream_independence(self):
        a={'name':'a','weight_count':100}
        b={'name':'b','weight_count':100}
        first=m.make_locations([a,b])
        self.assertEqual(first,m.make_locations([a,b]))
        def locations(rows):
            return sorted((r['bit'],r['flat_index']) for r in rows if r['layer']=='a')
        self.assertEqual(locations(first),locations(m.make_locations([b,a])))

    def test_equal_bit_mean_does_not_cancel_harm(self):
        rows=[{'bit':b,'loss':.1 if b==7 else 0.} for b in range(8) for _ in range(10)]
        result=m.stratified_loss_summary(rows,draws=200)
        self.assertAlmostEqual(result['equal_bit_mean_nonnegative_AP_loss'],.0125)
        self.assertAlmostEqual(result['max_observed_loss'],.1)
        self.assertEqual(result,m.stratified_loss_summary(rows,draws=200))

    def test_incomplete_or_unbalanced_summaries_rejected(self):
        balanced=[{'bit':b,'loss':0.} for b in range(8)]
        for rows in [balanced[:-1],balanced+[{'bit':0,'loss':0.}]]:
            with self.assertRaises(ValueError):
                m.stratified_loss_summary(rows,draws=10)

    def test_nonfinite_or_negative_losses_rejected(self):
        for value in [float('nan'),float('inf'),-.1]:
            rows=[{'bit':b,'loss':0.} for b in range(8)]
            rows[0]['loss']=value
            with self.assertRaises(ValueError):
                m.stratified_loss_summary(rows,draws=10)

    def test_exclusive_write_and_partial_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json'
            m.Campaign._exclusive_json(path,{'a':1})
            with self.assertRaises(FileExistsError):
                m.Campaign._exclusive_json(path,{'a':2})
            self.assertEqual(json.loads(path.read_bytes()),{'a':1})
            other=Path(directory)/'other.json'
            partial=other.with_name(other.name+'.part')
            partial.write_bytes(b'evidence')
            with self.assertRaises(FileExistsError):
                m.Campaign._exclusive_json(other,{'b':2})
            self.assertEqual(partial.read_bytes(),b'evidence')

    def test_prediction_execution_includes_zero_detection_images(self):
        campaign=m.Campaign.__new__(m.Campaign)
        campaign.ids=[1,2]
        campaign.category_ids={1}
        data={'image_ids':[1,2],'predictions':[],
              'per_image':[{'image_id':1,'count':0,'degenerate':0},{'image_id':2,'count':0,'degenerate':0}]}
        campaign._validate_predictions(data)
        missing=copy.deepcopy(data)
        missing['per_image'].pop()
        with self.assertRaises(RuntimeError):
            campaign._validate_predictions(missing)

    def test_prediction_nonfinite_and_count_mismatch(self):
        campaign=m.Campaign.__new__(m.Campaign)
        campaign.ids=[1]
        campaign.category_ids={1}
        data={'image_ids':[1],'predictions':[{'image_id':1,'category_id':1,'bbox':[0.,0.,1.,1.],'score':.5}],
              'per_image':[{'image_id':1,'count':1,'degenerate':0}]}
        campaign._validate_predictions(data)
        for change in ('count','nan','reference'):
            bad=copy.deepcopy(data)
            if change=='count': bad['per_image'][0]['count']=0
            if change=='nan': bad['predictions'][0]['score']=float('nan')
            if change=='reference': bad['predictions'][0]['image_id']=9
            with self.assertRaises(RuntimeError):
                campaign._validate_predictions(bad)

    def test_resume_provenance_mismatch_rejected(self):
        campaign=m.Campaign.__new__(m.Campaign)
        campaign.fingerprint='current'
        campaign.plan_sha='plan'
        class FakeBase:
            @staticmethod
            def read_record(path):
                return {'kind':'predictions','session_fingerprint':'other','plan_sha256':'plan'}
        campaign.base=FakeBase()
        with self.assertRaises(RuntimeError):
            campaign._read('unused','predictions')

    def test_no_ranking_on_partial_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign=m.Campaign.__new__(m.Campaign)
            campaign.public=Path(directory)
            campaign.plan={'inventory':[{'name':'layer'}],'limitations':[],'confirmation':'required'}
            campaign.plan_sha='plan'
            campaign.fingerprint='session'
            campaign.clean_metrics={'AP50_95':.5}
            campaign._summary_cache={}
            record={'scenario':{'layer':'layer','bit':0},'loss':.01,'AP50_95_change':-.01}
            result=campaign._write_progress([record])
            self.assertEqual(result['exploratory_ranking'],[])
            self.assertEqual(result['status'],'PARTIAL_SCREENING_NO_RANKING')


if __name__=='__main__':
    unittest.main()
