"""Scientific failure-mode checks. All generated data are synthetic fixtures."""
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
from checks import HERE,runtime_fixture,random_model
from checks_closure import weights
from features import feature_mask,TOKEN_DIM,INPUT_DIM,FEATURE_SCHEMA
from policy_protocol import DEFAULT_POLICY,POLICY_SCHEMA,validate_model_policy,state_identity
from policy_calibration import complete_groups,calibrate,image_summary
from exchange_numpy import proposals,select
from scorer_numpy import image_split,PairScorer


def records_for(image,n=16,schedule='FIRST_SHIFT'):
    return [dict(source_hash='SYNTHETIC_ONLY',image_id=image,question_id='q',stage1='COVER',
        schedule_name=schedule,prefix_hash='SAME_PREFIX',after=3,depth=32,next_after=13,final_k=4,
        baseline_patch_ids=list(range(8)),donor_patch_ids=[j%8],incoming_patch_ids=[8+j],
        group_size=1,proposal_shortlist=16,candidate_limit=16,catalogue_size=n,catalogue_index=j,
        policy_schema=POLICY_SCHEMA,label_kind='policy') for j in range(n)]


def calibration_fixture():
    images=[]
    for i in range(1000):
        key='SYNTHETIC_CALIBRATION_'+str(i)
        if image_split([key])[1][0]:images.append(key)
        if len(images)==20:break
    rec=[r for im in images for r in records_for(im)]
    x=np.zeros((len(rec),2,INPUT_DIM))
    x[:,0,0]=np.tile(np.linspace(-.1,.1,16),20)
    y=x[:,0,0].copy();validation=np.ones(len(x),dtype=bool)
    class Model:
        def predict(self,x):return x[:,0,0]
    return Model(),x,y,np.asarray([r['image_id'] for r in rec]),rec,validation


class RevisedChecks(unittest.TestCase):
    def test_deployment_catalogue_same_as_collection_and_has_bounded_size(self):
        score=np.linspace(-1,1,172);base=np.arange(56,172)
        d,i=proposals(score,base,candidate_limit=16)
        self.assertEqual(d.shape,(16,1));self.assertEqual(len(set(zip(d[:,0],i[:,0]))),16)
        class First:
            def predict(self,x):return np.arange(len(x),0,-1.)
        got,trace=select(np.zeros((172,TOKEN_DIM)),score,base,First(),after=3,depth=32,
            next_after=13,final_k=32,rounds=1,candidate_limit=16)
        self.assertEqual(trace[0]['donors'],d[0].tolist())
        self.assertEqual(trace[0]['incoming'],i[0].tolist())
        self.assertEqual(len(set(got)-set(base)),1)

    def test_controls_do_not_sneak_history_into_current_comparison(self):
        current=feature_mask('current');shuffled=feature_mask('shuffled');noctx=feature_mask('no_context')
        np.testing.assert_array_equal(current,shuffled)
        np.testing.assert_array_equal(current[:TOKEN_DIM],noctx[:TOKEN_DIM])
        self.assertTrue(np.all(noctx[TOKEN_DIM:2*TOKEN_DIM]==0))
        np.testing.assert_array_equal(current[2*TOKEN_DIM:],noctx[2*TOKEN_DIM:])

    def test_same_prefix_different_future_policy_never_merges(self):
        a=records_for('SYNTHETIC',schedule='FIRST_SHIFT')
        b=records_for('SYNTHETIC',schedule='FIRST_NEXT_WMATCH')
        self.assertNotEqual(state_identity(a[0]),state_identity(b[0]))
        self.assertEqual(len(complete_groups(a+b)),2)
        self.assertEqual(len(complete_groups(a[:-1]+b)),1)

    def test_calibration_can_abstain_and_never_reads_test_labels(self):
        model,x,y,images,rec,val=calibration_fixture()
        report=calibrate(model,x,y,images,rec,val,[1e-6])
        self.assertEqual(report['status'],'CALIBRATED_ACCEPT')
        self.assertTrue(report['accept_enabled'])
        bad=calibrate(model,x,-y,images,rec,val,[1e-6])
        self.assertEqual(bad['status'],'CALIBRATED_ABSTAIN')
        more=records_for('SYNTHETIC_TEST_ONLY')
        xx=np.concatenate([x,np.ones((16,2,INPUT_DIM))])
        yy=np.r_[y,np.full(16,1e6)];vv=np.r_[val,np.zeros(16,dtype=bool)]
        report2=calibrate(model,xx,yy,np.r_[images,['SYNTHETIC_TEST_ONLY']*16],rec+more,vv,[1e-6])
        self.assertEqual(report,report2)

    def test_incomplete_or_cached_only_labels_cannot_calibrate(self):
        model,x,y,images,rec,val=calibration_fixture()
        r=calibrate(model,x,y,images,rec,val,[])
        self.assertEqual(r['status'],'MISSING_GENUINE_NOOP')
        ix=np.array([j%16!=0 for j in range(len(rec))])
        r=calibrate(model,x[ix],y[ix],images[ix],[r for j,r in enumerate(rec) if ix[j]],val[ix],[0.])
        self.assertEqual(r['status'],'INSUFFICIENT_COMPLETE_VALIDATION')

    def test_image_average_not_action_count_weighting(self):
        rows=[{'image_id':'A','gain':1.}]*100+[{'image_id':'B','gain':-1.}]
        self.assertEqual(image_summary(rows)['mean_gain'],0.)

    def test_current_runs_without_full_and_completed_acc_is_not_validation(self):
        from core_matrix import build_core
        from core_progress import inspect
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            r=runtime_fixture(d);weights(r);(Path(d)/'full.npz').unlink()
            m=build_core(r)
            self.assertTrue(inspect(r)['primary_models_ready'])
            self.assertEqual(m['counts']['configurations'],15)
            done={'states':{j['job_id']:'done' for j in m['jobs']}}
            status=inspect(r,[],done,m)
            self.assertTrue(status['primary_acc_comparison_complete'])
            self.assertFalse(status['validated_algorithm_gain'])

    def test_bad_model_shapes_and_uncalibrated_legacy_models_rejected(self):
        m=random_model()
        with self.assertRaises(ValueError):PairScorer(m.mean,m.std*0,m.mask,m.w,m.b,m.v)
        with self.assertRaises(ValueError):validate_model_policy({'schema':FEATURE_SCHEMA})

    def test_fit_roundtrip_marks_synthetic_and_checks_all_image_splits(self):
        from fit_models import train
        from core_matrix import learned_spec
        rng=np.random.default_rng(29);records=[];inputs=[]
        for j in range(100):
            records.extend(records_for('SYNTHETIC_TRAIN_'+str(j)))
            block=rng.normal(size=(16,2,INPUT_DIM));block[:,1,TOKEN_DIM:]=block[:,0,TOKEN_DIM:]
            inputs.extend(block)
        x=np.asarray(inputs);y=.02*(x[:,0,0]-x[:,1,0])
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            root=Path(d);shard=root/'synthetic.npz'
            np.savez(shard,kind='synthetic',schema=FEATURE_SCHEMA,pairs=x,labels=y,
                records=np.array([json.dumps(r) for r in records]),
                noop_records=np.array([json.dumps({'actually_recomputed':True,'nll_difference':0.})]))
            with self.assertRaises(ValueError):train([shard],root/'forbidden',epochs=2)
            report=train([shard],root/'models',epochs=8,hidden=8,allow_synthetic=True)
            self.assertEqual(report['full']['status'],'UNAVAILABLE_MISSING_HISTORY')
            _,meta=PairScorer.load(root/'models/current.npz')
            self.assertEqual(meta['training_kind'],'synthetic')
            self.assertEqual(meta['deployment_policy'],DEFAULT_POLICY)
            runtime=runtime_fixture(root/'models');runtime['kind']='real'
            with self.assertRaises(ValueError):learned_spec(runtime,'current')

    def test_catalogue_matched_random_uses_same_action_space(self):
        from selector_audit import make_trace,matched_random_numpy
        score=np.arange(24.);base=np.arange(12);ids=np.arange(100,124)
        d,i=proposals(score,base,candidate_limit=16)
        chosen=np.setdiff1d(base,d[0]);chosen=np.r_[chosen,i[0]]
        row=make_trace(request_key='q',config_id='m',source_hash='s',baseline_ids=ids[base],
            selected_ids=ids[chosen],eligible_ids=ids,hook_called=True)
        actions={(tuple(a),tuple(b)) for a,b in zip(d,i)}
        for seed in range(10):
            got=matched_random_numpy(score,base,ids,row,'q',seed=seed,candidate_limit=16)
            self.assertIn((tuple(sorted(set(base)-set(got))),tuple(sorted(set(got)-set(base)))),actions)

    def test_factorial_interaction_zero_and_task_rescoring(self):
        from checks import StatisticsChecks
        from factorial_interaction import interaction
        data=StatisticsChecks().fixture()
        r=interaction(data,copy.deepcopy(data),copy.deepcopy(data),copy.deepcopy(data),pope_metric='f1',replicates=100)
        self.assertEqual(r['interaction_cover_minus_base_schedule_B_minus_A']['R4'],0.)
        self.assertEqual(r['paired_image_95pct_interval']['R4'],[0.,0.])

    def test_bounded_driver_from_collection_to_confirmation_is_explicitly_synthetic(self):
        from build_matrix import digest
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            root=Path(d);r=runtime_fixture(root/'models')
            r['schedules']['FIRST_NEXT_WMATCH']=[[3,110],[12,32]]
            r['fit_epochs']=3;r['estimated_job_seconds']={t:.01 for t in r['metric_keys']}
            collector=root/'collector.py'
            collector.write_text("""import json,sys
import numpy as np
from checks_r2 import records_for
from features import INPUT_DIM,FEATURE_SCHEMA
rng=np.random.default_rng(112);records=[];inputs=[]
for i in range(100):
    rows=records_for('SYNTHETIC_DRIVER_'+str(i))
    for row in rows: row['source_hash']=sys.argv[2]
    records.extend(rows)
    x=rng.normal(size=(16,2,INPUT_DIM));x[:,1,22:]=x[:,0,22:];inputs.extend(x)
x=np.asarray(inputs);y=.02*(x[:,0,0]-x[:,1,0])
np.savez(sys.argv[1],kind='synthetic',schema=FEATURE_SCHEMA,pairs=x,labels=y,
    records=np.array([json.dumps(r) for r in records]),
    noop_records=np.array([json.dumps({'actually_recomputed':True,'nll_difference':0.})]))
""")
            collector.write_text('import sys\nsys.path.insert(0,'+repr(str(HERE.resolve()))+')\n'+collector.read_text())
            r['collection']={'command':[sys.executable,str(collector),'{out}',digest(r['source'])],
                             'output_shard':str(root/'pairs.npz')}
            child=root/'child.py'
            child.write_text("""import json,sys
from pathlib import Path
p=Path(sys.argv[1]);j=json.loads(p.read_text());out=p.parent
records=out/'records.json';records.write_text('[]')
r={'job_hash':j['job_hash'],'panel_ids_hash':j['panel']['ids_hash'],
   'metric_key':j['metric_key'],'metric_value':1.,'records_path':str(records),
   'fixture_kind':'synthetic'}
s=j['config']['selector']
if s['kind']=='continuation':r['scorer_sha256']=s['model_sha256']
(out/'result.json').write_text(json.dumps(r))
""")
            rp=root/'runtime.json';rp.write_text(json.dumps(r))
            bp=root/'bridge.json';bp.write_text(json.dumps({'commands':{'ACC':[sys.executable,str(child),'{job_json}']}}))
            result=subprocess.run([sys.executable,str(HERE/'close_core.py'),str(rp),'--bridge-config',str(bp),
                '--out',str(root/'out'),'--hours','.02','--collection-seconds','5','--screen-seconds','15',
                '--reserve-seconds','0','--skip-timing','--allow-synthetic'],capture_output=True,text=True,timeout=45)
            logs='\n'.join(f.read_text() for f in (root/'out').glob('*error.json'))
            log=root/'out/collection.log'
            if log.is_file():logs+='\n'+log.read_text()
            self.assertEqual(result.returncode,0,result.stdout+result.stderr+logs)
            status=json.loads((root/'out/final_core_progress.json').read_text())
            self.assertTrue(status['primary_acc_comparison_complete'])
            self.assertFalse(status['validated_algorithm_gain'])
            self.assertTrue((root/'out/core_confirm_summary.json').is_file())
            _,meta=PairScorer.load(root/'models/current.npz')
            self.assertEqual(meta['training_kind'],'synthetic')

    def test_external_anchor_needs_real_trace_not_reported_W(self):
        from measurement_anchors import external_anchors
        r=runtime_fixture('.')
        r['external_anchors']=[{'stage1':'OFFICIAL','schedule_name':'S0','selector_name':'reference',
            'selector':{'kind':'reference'},'native_method':'ViTCoP_S0','application':'original_project_path',
            'budget':{'visual_token_layers':2034}}]
        with self.assertRaises(ValueError):external_anchors(r)


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch unavailable; CUDA/VLM checks still required')
class TorchR2Checks(unittest.TestCase):
    def test_capped_catalogue_selection_parity(self):
        import torch
        import module_torch as mt
        r=np.random.default_rng(101);x=r.normal(size=(172,TOKEN_DIM));score=r.normal(size=172)
        base=np.arange(116);m=random_model()
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d)/'m.npz';m.save(p,{'mode':'current'});tm=mt.PairScorer(p)
            got,_=select(x,score,base,m,after=3,depth=32,next_after=13,final_k=32,candidate_limit=16)
            actual=mt.select(torch.tensor(x,dtype=torch.float32),torch.tensor(score),torch.tensor(base),tm,
                after=3,depth=32,next_after=13,final_k=32,candidate_limit=16)
            np.testing.assert_array_equal(got,actual.numpy())


if __name__=='__main__':unittest.main(verbosity=2)
