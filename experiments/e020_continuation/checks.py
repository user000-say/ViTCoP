"""Meaningful local checks. Synthetic fixtures are never VLM results."""
import copy
import contextlib
import io
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from budget import profile,matched_two_event
from features import token_features,pair_inputs,INPUT_DIM,TOKEN_DIM
from scorer_numpy import PairScorer,fit,image_split
from exchange_numpy import select,proposals
from build_matrix import build,digest,job_digest
from paired_stats import compare,prepare
from pairdata import collect

HERE=Path(__file__).parent


def runtime_fixture(model_dir):
    schedules={'ONE':[[8,32]],'EARLY':[[4,102],[12,32]],'LATE':[[4,102],[16,32]],
               'FIRST_SHIFT':[[2,116],[12,32]],'LATE_WMATCH':[[4,78],[16,32]],
               'THREE_WMATCH':[[4,103],[8,101],[12,32]]}
    panels={p:{t:{'path':'synthetic_only.json','ids_hash':'test_'+t} for t in ('GQA','POPE','MME','MMBench')} for p in ('screen','confirm')}
    return {'schema':'e020_runtime_v1','kind':'synthetic','model_depth':32,'initial_visual_tokens':172,
            'layer_convention':'zero_based_after_block','schedules':schedules,'model_directory':str(model_dir),
            'source':{k:'TEST_ONLY' for k in ('checkpoint_hash','code_commit','prompt_hash','input_pipeline_hash','cache_policy','query_rule','feature_protocol_hash')},
            'panels':panels,'metric_keys':{t:'test_metric' for t in panels['screen']},'expected_visual_work':{'EARLY':2284}}


def random_model(seed=20):
    r=np.random.default_rng(seed)
    return PairScorer(np.zeros(INPUT_DIM),np.ones(INPUT_DIM),np.ones(INPUT_DIM),
                      r.normal(0,.1,(INPUT_DIM,8)),np.zeros(8),r.normal(0,.1,8))


class MathematicalChecks(unittest.TestCase):
    def test_budget_convention_and_first_three_work(self):
        first=profile(32,172,[[2,116],[12,32]],'zero_based_after_block')
        three=profile(32,172,[[4,103],[8,101],[12,32]],'zero_based_after_block')
        early=profile(32,172,[[4,102],[12,32]],'zero_based_after_block')
        self.assertEqual(first['visual_token_layers'],2284)
        self.assertEqual(three['visual_token_layers'],2284)
        self.assertEqual(profile(32,172,[[4,78],[16,32]],'zero_based_after_block')['visual_token_layers'],2276)
        self.assertEqual(early['visual_token_layers'],2284)
        self.assertLess(first['visual_square_sum'],early['visual_square_sum'])
        wrong=profile(32,172,[[2,116],[12,32]],'completed_blocks')
        self.assertEqual(first['visual_token_layers']-wrong['visual_token_layers'],140)

    def test_matched_budget_integer_feasibility(self):
        for after in range(1,10):
            e=matched_two_event(32,172,2284,after,13,32)
            w=profile(32,172,e)['visual_token_layers']
            self.assertLessEqual(w,2284);self.assertLess(2284-w,13-after)
        with self.assertRaises(ValueError):matched_two_event(32,172,2284,10,13,32)

    def test_feature_prompt_identity_and_projection(self):
        r=np.random.default_rng(1);s=r.normal(size=12);a=r.random((3,12));h=r.normal(size=(12,24));p=r.normal(size=(24,8))
        x=token_features(s,a,previous=a[:,::-1],hidden=h,projection=p)
        self.assertEqual(x.shape,(12,TOKEN_DIM));self.assertTrue(np.isfinite(x).all())
        with self.assertRaises(ValueError):token_features(s,a,previous=a[:,:-1])

    def test_pair_context_is_remaining_set(self):
        r=np.random.default_rng(2);x=r.normal(size=(12,TOKEN_DIM));s=np.arange(8)
        p=pair_inputs(x,s,np.array([[6,7]]),np.array([[8,9]]),after=3,depth=32,next_after=13,final_k=4)
        np.testing.assert_allclose(p[0,0,TOKEN_DIM:TOKEN_DIM*2],x[:6].mean(0))
        np.testing.assert_allclose(p[0,0,TOKEN_DIM:],p[0,1,TOKEN_DIM:])
        with self.assertRaises(ValueError):pair_inputs(x,s,np.array([[6,6]]),np.array([[8,9]]),after=3,depth=32,next_after=13,final_k=4)

    def test_predictor_swap_antisymmetry_and_persistence(self):
        r=np.random.default_rng(3);x=r.normal(size=(7,2,INPUT_DIM));m=random_model()
        np.testing.assert_allclose(m.predict(x),-m.predict(x[:,::-1]),atol=1e-12)
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            path=Path(d)/'m.npz';m.save(path,{'mode':'full'})
            got,_=PairScorer.load(path);np.testing.assert_allclose(m.predict(x),got.predict(x))

    def test_real_learning_and_image_split(self):
        r=np.random.default_rng(4);x=r.normal(size=(500,2,INPUT_DIM))
        # Same shared context on both sides, matching the deployed pair format.
        x[:,1,TOKEN_DIM:]=x[:,0,TOKEN_DIM:]
        y=.04*(x[:,0,6]-x[:,1,6])+.02*(x[:,0,0]-x[:,1,0])
        ids=np.array(['synthetic_image_'+str(i//2) for i in range(500)])
        m,report=fit(x,y,ids,epochs=90,hidden=12)
        self.assertLess(report['test']['rmse'],report['test']['zero_predictor_rmse']*.4)
        self.assertGreater(report['test']['sign_accuracy_nonzero'],.85)
        split=image_split(ids)
        self.assertFalse(set(ids[split[0]])&set(ids[split[1]]))
        self.assertFalse(set(ids[split[0]])&set(ids[split[2]]))

    def test_exchange_budget_protected_and_zero_rounds(self):
        r=np.random.default_rng(5);x=r.normal(size=(24,TOKEN_DIM));score=np.arange(24.)
        s=np.arange(12);protect=np.zeros(24,dtype=bool);protect[:4]=True
        for g in (1,4):
            got,_=select(x,score,s,random_model(),after=3,depth=32,next_after=13,final_k=4,
                         group_size=g,rounds=4,protected=protect,mode='random')
            self.assertEqual(len(got),12);self.assertEqual(len(set(got)),12)
            self.assertTrue(set(range(4))<=set(got))
        got,_=select(x,score,s,None,after=3,depth=32,next_after=13,final_k=4,rounds=0)
        np.testing.assert_array_equal(got,s)

    def test_reject_negative_predicted_actions(self):
        class Negative:
            def predict(self,x):return -np.ones(len(x))
        x=np.zeros((12,TOKEN_DIM));s=np.arange(8)
        got,diag=select(x,np.arange(12.),s,Negative(),after=3,depth=32,next_after=13,final_k=4)
        np.testing.assert_array_equal(got,s);self.assertEqual(diag,[])

    def test_matrix_72_configs_288_jobs_and_real_guards(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            runtime=runtime_fixture(d)
            for mode in ('full','current','no_context','shuffled'):
                random_model().save(Path(d)/(mode+'.npz'),{'mode':mode,'source_hash':digest(runtime['source'])})
            matrix=build(runtime)
            self.assertEqual(matrix['counts'],{'configurations':72,'task_jobs':288})
            self.assertEqual(len({j['job_hash'] for j in matrix['jobs']}),288)
            self.assertEqual(build(runtime,only='independent')['counts']['configurations'],36)
            runtime['kind']='real'
            with self.assertRaises(ValueError):build(runtime)

    def test_complete_four_task_promotion(self):
        from summarize import aggregate,promote
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            r=runtime_fixture(d);m=build(r,only='independent');results={}
            for j in m['jobs']:
                results[j['job_id']]={'metric_value':90. if j['task']!='MME' else 1700.}
            state={'results':results};summary=aggregate(m,state)
            self.assertEqual(summary['completed_four_task_configs'],36)
            confirm=promote(r,[m],[state],top=4)
            self.assertEqual(confirm['counts']['configurations'],8)
            j=m['jobs'][0];results.pop(j['job_id'])
            self.assertEqual(aggregate(m,state)['completed_four_task_configs'],35)


class RealSubprocessChecks(unittest.TestCase):
    def test_replay_adapter_baseline_reuse_and_actual_prefix_check(self):
        from replay_adapter import ReplayAdapter
        class Backend:
            kind='synthetic'
            calls=0
            def examples(self):return [{'image_id':'toy_replay','question_id':'q'}]
            def run_teacher_forced(self,e,*,stage1,schedule_name,first_keep_patch_ids,forced_final_ids,capture_prompt_state):
                self.calls+=1
                s={**e,'features':np.zeros((12,TOKEN_DIM)),'scores':np.arange(12.),'selected':np.arange(8),
                   'patch_ids':np.arange(100,112),'after':3,'depth':32,'next_after':13,'final_k':4}
                ids=np.arange(100,108) if first_keep_patch_ids is None else np.asarray(first_keep_patch_ids)
                return {'snapshot':s,'prefix_hash':'actual_toy_prompt_hash','cache_policy':'TEST',
                        'answer_nll':float(ids.mean()),'answer_tokens_hash':'ANSWER','answer_token_count':2}
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            backend=Backend();adapter=ReplayAdapter(backend,{'source':{'test':'synthetic'},'arms':['COVER']})
            r=collect(adapter,Path(d)/'p.npz',max_states=1,actions_per_state=3,identity_every=0)
            self.assertEqual(r['paired_actions'],3)
            self.assertEqual(backend.calls,4)  # one baseline, three real replays

    def test_real_subprocess_retry_resume_isolated_gpu(self):
        from run_matrix import run
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            root=Path(d);r=runtime_fixture(root);matrix=build(r,only='independent')
            matrix['jobs']=matrix['jobs'][:2]
            for j in matrix['jobs']:j['estimated_seconds']=.1
            child=root/'child.py';child.write_text('''import json,os,sys
from pathlib import Path
out=Path(sys.argv[1]);j=json.loads((out/'job.json').read_text())
flag=out.parent/'has_failed'
if not flag.exists():flag.write_text('yes');raise SystemExit(2)
rec=out/'records.json';rec.write_text('[]')
data={'job_hash':j['job_hash'],'metric_key':j['metric_key'],'panel_ids_hash':j['panel']['ids_hash'],
      'metric_value':1.,'records_path':str(rec.resolve()),'fixture_kind':'synthetic',
      'observed_gpu':os.environ['CUDA_VISIBLE_DEVICES']}
(out/'result.json').write_text(json.dumps(data))
''')
            class Bridge:
                def command(self,job,outdir):return [sys.executable,str(child),outdir]
                def read_result(self,job,outdir):return json.loads((Path(outdir)/'result.json').read_text())
            with contextlib.redirect_stdout(io.StringIO()):
                state=run(matrix,Bridge(),root/'queue',('0','2'),seconds=10,reserve=0,allow_synthetic=True)
                again=run(matrix,Bridge(),root/'queue',('0','2'),seconds=10,reserve=0,allow_synthetic=True)
            self.assertTrue(all(v=='done' for v in state['states'].values()))
            self.assertTrue(all(v['observed_gpu'] in ('0','2') for v in state['results'].values()))
            self.assertEqual(state['attempts'],again['attempts'])
            with self.assertRaises(ValueError):run(matrix,Bridge(),root/'forbidden',seconds=10,reserve=0)

    def test_real_deadline_kills_only_owned_process(self):
        from run_matrix import run
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            root=Path(d);r=runtime_fixture(root);m=build(r,only='independent');m['jobs']=m['jobs'][:1]
            m['jobs'][0]['estimated_seconds']=.01
            class Bridge:
                def command(self,job,outdir):return [sys.executable,'-c','import time;time.sleep(30)']
                def read_result(self,job,outdir):raise AssertionError('should not complete')
            with contextlib.redirect_stdout(io.StringIO()):
                s=run(m,Bridge(),root/'q',('0',),seconds=.25,reserve=0,allow_synthetic=True)
            self.assertIn('interrupted',s['states'].values())

    def test_collection_clones_prefix_and_refuses_synthetic_fit(self):
        from fit_models import train
        class Adapter:
            kind='synthetic'
            def snapshots(self):
                r=np.random.default_rng(7)
                yield dict(image_id='synthetic',question_id='q',stage1='COVER',schedule_name='FIRST_SHIFT',source_hash='TEST',cache_policy='TEST',
                           features=r.normal(size=(12,TOKEN_DIM)),scores=np.arange(12.),selected=np.arange(8),
                           patch_ids=np.arange(100,112),after=3,depth=32,next_after=13,final_k=4)
            def clone(self,s):return copy.deepcopy(s)
            def prefix_digest(self,s):return digest(np.asarray(s['selected']).tolist())
            def run_suffix(self,s,keep_patch_ids,forced_final_ids=None):
                return {'answer_nll':float(np.mean(keep_patch_ids)),'prefix_hash':self.prefix_digest(s),
                        'cache_policy':'TEST','answer_tokens_hash':'TEST_ANSWER','answer_token_count':2}
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            shard=Path(d)/'p.npz';summary=collect(Adapter(),shard,max_states=1,actions_per_state=4,identity_every=0)
            self.assertEqual(summary['paired_actions'],4)
            with self.assertRaises(ValueError):train([shard],Path(d)/'models')
            from diagnostics import diagnose
            class Many(Adapter):
                def snapshots(self):
                    for image in range(32):
                        s=next(super().snapshots());s['image_id']='toy_'+str(image)
                        s['features'][:,0]=np.arange(12.)
                        yield s
                def prefix_digest(self,s):return digest([s['image_id'],np.asarray(s['selected']).tolist()])
            more=Path(d)/'more.npz';collect(Many(),more,max_states=32,actions_per_state=3,identity_every=0)
            with self.assertRaises(ValueError):diagnose([more],d)
            report=diagnose([more],d,allow_synthetic=True)
            self.assertEqual(report['sampled_one_step_oracle_positive_gain'],0.)
            self.assertLess(report['methods']['current_score_difference']['mean_one_step_real_nll_gain'],0.)


class StatisticsChecks(unittest.TestCase):
    def fixture(self):
        return {t:[{'id':str(i),'image_id':str(i//2),'correct':int(i%3!=0),'label':i%2,
                    'prediction':int(i%3!=0),'category':'one'} for i in range(40)]
                for t in ('GQA','POPE','MME','MMBench')}

    def test_bootstrap_identical_pair_zero_and_pope_rescoring(self):
        data=self.fixture();r=compare(data,copy.deepcopy(data),pope_metric='f1',replicates=100)
        self.assertEqual(r['delta']['R4'],0)
        self.assertEqual(r['paired_95pct_interval']['R4'],[0.,0.])
        _,score=prepare(data,'f1');_,accuracy=prepare(data,'accuracy')
        self.assertNotEqual(score(np.ones(80))['POPE'],accuracy(np.ones(80))['POPE'])

    def test_pair_identity_errors(self):
        a=self.fixture();b=copy.deepcopy(a);b['GQA'].pop()
        with self.assertRaises(ValueError):compare(a,b,pope_metric='f1',replicates=100)
        a=self.fixture();a['MME'].pop()
        with self.assertRaises(ValueError):prepare(a,'f1')


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch unavailable locally; run on server')
class TorchChecks(unittest.TestCase):
    def test_full_feature_scorer_and_selection_parity(self):
        import torch
        import module_torch as mt
        r=np.random.default_rng(8);score=r.normal(size=24);a=r.random((4,24));prev=r.random((4,24))
        h=r.normal(size=(24,32));proj=r.normal(size=(32,8));m=random_model()
        x=token_features(score,a,previous=prev,hidden=h,projection=proj)
        xt=mt.token_features(torch.tensor(score),torch.tensor(a),previous=torch.tensor(prev),
                             hidden=torch.tensor(h),projection=torch.tensor(proj))
        np.testing.assert_allclose(x,xt.numpy(),atol=2e-6)
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d)/'m.npz';m.save(p,{'mode':'full'});tm=mt.PairScorer(p)
            for g in (1,4):
                selected=np.arange(12)
                expected,_=select(x,score,selected,m,after=3,depth=32,next_after=13,final_k=4,group_size=g,rounds=4)
                got=mt.select(xt,torch.tensor(score),torch.tensor(selected),tm,
                              after=3,depth=32,next_after=13,final_k=4,group_size=g,rounds=4)
                np.testing.assert_array_equal(expected,got.numpy())

    def test_teacher_forced_answer_only_nll(self):
        import torch
        from replay_adapter import teacher_forced_nll
        logits=torch.tensor([[[0.,2.],[3.,0.],[0.,1.]]])
        labels=torch.tensor([[-100,1,0]])
        r=teacher_forced_nll(logits,labels)
        expected=(torch.nn.functional.cross_entropy(logits[0,:2],torch.tensor([1,0]))).item()
        self.assertAlmostEqual(r['answer_nll'],expected,places=6)
        self.assertEqual(r['answer_token_count'],2)


if __name__=='__main__':unittest.main(verbosity=2)
