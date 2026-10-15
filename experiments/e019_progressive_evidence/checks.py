"""Mathematical and real-subprocess tests, not VLM/GPU performance results."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from evidence_numpy import select,align_previous,group_queries,prepare,objective
from schedules import work,variants
from build_matrix import build
from run_matrix import run
from summarize import aggregate
from promote import promote
from workflow import extra_matrix


def synthetic_runtime():
    return {'schema':'e019_runtime_v1','kind':'synthetic','model_depth':32,
            'initial_visual_tokens':200,
            'schedules':{'ONE':[{'after':8,'keep':40}],
                         'EARLY':[{'after':4,'keep':120},{'after':12,'keep':40}],
                         'LATE':[{'after':4,'keep':120},{'after':16,'keep':40}]},
            # Must match schedules.work / e018_contract.token_work (count prune layer at pre-prune N).
            'expected_visual_work':{'EARLY':2720,'LATE':3040},
            'source':{k:'synthetic-only' for k in ['checkpoint_hash','code_commit','prompt_hash','input_pipeline_hash','cache_policy','query_rule']},
            'metric_keys':{t:'synthetic_'+t for t in ['GQA','POPE','MME','MMBench']},
            'panels':{phase:{t:{'path':'synthetic-panel-not-a-real-dataset','ids_hash':phase+'_'+t} for t in ['GQA','POPE','MME','MMBench']} for phase in ['screen','confirm']},
            'estimated_job_seconds':{t:.05 for t in ['GQA','POPE','MME','MMBench']}}


class FakeBridge:
    def __init__(self,retry_once=False):self.retry_once=retry_once
    def command(self,job,outdir):
        if self.retry_once and job['task']=='POPE' and Path(outdir).name=='attempt_1':
            raise RuntimeError('synthetic one-time launch error')
        result={'job_hash':job['job_hash'],'metric_key':job['metric_key'],
                'panel_ids_hash':job['panel']['ids_hash'],'metric_value':1.,
                'records_path':str(Path(outdir)/'synthetic_records.jsonl')}
        script=('import json,os; from pathlib import Path; '
                f'p=Path({outdir!r}); '
                '(p/"synthetic_records.jsonl").write_text("{\\\"synthetic\\\":true}\\n"); '
                f'(p/"result.json").write_text({json.dumps(result)!r}); '
                'print(os.environ["CUDA_VISIBLE_DEVICES"])')
        return [sys.executable,'-c',script]
    def read_result(self,job,outdir):return json.loads((Path(outdir)/'result.json').read_text())


class Checks(unittest.TestCase):
    def test_query_groups(self):
        result=group_queries(np.array([[1.,0.],[3.,0.],[0.,2.]]),np.array([0,0,1]),3)
        np.testing.assert_array_equal(result,[[2,0],[0,2],[0,0]])

    def test_patch_identity(self):
        p=np.arange(12).reshape(3,4)
        np.testing.assert_array_equal(align_previous(p,[11,22,33,44],[44,11]),p[:,[3,0]])
        with self.assertRaises(ValueError):align_previous(p,[11,22,33,44],[55])

    def test_passthrough(self):
        s=np.array([4.,3.,2.,1.]);a=np.eye(4)
        result,_=select(s,a,np.array([2,0]),alpha=0,lam=0)
        np.testing.assert_array_equal(result,[0,2])
        ranked,_=select(s,a,np.array([2,0]),alpha=0,lam=0,mode='score')
        np.testing.assert_array_equal(ranked,[0,1])

    def test_facet_repair(self):
        s=np.array([1.01,1.,.99,.98]);a=np.array([[1.,1.,0,0],[0,0,1.,1.]])
        result,info=select(s,a,np.array([0,1]),alpha=0,swaps=4)
        self.assertTrue(any(i>=2 for i in result));self.assertEqual(len(result),2)
        self.assertGreater(info['objective_after'],info['objective_before'])

    def test_monotonic_budget_randomized(self):
        rng=np.random.default_rng(19)
        for n,k in [(8,3),(32,8),(172,32)]:
            scores=rng.random(n);cur=rng.random((4,n));prev=rng.random((4,n))
            ids=np.argsort(-scores)[:k]
            result,info=select(scores,cur,ids,previous=prev)
            self.assertEqual(len(set(result)),k)
            self.assertTrue(np.all(np.diff(result)>0))
            self.assertGreaterEqual(info['objective_after']+1e-10,info['objective_before'])
            self.assertLessEqual(info['accepted_swaps'],4)

    def test_empty_rows_and_random_control(self):
        s=np.ones(8);a=np.zeros((4,8));ids=np.array([0,2,4])
        result,info=select(s,a,ids,alpha=0)
        self.assertTrue(np.isfinite(info['objective_after']))
        random,_=select(s,a,ids,alpha=0,mode='random',seed=2)
        self.assertEqual(len(random),3)

    def test_temporal_is_not_scalar_average(self):
        s=np.array([1.01,1.,.99,.98]);a=np.array([[1.,1.,0,0],[0,0,1.,1.]])
        mean,_=select(s,a,[0,1],alpha=0,mode='mean')
        exchange,_=select(s,a,[0,1],alpha=0)
        self.assertFalse(np.array_equal(mean,exchange))

    def test_exchange_matches_exhaustive_one_swap(self):
        rng=np.random.default_rng(6);s=rng.random(10);a=rng.random((3,10));ids=np.array([0,2,4])
        normalized,u,w=prepare(s,a,alpha=0)
        best=objective(ids,normalized,u,w)
        for removed in ids:
            for added in set(range(10))-set(ids):
                proposed=[i for i in ids if i!=removed]+[added]
                best=max(best,objective(proposed,normalized,u,w))
        result,info=select(s,a,ids,alpha=0,swaps=1)
        self.assertAlmostEqual(info['objective_after'],best,places=10)

    def test_deadline_interrupts_owned_child(self):
        class Slow(FakeBridge):
            def command(self,job,outdir):return [sys.executable,'-c','import time;time.sleep(3)']
        matrix=build(synthetic_runtime());matrix['jobs']=matrix['jobs'][:1]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            state=run(matrix,Slow(),folder,seconds=.25,reserve=0,allow_synthetic=True)
            self.assertEqual(list(state['states'].values()),['interrupted'])

    def test_one_failure_does_not_block_other_job(self):
        class Partial(FakeBridge):
            def command(self,job,outdir):
                if job['task']=='GQA':raise RuntimeError('intentional synthetic failure')
                return super().command(job,outdir)
        matrix=build(synthetic_runtime());matrix['jobs']=matrix['jobs'][:2]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            state=run(matrix,Partial(),folder,seconds=2,reserve=0,retries=0,allow_synthetic=True)
            self.assertEqual(state['states'][matrix['jobs'][0]['job_id']],'failed')
            self.assertEqual(state['states'][matrix['jobs'][1]['job_id']],'done')

    def test_schedule_and_matrix(self):
        runtime=synthetic_runtime();matrix=build(runtime)
        self.assertEqual(matrix['counts'],{'configurations':48,'task_jobs':192})
        self.assertEqual(len({j['job_id'] for j in matrix['jobs']}),192)
        self.assertEqual({c['stage1'] for c in matrix['configurations'][:2]},{'BASE','COVER'})
        eta=copy.deepcopy(runtime);eta['estimated_job_seconds']['GQA']=999
        adjusted=build(eta)
        self.assertEqual(matrix['runtime_hash'],adjusted['runtime_hash'])
        self.assertEqual([j['job_hash'] for j in matrix['jobs']],[j['job_hash'] for j in adjusted['jobs']])
        plans,annotation,skipped=variants(32,200,runtime['schedules'])
        self.assertEqual(work(32,200,plans['EARLY']),2560)
        for key in ['LATE_WMATCH','FIRST_SHIFT_WMATCH','THREE_WMATCH']:
            self.assertLessEqual(annotation[key]['visual_token_layers'],2560)
        wrong=copy.deepcopy(runtime);wrong['expected_visual_work']['EARLY']=0
        with self.assertRaises(ValueError):build(wrong)

    def test_supervisor_retry_resume_and_isolation(self):
        matrix=build(synthetic_runtime());matrix['jobs']=matrix['jobs'][:2]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            with self.assertRaises(ValueError):run(matrix,FakeBridge(),folder,seconds=3,reserve=0)
            state=run(matrix,FakeBridge(True),folder,gpus=('fake0','fake1'),seconds=3,reserve=0,allow_synthetic=True)
            self.assertTrue(all(s=='done' for s in state['states'].values()))
            self.assertEqual(state['attempts'][matrix['jobs'][1]['job_id']],2)
            before=dict(state['attempts'])
            again=run(matrix,FakeBridge(),folder,seconds=3,reserve=0,allow_synthetic=True)
            self.assertEqual(before,again['attempts'])
            texts=[p.read_text().strip() for p in Path(folder).glob('*/attempt_*/process.log')]
            self.assertTrue(set(texts).issubset({'fake0','fake1'}))

    def test_summarize_and_promotion(self):
        runtime=synthetic_runtime();matrix=build(runtime);state={'states':{},'results':{}}
        for job in matrix['jobs']:
            state['states'][job['job_id']]='done'
            state['results'][job['job_id']]={'metric_value':{'GQA':58.,'POPE':85.,'MME':1800.,'MMBench':62.}[job['task']]}
        rows=aggregate(matrix,state);self.assertEqual(len(rows),48)
        promoted=promote(matrix,state,runtime,top=4)
        self.assertEqual(promoted['counts'],{'configurations':8,'task_jobs':32})
        runtime['timing_panel']={'path':'synthetic-panel','ids_hash':'synthetic-timing','n':1024}
        self.assertEqual(extra_matrix(promoted,runtime,'timing')['counts']['task_jobs'],8)
        state['states'][matrix['jobs'][0]['job_id']]='failed'
        self.assertEqual(len(aggregate(matrix,state)),47)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch unavailable: server parity needed')
    def test_torch_parity(self):
        import torch
        from evidence_torch import select as gpu_select
        rng=np.random.default_rng(2)
        s=rng.random(70);a=rng.random((4,70));p=rng.random((4,70));ids=np.argsort(-s)[:16].copy()
        cpu,_=select(s,a,ids,previous=p)
        gpu=gpu_select(torch.tensor(s),torch.tensor(a),torch.tensor(ids),previous=torch.tensor(p))
        np.testing.assert_array_equal(cpu,gpu.numpy())


if __name__=='__main__':unittest.main(verbosity=2)
