import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch=None

from make_catalog import MAIN, SHAPES, TASKS, build_catalog, digest
from elva.cover_binding import NativeCover, select_checked, origin
from elva.workflow import (validate_binding, blocked_reason, result_key, validate_result,
                           collect_complete, freeze, validate_decision, trace_for)
from native_result_io import finish_task
from gpu_resources import GPU, Lease, parse_inventory, choose, respect_visible
from partition_manifest import partition
from run_queue import run


def fixture_cover(*, metric_features, scores, patch_ids, k, context):
    # Fixture callback deliberately uses a different subset from score Top-K.
    return np.arange(k, dtype=np.int64)


def fixture_torch_cover(*, metric_features, scores, patch_ids, k, context):
    return torch.arange(k,device=scores.device,dtype=torch.long)


def fixture_spec(family="legacy_native", supported=(172,)):
    source=Path(__file__).resolve()
    return dict(family=family, supported_k=list(supported), source_files={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},
                replacement_slots={"172":32,"60":11}, legacy172_parity_verified=True)


def fixture_binding():
    from elva.low_budget_cover import make_spec
    protected = make_spec()
    protected['implementation_parity_verified'] = True  # FIXTURE ONLY
    return dict(fixture=True, source_commit="FIXTURE", model_sha="FIXTURE", processor_sha="FIXTURE",
                tokenizer_sha="FIXTURE", prompt_sha="FIXTURE", dataset_sha="FIXTURE", scorer_sha="FIXTURE",
                backend="FIXTURE", cache_mode="native", original_tokens=576, depth=32,
                native_selectors={s:"FIXTURE" for s in ("Q_ATTN","RDRIFT_NATIVE","QED_NATIVE","ATTN_COVER_NATIVE")},
                preflight=dict(off_path_equal=True, clip_layers_checked=True, pen_payload_equal=True,
                               positions_and_kv_checked=True,
                               trace_verified={"2048":True,"1024":True}),
                task_samples={task:1 for task in TASKS},
                native_cover={"legacy_native":fixture_spec(),
                              "protected_coverage":protected,
                              "parameterized_native":fixture_spec("parameterized_native",(60,))})


def fixture_records(c,b):
    m=c["method"]
    family=m["stage1"]["cover_family"]
    k=m["schedule"]["counts"][0]
    if family=="none":
        proof=dict(cover_family=family, requested_k=k,native_calls=0,cover_calls=0,
                   origin=origin(family),replacement_slots=0,fallback=False,
                   payload="pen",cover_source_sha=None,native_k=None)
    else:
        from elva.cover_binding import spec_fingerprint
        spec=b['native_cover'][family]
        native_k=172 if family=="legacy_pool172_then_topk" else k
        slots=m['stage1'].get('coverage_slots') if family=='protected_coverage' else spec['replacement_slots'][str(native_k)]
        proof=dict(cover_family=family,requested_k=k,cover_calls=1,
                   native_calls=int(family!='protected_coverage'),origin=origin(family),
                   replacement_slots=slots,fallback=False,payload="pen",
                   cover_source_sha=spec_fingerprint(b["native_cover"][family]),
                   native_k=native_k)
    return [dict(image_key="FIXTURE_IMAGE",question_key="FIXTURE_QUESTION",prediction="FIXTURE",
                 stage1_ids=list(range(k)),stage1_audit=proof,trace=trace_for(c))]


class BudgetTests(unittest.TestCase):
    def test_only_two_budgets_and_exact_inverse(self):
        from fractions import Fraction
        configs=build_catalog()
        self.assertEqual({c["method"]["budget"] for c in configs},{2048,1024})
        self.assertEqual(sum(c["phase"]=="stage1_core" for c in configs),10)
        for budget, shapes in SHAPES.items():
            for p in shapes:
                self.assertEqual(p.work,budget)
                self.assertEqual(p.depth,32)
                self.assertEqual((1-p.compression(576))*576*32,budget)
                self.assertEqual(p.counts[0],MAIN[budget].counts[0])
                self.assertEqual(p.counts[-1],MAIN[budget].counts[-1])
        self.assertEqual(MAIN[2048].events,((4,90),(12,32)))
        self.assertEqual(MAIN[1024].events,((10,28),(20,12)))

    def test_no_unconditional_stage2_or_learning_and_explicit_pool_identity(self):
        configs=build_catalog()
        self.assertFalse(any(c["phase"]=="stage2" or c["method"]["learned"] for c in configs))
        pooled=build_catalog(low_cover="legacy_pool172_then_topk")
        a=next(c for c in configs if 'W1024_V4_' in c['id'])
        b=next(c for c in pooled if 'W1024_POOL172_CONTROL_V4_' in c['id'])
        self.assertNotEqual(a['config_hash'],b['config_hash'])
        self.assertIn('POOL172',b['id'])
        self.assertEqual(b['phase'],'diagnostic_pool')
        self.assertEqual(sum(c['phase']=='stage1_core' for c in pooled),10)


class CoverTests(unittest.TestCase):
    def setUp(self):
        self.x=np.arange(576*3,dtype=float).reshape(576,3)
        self.s=np.linspace(0,1,576)
        self.ids=np.arange(576)*7+5
        self.config=next(c for c in build_catalog() if 'W2048_V4_' in c['id'])

    def test_missing_cover_is_an_error_not_topk(self):
        with self.assertRaises(RuntimeError):
            select_checked(self.x,self.s,self.ids,self.config)

    def test_native172_calls_source_and_preserves_raw_pen_payload(self):
        cover=NativeCover(fixture_cover,fixture_spec())
        out=select_checked(self.x,self.s,self.ids,self.config,native_cover=cover)
        np.testing.assert_array_equal(out['payload'],self.x[out['keep']])
        np.testing.assert_array_equal(out['keep'],np.arange(172))
        self.assertEqual(out['stage1_audit']['native_calls'],1)
        self.assertEqual(cover.events[0]['replacement_slots'],32)

    def test_fixed172_cannot_claim_direct_k60(self):
        cover=NativeCover(fixture_cover,fixture_spec())
        with self.assertRaises(RuntimeError):
            cover(metric_features=self.x,scores=self.s,patch_ids=self.ids,k=60)
        self.assertEqual(cover.events,[])

    def test_explicit_pool_route_calls172_then_returns60(self):
        cover=NativeCover(fixture_cover,fixture_spec('legacy_pool172_then_topk',(60,)))
        c=next(c for c in build_catalog(low_cover='legacy_pool172_then_topk') if 'W1024_POOL172_CONTROL_V4_' in c['id'])
        out=select_checked(self.x,self.s,self.ids,c,native_cover=cover)
        np.testing.assert_array_equal(out['keep'],np.arange(112,172))
        np.testing.assert_array_equal(out['payload'],self.x[out['keep']])
        self.assertEqual(out['stage1_audit']['native_k'],172)
        self.assertEqual(out['stage1_audit']['requested_k'],60)

    def test_tail_replacement_then_trim_can_be_identically_topk60(self):
        n=576
        scores=np.arange(n,dtype=float)[::-1]
        ids=np.arange(n)
        original_top172=np.arange(172)
        pool=np.r_[original_top172[:140],np.arange(200,232)]
        trimmed=pool[np.argsort(-scores[pool],kind='stable')[:60]]
        np.testing.assert_array_equal(trimmed,np.arange(60))

    def test_changed_native_source_is_blocked(self):
        spec=fixture_spec()
        spec['source_files'][str(Path(__file__).resolve())]='0'*64
        with self.assertRaises(RuntimeError):
            NativeCover(fixture_cover,spec)

    def test_cover_can_legitimately_return_same_ids_as_topk(self):
        # Equal sets alone are not proof that the callback was skipped.
        cover=NativeCover(fixture_cover,fixture_spec())
        out=select_checked(self.x,self.s[::-1].copy(),self.ids,self.config,native_cover=cover)
        self.assertEqual(out['stage1_audit']['native_calls'],1)
        np.testing.assert_array_equal(out['keep'],np.arange(172))


@unittest.skipIf(torch is None,'PyTorch is absent locally; execute this server-side')
class TorchCoverTests(unittest.TestCase):
    def test_native_and_pooled_wrapper_preserve_torch_payload_and_ids(self):
        x=torch.arange(576*3,dtype=torch.float32).reshape(576,3)
        s=torch.linspace(0,1,576)
        ids=torch.arange(576,dtype=torch.long)*7+5
        high=next(c for c in build_catalog() if 'W2048_V5_' in c['id'])
        cover=NativeCover(fixture_torch_cover,fixture_spec())
        out=select_checked(x,s,ids,high,early=x.flip(0),early_scores=s.flip(0),early_ids=ids,
                           native_cover=cover,backend='torch')
        torch.testing.assert_close(out['payload'],x.index_select(0,out['keep']),rtol=0,atol=0)
        torch.testing.assert_close(out['ids'],ids.index_select(0,out['keep']),rtol=0,atol=0)
        low=next(c for c in build_catalog(low_cover='legacy_pool172_then_topk') if 'W1024_POOL172_CONTROL_V4_' in c['id'])
        pool=NativeCover(fixture_torch_cover,fixture_spec('legacy_pool172_then_topk',(60,)))
        out=select_checked(x,s,ids,low,native_cover=pool,backend='torch')
        torch.testing.assert_close(out['keep'],torch.arange(112,172),rtol=0,atol=0)
        self.assertEqual(out['stage1_audit']['native_k'],172)


class GPUAndWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.binding=fixture_binding()
        self.configs=build_catalog()
        self.config=next(c for c in self.configs if 'W2048_V1_' in c['id'])

    def tearDown(self):
        self.temp.cleanup()

    def make_result(self,c,task):
        key=result_key(c,self.binding,task)
        folder=self.root/key
        value=finish_task(folder/'result.json',records=fixture_records(c,self.binding),official_metrics={'FIXTURE':1.},
                          config=c,binding=self.binding,task=task,run_key=key,fixture=True)
        return value,folder,key

    def test_zero_utilization_is_not_an_empty_gpu(self):
        devices=parse_inventory('0, GPU-a, 10, 0\n2, GPU-b, 14632, 0\n4, GPU-c, 5, 0\n', 'GPU-a, 2050863\n')
        self.assertFalse(devices[0].available())
        self.assertFalse(devices[1].available())
        self.assertTrue(devices[2].available())
        acc,unused=choose(devices,['auto'],3,0)
        self.assertEqual([g.index for g in acc],['4'])
        self.assertEqual(choose(devices,['auto'],3,1)[0],[])

    def test_gpu_mapping_unknown_or_na_is_not_assumed_free(self):
        for g,p in [('0, GPU-a, N/A, 0\n',''),('0, GPU-a, 5, 0\n','GPU-b, 1\n')]:
            with self.assertRaises(ValueError): parse_inventory(g,p)
        with self.assertRaises(ValueError): choose([GPU('0','GPU-a',0,0,())],['0','0'],2,0)

    def test_inherited_gpu_allocation_is_respected(self):
        devices=[GPU('0','GPU-a',0,0,()),GPU('7','GPU-b',0,0,())]
        self.assertEqual(respect_visible(devices,'7'),[devices[1]])
        self.assertEqual(respect_visible(devices,'GPU-a'),[devices[0]])
        self.assertEqual(respect_visible(devices,''),[])
        with self.assertRaises(ValueError):respect_visible(devices,'MIG-unknown')

    def test_global_limit_precedes_sharding_and_image_dedup(self):
        rows=[dict(image_key=f'FIXTURE_{i}',question_key=f'Q{q}') for i in range(40) for q in range(2)]
        groups=partition(rows,shards=4,limit=32)
        self.assertEqual(list(map(len,groups)),[8,8,8,8])
        self.assertEqual(len({r['image_key'] for group in groups for r in group}),32)
        with self.assertRaises(ValueError):partition(rows+rows[:1],shards=4,limit=32,mode='vlm_question')

    def test_lease_is_exclusive_and_released(self):
        with Lease(self.root,'GPU-a'):
            with self.assertRaises(BlockingIOError):
                with Lease(self.root,'GPU-a'): pass
        with Lease(self.root,'GPU-a'): pass

    def test_fixture_and_missing_native_capacity_are_blocked(self):
        with self.assertRaises(ValueError):validate_binding(self.binding)
        validate_binding(self.binding,allow_fixture=True)
        c=next(c for c in self.configs if 'W1024_V4_' in c['id'])
        b=copy.deepcopy(self.binding);del b['native_cover']['protected_coverage']
        self.assertIn('BLOCKED_COVER',blocked_reason(c,b))
        self.assertIsNone(blocked_reason(self.config,b))

    def test_per_sample_result_proof_rejects_clip_pilot_and_wrong_trace(self):
        value,folder,key=self.make_result(self.config,'GQA')
        validate_result(value,self.config,self.binding,'GQA',key,folder,allow_fixture=True)
        bad=copy.deepcopy(value);bad['inference_kind']='clip_selection_only'
        with self.assertRaises(ValueError):validate_result(bad,self.config,self.binding,'GQA',key,folder,allow_fixture=True)
        records=fixture_records(self.config,self.binding)
        records[0]['trace'][0]-=1
        with self.assertRaises(ValueError):
            finish_task(folder/'wrong.json',records=records,official_metrics={'FIXTURE':1.},config=self.config,
                        binding=self.binding,task='GQA',run_key=key,fixture=True)

    def test_fake_cover_fallback_is_rejected(self):
        c=next(c for c in self.configs if 'W2048_V4_' in c['id'])
        rows=fixture_records(c,self.binding)
        rows[0]['stage1_audit']['native_calls']=0
        with self.assertRaises(ValueError):
            finish_task(self.root/'bad/result.json',records=rows,official_metrics={'FIXTURE':1.},config=c,
                        binding=self.binding,task='GQA',run_key=result_key(c,self.binding,'GQA'),fixture=True)

    def test_complete_core_freeze_and_changed_results_invalidate(self):
        with self.assertRaises(OSError):collect_complete(self.configs,self.binding,self.root,2048,allow_fixture=True)
        for c in self.configs:
            if c['phase']=='stage1_core' and c['method']['budget']==2048:
                for task in TASKS:self.make_result(c,task)
        decision=freeze(self.configs,self.binding,self.root,2048,self.config['id'],'FIXTURE rationale',allow_fixture=True)
        validate_decision(decision,self.binding,self.configs,self.root,2048,allow_fixture=True)
        generated=build_catalog(decisions=decision)
        self.assertEqual(sum(c['phase']=='stage2' for c in generated),8)
        path=self.root/result_key(self.config,self.binding,'GQA')/'result.json'
        value=json.loads(path.read_text());value['metrics']['FIXTURE']=.5;path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):validate_decision(decision,self.binding,self.configs,self.root,2048,allow_fixture=True)

    def test_completed_optional_beta_can_be_frozen_after_core(self):
        for c in self.configs:
            if c['method']['budget']==2048:
                for task in TASKS:self.make_result(c,task)
        winner=next(c for c in self.configs if c['method']['budget']==2048 and c['method']['stage1']['beta']==.25)
        decision=freeze(self.configs,self.binding,self.root,2048,winner['id'],'FIXTURE optional beta',allow_fixture=True)
        validate_decision(decision,self.binding,self.configs,self.root,2048,allow_fixture=True)
        self.assertEqual(decision['budgets']['2048']['stage1']['beta'],.25)

    def test_queue_runs_resumes_and_rejects_stage2_before_freeze(self):
        script=self.root/'fixture_runner.py'
        script.write_text('''import argparse,json\nfrom pathlib import Path\nfrom native_result_io import finish_task\nfrom elva.workflow import trace_for\np=argparse.ArgumentParser()\nfor name in ('config','binding','task','out','run_key'):p.add_argument('--'+name,required=True)\na=p.parse_args()\nc=json.loads(Path(a.config).read_text());b=json.loads(Path(a.binding).read_text())\nk=c['method']['schedule']['counts'][0]\naudit=dict(cover_family='none',requested_k=k,native_calls=0,cover_calls=0,origin='none',replacement_slots=0,fallback=False,payload='pen',cover_source_sha=None,native_k=None)\nrows=[dict(image_key='FIXTURE_IMAGE',question_key='FIXTURE_Q',prediction='FIXTURE',stage1_ids=list(range(k)),stage1_audit=audit,trace=trace_for(c))]\nfinish_task(a.out,records=rows,official_metrics={'FIXTURE':1.},config=c,binding=b,task=a.task,run_key=a.run_key,fixture=True)\n''')
        command=[sys.executable,str(script),'--config','{config}','--binding','{binding}',
                 '--task','{task}','--out','{out}','--run_key','{run_key}']
        probe=lambda:[GPU('0','GPU-FIXTURE',0,0,())]
        import os
        old=os.environ.get('PYTHONPATH')
        os.environ['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
        try:
            kwargs=dict(phase='stage1_core',requested=['auto'],workers=1,reserve=0,hours=.02,
                        tasks=['GQA'],budgets=[2048],gpu_probe=probe,allow_fixture=True,
                        lease_directory=self.root/'leases')
            states=run({'configs':[self.config]},self.binding,command,self.root/'queue',**kwargs)
            self.assertTrue(any(s['status']=='DONE' for s in states),states)
            states=run({'configs':[self.config]},self.binding,command,self.root/'queue',**kwargs)
            self.assertTrue(any(s['status']=='REUSED_VERIFIED' for s in states))
            kwargs['phase']='stage2'
            with self.assertRaises(ValueError):run({'configs':[self.config]},self.binding,command,self.root/'queue',**kwargs)
        finally:
            if old is None:os.environ.pop('PYTHONPATH',None)
            else:os.environ['PYTHONPATH']=old


if __name__=='__main__':unittest.main()
