"""Regression checks for the E020.1 dependency/measurement fixes; no VLM data."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import numpy as np
from checks import runtime_fixture,random_model,HERE
from build_matrix import build,digest
from core_matrix import build_core,confirm_matrix
from core_progress import inspect
from features import INPUT_DIM,FEATURE_SCHEMA
from selector_audit import make_trace,append_trace,load_targets,matched_random_numpy,jitter_numpy
from scorer_numpy import image_split


def weights(runtime):
    for mode in ('current','full','no_context','shuffled'):
        random_model().save(Path(runtime['model_directory'])/(mode+'.npz'),
                            {'mode':mode,'source_hash':digest(runtime['source'])})


class ClosureChecks(unittest.TestCase):
    def test_official_mme_categories_and_gold_overlap_ceiling(self):
        from audit_mme import compare as mme
        from strategy_overlap import compare as overlap
        a={};b={}
        for task in ('GQA','POPE','MME','MMBench'):
            a[task]=[{'id':str(i),'image_id':str(i//2),'correct':i%2,'category':'toy',
                      'label':i%2,'prediction':i%2 if i%2 else 1-i%2} for i in range(40)]
            b[task]=[{**row,'correct':1-row['correct'],'prediction':1-row['prediction']} for row in a[task]]
        r=overlap(a,b,'f1');self.assertGreater(r['oracle_R4_minus_best_fixed_R4'],0)
        self.assertAlmostEqual(r['gold_querywise_oracle']['MME'],200.)
        mm=mme(a['MME'],b['MME']);self.assertEqual(mm['delta_MME'],0.)
        with self.assertRaises(ValueError):mme(a['MME']+[a['MME'][0]],b['MME'])

    def test_core_matrix_and_strict_missing_dependency_status(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            r=runtime_fixture(d)
            with self.assertRaises(ValueError):build_core(r)
            self.assertFalse(inspect(r)['core_hypothesis_tested'])
            weights(r);m=build_core(r)
            self.assertEqual(m['counts'],{'configurations':16,'task_jobs':64})
            self.assertEqual(len([j for j in m['jobs'] if j.get('depends_on')]),16)
            self.assertTrue(inspect(r)['primary_models_ready'])
            learned=[j for j in m['jobs'] if j['config']['selector']['kind']=='continuation']
            one=inspect(r,[],{'states':{learned[0]['job_id']:'done'}},m)
            self.assertTrue(one['any_learned_job_executed']);self.assertFalse(one['core_hypothesis_tested'])
            cid=learned[0]['config']['config_id']
            done={j['job_id']:'done' for j in learned if j['config']['config_id']==cid}
            progress=inspect(r,[],{'states':done},m)
            self.assertTrue(progress['any_learned_four_task_result'])
            self.assertFalse(progress['core_hypothesis_tested'])
            self.assertFalse(progress['validated_algorithm_gain'])

    def test_actual_exported_next_boundary_is_not_replaced(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            r=runtime_fixture(d);r['schedules']['FIRST_NEXT_WMATCH']=[[3,110],[12,32]]
            m=build(r,only='independent')
            c=next(c for c in m['configurations'] if c['config_id']=='COVER__FIRST_NEXT_WMATCH__reference')
            self.assertEqual(c['events'],[[4,110],[13,32]])
            self.assertEqual(c['budget']['visual_token_layers'],2286)

    def test_trace_matching_identity_and_final_change_count(self):
        score=np.arange(24.);base=np.arange(12,24);ids=np.arange(100,124)
        chosen=np.r_[np.arange(12,22),10,11]
        r=make_trace(request_key='toy_q',config_id='CURRENT',source_hash='TEST',
                     baseline_ids=ids[base],selected_ids=ids[chosen],eligible_ids=ids,hook_called=True,scorer_sha256='TEST')
        self.assertEqual(r['net_changes'],2)
        got=matched_random_numpy(score,base,ids,r,'toy_q')
        self.assertEqual(len(got),12);self.assertEqual(len(set(got)-set(base)),2)
        np.testing.assert_array_equal(got,matched_random_numpy(score,base,ids,r,'toy_q'))
        self.assertFalse(np.array_equal(got,matched_random_numpy(score,base,ids,r,'toy_q',seed=21)))
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            path=Path(d)/'trace.jsonl';append_trace(path,r)
            self.assertIn('toy_q',load_targets(path,'CURRENT','TEST'))
            with self.assertRaises(ValueError):load_targets(path,'WRONG','TEST')
        with self.assertRaises(ValueError):matched_random_numpy(score,base,ids,r,'wrong_q')

    def test_jitter_cardinality_and_sample_determinism(self):
        score=np.arange(172.);base=np.arange(56,172)
        a=jitter_numpy(score,base,'toy_image_q')
        b=jitter_numpy(score,base,'toy_image_q')
        np.testing.assert_array_equal(a,b);self.assertEqual(len(a),116);self.assertEqual(len(set(a)),116)
        self.assertTrue(set(range(72,172))<=set(a))

    def test_group_specific_oracle_and_genuine_noop_noise(self):
        from diagnostics import diagnose
        image=next('test_'+str(i) for i in range(1000) if image_split(['test_'+str(i)])[2][0])
        records=[]
        for g in (1,4):
            records.append({'image_id':image,'prefix_hash':'SAME_PREFIX','donor_patch_ids':list(range(g)),
                            'incoming_patch_ids':list(range(10,10+g)),'stage1':'COVER','schedule_name':'FIRST_SHIFT',
                            'proposal_shortlist':16})
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d)/'pairs.npz'
            np.savez(p,kind='synthetic',schema=FEATURE_SCHEMA,pairs=np.zeros((2,2,INPUT_DIM)),labels=np.array([.1,1.]),
                     records=np.array([json.dumps(r) for r in records]),
                     noop_records=np.array([json.dumps({'actually_recomputed':True,'nll_difference':.00002})]))
            one=diagnose([p],d,True,group_size=1);four=diagnose([p],d,True,group_size=4)
            self.assertAlmostEqual(one['sampled_one_step_oracle_positive_gain'],.1)
            self.assertAlmostEqual(four['sampled_one_step_oracle_positive_gain'],1.)
            self.assertEqual(one['noop_replay_noise']['actually_recomputed'],1)

    def test_confirmation_retains_new_best_and_base_extra_work_anchor(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            r=runtime_fixture(d);weights(r);m=build_core(r)
            results={j['job_id']:{'metric_value':90. if j['task']!='MME' else 1700.} for j in m['jobs']}
            confirm=confirm_matrix(r,m,{'results':results})
            ids={c['config_id'] for c in confirm['configurations']}
            self.assertIn('COVER__FIRST_NEXT_WMATCH__reference',ids)
            self.assertIn('BASE__FIRST_KPLUS8__reference',ids)
            self.assertIn('COVER__FIRST_NEXT_WMATCH__matched_random_current',ids)
            self.assertIn('COVER__FIRST_NEXT_WMATCH__cf_current_g1_r1',ids)
            self.assertLessEqual(len(ids),10)
            from workflow import extra_matrix
            r['timing_panel']={'panel':{'path':'synthetic_only','ids_hash':'TEST'},'task':'TTFT'}
            timing=extra_matrix(confirm,r,'timing')
            self.assertFalse(any(j['config']['selector']['kind']=='matched_random' for j in timing['jobs']))

    def test_missing_collection_is_reported_without_launching_a_sweep(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            root=Path(d);r=runtime_fixture(root);(root/'runtime.json').write_text(json.dumps(r))
            (root/'bridge.json').write_text(json.dumps({'commands':{}}))
            result=subprocess.run([sys.executable,str(HERE/'close_core.py'),str(root/'runtime.json'),
                                   '--bridge-config',str(root/'bridge.json'),'--out',str(root/'out'),'--allow-synthetic'],
                                  capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,2)
            self.assertFalse(json.loads((root/'out/core_blocked.json').read_text())['core_hypothesis_tested'])
            self.assertFalse((root/'out/core_screen').exists())

    def test_real_queue_dependencies_and_graceful_drain(self):
        from run_matrix import run,validate_result
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            root=Path(d);r=runtime_fixture(root);m=build(r,only='independent');jobs=m['jobs'][:3]
            for j in jobs:j['estimated_seconds']=.01
            child=root/'child.py';child.write_text('''import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]);j=json.loads((p/'job.json').read_text())
time.sleep(.2)
r=p/'predictions.json';r.write_text('[]')
(p/'result.json').write_text(json.dumps({'job_hash':j['job_hash'],'metric_key':j['metric_key'],
 'panel_ids_hash':j['panel']['ids_hash'],'metric_value':1.,'records_path':str(r),'fixture_kind':'synthetic'}))
''')
            class Bridge:
                def command(self,j,out):return [sys.executable,str(child),out]
                def read_result(self,j,out):return json.loads((Path(out)/'result.json').read_text())
            parent,dependent=jobs[:2];dependent['depends_on']=[parent['job_id']]
            mm={**m,'jobs':[dependent,parent]}
            with contextlib.redirect_stdout(io.StringIO()):
                result=run(mm,Bridge(),root/'deps',('0','2'),seconds=5,reserve=0,allow_synthetic=True)
            self.assertTrue(all(s=='done' for s in result['states'].values()))
            drain=threading.Event();timer=threading.Timer(.1,drain.set);timer.start()
            # Independent jobs for the drain test: only the already-running one finishes.
            jobs[1].pop('depends_on',None)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    rr=run({**m,'jobs':jobs},Bridge(),root/'drain',('0',),seconds=5,reserve=0,
                           allow_synthetic=True,drain_event=drain)
            finally:timer.cancel()
            self.assertEqual(sum(s=='done' for s in rr['states'].values()),1)
            self.assertEqual(sum(s=='pending' for s in rr['states'].values()),2)
            fake={**parent,'config':{'selector':{'kind':'continuation','model_sha256':'ACTUAL_WEIGHT_HASH'}}}
            result=next(iter(result['results'].values())).copy()
            result.update(job_hash=fake['job_hash'],metric_key=fake['metric_key'],panel_ids_hash=fake['panel']['ids_hash'])
            with self.assertRaises(ValueError):validate_result(result,fake)
            result['scorer_sha256']='ACTUAL_WEIGHT_HASH';validate_result(result,fake)


if __name__=='__main__':unittest.main(verbosity=2)
