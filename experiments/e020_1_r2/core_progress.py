"""Report dependency completion separately from ACC/TTFT job completion."""
import argparse
import json
from pathlib import Path
import numpy as np
from build_matrix import digest
from features import FEATURE_SCHEMA
from core_matrix import learned_spec


def inspect(runtime,shards=(),state=None,matrix=None):
    weights={};source=digest(runtime['source'])
    for mode in ('current','full','no_context','shuffled'):
        path=Path(runtime['model_directory'])/(mode+'.npz')
        entry={'exists':path.is_file(),'valid':False,'path':str(path)}
        if path.is_file():
            try:
                with np.load(path,allow_pickle=False) as d:
                    meta=json.loads(str(d['metadata']))
                    entry.update(valid=meta.get('source_hash')==source and meta.get('schema')==FEATURE_SCHEMA and
                                 meta.get('mode')==mode and (runtime.get('kind')!='real' or meta.get('training_kind')=='real'),
                                 training_images=len(meta.get('trained_image_ids',[])),label_count=meta.get('label_count'))
                learned_spec(runtime,mode)
                entry.update(calibration_status=meta.get('calibration',{}).get('status'),
                             accept_enabled=meta.get('accept_enabled'))
            except Exception as e:entry.update(valid=False,error=repr(e))
        weights[mode]=entry
    data=[]
    for path in shards:
        entry={'path':str(path),'exists':Path(path).is_file()}
        if entry['exists']:
            try:
                with np.load(path,allow_pickle=False) as d:
                    records=[json.loads(str(r)) for r in d['records']]
                    entry.update(kind=str(d['kind']),paired_actions=len(d['labels']),
                                 images=len({str(r['image_id']) for r in records}),
                                 schedules=sorted({r['schedule_name'] for r in records}),
                                 arms=sorted({r['stage1'] for r in records}))
            except Exception as e:entry['error']=repr(e)
        data.append(entry)
    learned_total=learned_done=current_done=full_done=0;complete={}
    if matrix and state:
        for job in matrix['jobs']:
            if job.get('config',{}).get('selector',{}).get('kind')!='continuation':continue
            learned_total+=1
            if state.get('states',{}).get(job['job_id'])=='done':
                learned_done+=1
                current_done+=job['config']['selector'].get('model')=='current'
                full_done+=job['config']['selector'].get('model')=='full'
                complete.setdefault(job['config']['config_id'],set()).add(job['task'])
    completed_configs=sorted(k for k,tasks in complete.items() if {'GQA','POPE','MME','MMBench'}<=tasks)
    ready=weights['current']['valid']
    primary=['COVER__FIRST_NEXT_WMATCH__'+name for name in
             ('reference','cf_current_g1_r1','matched_random_current','e019_current')]
    primary_complete=False
    if matrix and state:
        done={}
        for job in matrix['jobs']:
            if state.get('states',{}).get(job['job_id'])=='done':
                done.setdefault(job['config']['config_id'],set()).add(job['task'])
        primary_complete=all({'GQA','POPE','MME','MMBench'}<=done.get(c,set()) for c in primary)
    return {'fitted_models':weights,'paired_shards':data,'primary_models_ready':ready,
            'learned_jobs_done':learned_done,'learned_jobs_total_in_manifest':learned_total,
            'current_jobs_done':current_done,'full_jobs_done':full_done,
            'any_learned_job_executed':learned_done>0,'completed_four_task_learned_configs':completed_configs,
            'any_learned_four_task_result':bool(completed_configs),
            'primary_acc_comparison_complete':primary_complete,
            'core_hypothesis_tested':primary_complete,
            'evidence_stage':'P_PRELIMINARY_ACC' if completed_configs else 'I_LOCAL_IMPLEMENTATION',
            'validated_algorithm_gain':False,
            'core_status':('PRIMARY_ACC_COMPARISON_COMPLETE' if primary_complete else 'PRELIMINARY_LEARNED_ACC' if completed_configs else 'PARTIAL_LEARNED_TASKS') if learned_done else ('READY_NOT_YET_TESTED' if ready else 'NOT_TESTED_MISSING_VALID_MODELS'),
            'caution':'Completed ACC does not imply a positive effect, independent confirmation, real efficiency, or a paper contribution.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runtime');p.add_argument('--shards',nargs='*',default=[])
    p.add_argument('--state');p.add_argument('--matrix');p.add_argument('--out',required=True)
    a=p.parse_args();load=lambda f:json.loads(Path(f).read_text()) if f else None
    r=inspect(load(a.runtime),a.shards,load(a.state),load(a.matrix))
    Path(a.out).write_text(json.dumps(r,ensure_ascii=False,indent=2));print(r['core_status'])
