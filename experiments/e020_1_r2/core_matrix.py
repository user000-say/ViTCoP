"""E020.1 R2: 12 core factorial + 4 strong-point diagnostic configurations.
Only current-state weights are prerequisite; history is an optional ablation.
The legacy 72-arm builder is archival, not the default execution route.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from build_matrix import build,from_configs,digest,job_digest
from scorer_numpy import PairScorer
from policy_protocol import DEFAULT_POLICY,POLICY_SCHEMA,validate_model_policy


def identify(config):
    identity={k:v for k,v in config.items() if k not in ('config_id','config_hash','block','budget')}
    config['config_id']=config['stage1']+'__'+config['schedule_name']+'__'+config['selector_name']
    config['config_hash']=digest(identity)
    return config


def trace_path(runtime,phase,task,panel,config_id):
    root=Path(runtime.get('trace_directory',Path(runtime['model_directory']).parent/'e020_traces')).resolve()
    return str(root/phase/task/panel['ids_hash']/(config_id+'.jsonl'))


def attach_trace_dependencies(matrix,runtime):
    existing={j['job_id'] for j in matrix['jobs']}
    for job in matrix['jobs']:
        c=job['config'];kind=c['selector']['kind']
        if kind=='continuation':
            job['selector_trace_file']=trace_path(runtime,matrix['phase'],job['task'],job['panel'],c['config_id'])
        if kind=='matched_random':
            ref=c['stage1']+'__'+c['schedule_name']+'__cf_current_g1_r1'
            dependency=ref+'__'+job['task']
            if dependency not in existing:raise ValueError('matched control missing its learned reference job')
            job['depends_on']=[dependency]
            job['matched_trace_input']=trace_path(runtime,matrix['phase'],job['task'],job['panel'],ref)
            job['matched_trace_reference_config_id']=ref
        job['job_hash']=job_digest(job)
    return matrix


def learned_spec(runtime,mode):
    path=Path(runtime['model_directory'])/(mode+'.npz')
    if not path.is_file():return None
    _,meta=PairScorer.load(path)
    if meta.get('source_hash')!=digest(runtime['source']) or meta.get('mode')!=mode:
        raise ValueError('checkpoint source/mode differs from runtime: '+mode)
    if runtime['kind']=='real':
        if meta.get('training_kind')!='real':raise ValueError('synthetic weights cannot run real ACC')
        validate_model_policy(meta)
        forbidden=runtime.get('forbidden_evaluation_image_ids',[])
        if isinstance(forbidden,str):forbidden=json.loads(Path(forbidden).read_text())
        touched=set(map(str,meta.get('trained_image_ids',[])+meta.get('selection_image_ids',[])
                        +meta.get('diagnostic_test_image_ids',[])))
        if touched&set(map(str,forbidden)):raise ValueError('evaluation overlaps fitting, selection or diagnostic images')
    policy=meta.get('deployment_policy',DEFAULT_POLICY)
    return {'kind':'continuation','model':mode,**policy,'policy_schema':POLICY_SCHEMA,
            'threshold':meta.get('decision_threshold',0.),'accept_enabled':meta.get('accept_enabled',True),
            'model_path':str(path.resolve()),'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def build_core(runtime,phase='screen',allow_missing_models=False):
    if runtime.get('kind')=='real' and not runtime.get('schedules',{}).get('FIRST_NEXT_WMATCH'):
        raise ValueError('export actual NEXT events; the revised core never infers the incumbent from W')
    old=build(runtime,phase,'independent')
    lookup={c['config_id']:c for c in old['configurations']}
    configs=[];missing=[];specs={}
    for mode in ('current','full','no_context','shuffled'):
        try:specs[mode]=learned_spec(runtime,mode)
        except (ValueError,KeyError) as e:
            if mode=='current' and not allow_missing_models:raise
            specs[mode]=None;missing.append(mode+': '+str(e))
        if specs[mode] is None:missing.append(mode)
    if specs['current'] is None and not allow_missing_models:
        raise ValueError('CORE NOT READY: fitted/calibrated current-state scorer required')
    def add(ref,name,spec):
        c=copy.deepcopy(ref);c.update(selector_name=name,selector=copy.deepcopy(spec));configs.append(identify(c))
    for arm in ('COVER','BASE'):
        for schedule in ('FIRST_NEXT_WMATCH','FIRST_SHIFT'):
            ref=copy.deepcopy(lookup[arm+'__'+schedule+'__reference']);configs.append(ref)
            if specs['current'] is not None:
                add(ref,'cf_current_g1_r1',specs['current'])
                add(ref,'matched_random_current',{'kind':'matched_random','shortlist':16,'seed':20,'candidate_limit':16})
    ref=lookup['COVER__FIRST_NEXT_WMATCH__reference']
    for mode in ('full','no_context','shuffled'):
        if specs[mode] is not None:add(ref,'cf_'+mode+'_g1_r1',specs[mode])
    add(ref,'e019_current',{'kind':'coverage_exchange','alpha':0.,'lambda':.15,'rounds':4})
    def priority(c):
        kind=c['selector']['kind']
        tier=0 if kind=='reference' else 1 if kind=='continuation' and c['selector']['model']=='current' else 2 if kind=='matched_random' else 3
        return tier,c['schedule_name'],c['selector_name'],c['stage1']
    configs.sort(key=priority)
    result=attach_trace_dependencies(from_configs(runtime,configs,phase),runtime)
    result.update(missing_models=sorted(set(missing)),core_hypothesis_tested=False,
                  declared_full_capacity={'configurations':16,'task_jobs':64},
                  primary_comparison='COVER NEXT current vs NEXT REF, matched random, E019CUR; true TTFT')
    return result


def confirm_matrix(runtime,screen,state):
    # Fixed primary contrasts are retained regardless of development ranks.
    lookup={c['config_id']:c for c in screen['configurations']}
    names=[a+'__'+s+'__reference' for s in ('FIRST_NEXT_WMATCH','FIRST_SHIFT') for a in ('COVER','BASE')]
    names+=['COVER__FIRST_NEXT_WMATCH__'+n for n in ('cf_current_g1_r1','matched_random_current','e019_current')]
    names+=['BASE__FIRST_NEXT_WMATCH__cf_current_g1_r1']
    chosen=[lookup[n] for n in names if n in lookup]
    available=build(runtime,'confirm','independent')
    chosen.append(next(c for c in available['configurations'] if c['config_id']=='BASE__FIRST_KPLUS8__reference'))
    from measurement_anchors import external_anchors
    chosen+=external_anchors(runtime)
    return attach_trace_dependencies(from_configs(runtime,chosen,'confirm'),runtime)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runtime');p.add_argument('--out',required=True)
    p.add_argument('--phase',default='screen');p.add_argument('--allow-missing-models',action='store_true')
    a=p.parse_args();r=build_core(json.loads(Path(a.runtime).read_text()),a.phase,a.allow_missing_models)
    Path(a.out).write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r['counts']))
