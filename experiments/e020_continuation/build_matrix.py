"""72 distinct configurations / 288 four-task screen jobs, when ONE is exported.

Controls run without a fitted model. Learned jobs are never run with random or
missing weights. This catalogue capacity is not a promise of six-hour throughput.
"""
import argparse
import hashlib
import json
from pathlib import Path
from budget import profile, matched_two_event, canonical

TASKS = ['GQA','POPE','MME','MMBench']


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def job_digest(job):
    return digest({k:v for k,v in job.items() if k not in ('job_hash','job_id','estimated_seconds')})


def runtime_digest(runtime):
    return digest({k:v for k,v in runtime.items() if k not in ('estimated_job_seconds',)})


def schedule_specs(runtime):
    depth,n0=runtime['model_depth'],runtime['initial_visual_tokens']
    conv=runtime['layer_convention']; schedules={}; missing=[]
    for name,events in runtime['schedules'].items():
        if events is None: missing.append(name);continue
        schedules[name]=canonical(events,depth,conv)
    for key in ('EARLY','FIRST_SHIFT','THREE_WMATCH','LATE','LATE_WMATCH'):
        if key not in schedules: raise ValueError('export actual '+key+' schedule')
    first=schedules['FIRST_SHIFT'];early=schedules['EARLY']
    if len(first)!=2 or len(early)!=2: raise ValueError('export two-event FIRST_SHIFT and EARLY')
    target=profile(depth,n0,early)['visual_token_layers'];end,k=first[-1]
    # Local allocation neighbourhood, not an unbounded layer hyperparameter scan.
    for after,name in [(first[0][0]-1,'FIRST_PREV_WMATCH'),(first[0][0]+1,'FIRST_NEXT_WMATCH')]:
        if 0<=after<end:
            schedules[name]=matched_two_event(depth,n0,target,after,end,k)
    for delta,name in [(-8,'FIRST_KMINUS8'),(8,'FIRST_KPLUS8')]:
        new=first[0][1]+delta
        if k<=new<=n0:schedules[name]=[[first[0][0],new],list(first[-1])]
    budgets={name:profile(depth,n0,ev) for name,ev in schedules.items()}
    for name,b in budgets.items():
        b['delta_visual_work_vs_EARLY']=b['visual_token_layers']-target
        b['relative_visual_work_vs_EARLY']=b['visual_token_layers']/target
    for name,expected in runtime.get('expected_visual_work',{}).items():
        if name in budgets and expected is not None and budgets[name]['visual_token_layers']!=expected:
            raise ValueError('layer convention/work mismatch for '+name)
    return schedules,budgets,missing


def rules():
    out={'reference':{'kind':'reference'}}
    for model in ('current','full'):
        for g,r in [(1,2),(1,4),(4,1),(4,2)]:
            out[f'cf_{model}_g{g}_r{r}']={'kind':'continuation','model':model,'group_size':g,
                                         'rounds':r,'shortlist':16,'threshold':0.}
    for model in ('no_context','shuffled'):
        out[f'cf_{model}_g1_r4']={'kind':'continuation','model':model,'group_size':1,
                                 'rounds':4,'shortlist':16,'threshold':0.}
    out.update({
        'random_r2':{'kind':'random','group_size':1,'rounds':2,'shortlist':16,'seed':20},
        'random_r4':{'kind':'random','group_size':1,'rounds':4,'shortlist':16,'seed':20},
        'random_g4_r1':{'kind':'random','group_size':4,'rounds':1,'shortlist':16,'seed':20},
        'random_g4_r2':{'kind':'random','group_size':4,'rounds':2,'shortlist':16,'seed':20},
        'current_score':{'kind':'current_score'},
        'rank_drift_control':{'kind':'rank_drift','current_weight':.6},
        'e019_current':{'kind':'coverage_exchange','alpha':0.,'lambda':.15,'rounds':4},
        'e019_temporal':{'kind':'coverage_exchange','alpha':.5,'lambda':.15,'rounds':4},
    })
    return out


def validate_runtime(runtime,phase):
    if runtime.get('schema')!='e020_runtime_v1' or runtime.get('kind') not in ('real','synthetic'):
        raise ValueError('runtime must be exported as e020_runtime_v1 and kind real')
    for field in ('checkpoint_hash','code_commit','prompt_hash','input_pipeline_hash',
                  'cache_policy','query_rule','feature_protocol_hash'):
        if not runtime.get('source',{}).get(field):raise ValueError('missing source identity '+field)
    for t in TASKS:
        panel=runtime.get('panels',{}).get(phase,{}).get(t,{})
        if not panel.get('path') or not panel.get('ids_hash') or not runtime['metric_keys'].get(t):
            raise ValueError('export actual panel identity/metric for '+t)


def build(runtime,phase='screen',only='all',allow_missing_models=False):
    validate_runtime(runtime,phase)
    schedules,budgets,missing=schedule_specs(runtime); configs=[]; seen=set(); unavailable=[]
    choices=rules()
    def add(arm,schedule,rule,block):
        spec=dict(choices[rule]); learned=spec['kind']=='continuation'
        if only=='independent' and learned or only=='learned' and not learned:return
        if learned:
            path=Path(runtime['model_directory'])/(spec['model']+'.npz')
            if not path.is_file():
                unavailable.append(str(path))
                if allow_missing_models:return
                raise ValueError('missing fitted model '+str(path))
            import numpy as np
            with np.load(path,allow_pickle=False) as data:
                meta=json.loads(str(data['metadata']))
            from features import FEATURE_SCHEMA
            if meta.get('schema')!=FEATURE_SCHEMA:raise ValueError('fitted scorer feature schema mismatch')
            if meta.get('source_hash')!=digest(runtime['source']) or meta.get('mode')!=spec['model']:
                raise ValueError('fitted scorer source/mode does not match this runtime')
            if runtime['kind']=='real' and meta.get('training_kind')!='real':
                raise ValueError('synthetic scorer cannot be submitted as a real experiment')
            forbidden=runtime.get('forbidden_evaluation_image_ids',[])
            if isinstance(forbidden,str):forbidden=json.loads(Path(forbidden).read_text())
            if set(map(str,forbidden))&set(map(str,meta.get('trained_image_ids',[]))):
                raise ValueError('fitted scorer trained on forbidden evaluation image IDs')
            spec['model_path']=str(path.resolve());spec['model_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        identity={'stage1':arm,'schedule_name':schedule,'events':schedules[schedule],
                  'selector_name':rule,'selector':spec,'application':'first_pruning_event_only',
                  'continuation_horizon_after':schedules[schedule][-1][0],
                  'final_event_selector':'original_project_selector_unchanged'}
        h=digest(identity)
        if h not in seen:
            seen.add(h);configs.append({'config_id':arm+'__'+schedule+'__'+rule,
                                        'config_hash':h,'block':block,'budget':budgets[schedule],**identity})
    for arm in ('COVER','BASE'):
        for name in schedules:add(arm,name,'reference','A_allocation')
        for rule,spec in choices.items():
            if spec['kind']=='continuation':add(arm,'FIRST_SHIFT',rule,'B_continuation')
        for name in ('EARLY','THREE_WMATCH','FIRST_KMINUS8','FIRST_KPLUS8'):
            for rule in ('cf_full_g1_r4','cf_full_g4_r2'):add(arm,name,rule,'C_transfer')
        for rule,spec in choices.items():
            if spec['kind'] not in ('continuation','reference'):add(arm,'FIRST_SHIFT',rule,'D_controls')
    # Pair arms and prioritize actual best incumbent and primary module.
    def priority(c):
        if c['selector_name']=='reference' and c['schedule_name'] in ('FIRST_SHIFT','EARLY','THREE_WMATCH'):tier=0
        elif c['selector_name'] in ('cf_full_g1_r4','cf_current_g1_r4') and c['schedule_name']=='FIRST_SHIFT':tier=1
        elif c['block']=='D_controls':tier=2
        elif c['block']=='C_transfer':tier=3
        elif c['block']=='B_continuation':tier=4
        else:tier=5
        return tier,c['schedule_name'],c['selector_name'],c['stage1']
    configs.sort(key=priority)
    matrix=from_configs(runtime,configs,phase)
    matrix.update(missing_schedules=missing,unavailable_models=sorted(set(unavailable)))
    return matrix


def from_configs(runtime,configs,phase):
    jobs=[]
    for c in configs:
        for t in TASKS:
            job={'phase':phase,'kind':runtime['kind'],'task':t,'config':c,
                 'panel':runtime['panels'][phase][t],'source':runtime['source'],
                 'metric_key':runtime['metric_keys'][t],
                 'estimated_seconds':runtime.get('estimated_job_seconds',{}).get(t,120)}
            job['job_hash']=job_digest(job);job['job_id']=c['config_id']+'__'+t;jobs.append(job)
    return {'schema':'e020_matrix_v1','phase':phase,'kind':runtime['kind'],
            'runtime_hash':runtime_digest(runtime),'configurations':configs,'jobs':jobs,
            'counts':{'configurations':len(configs),'task_jobs':len(jobs)}}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('runtime');ap.add_argument('--out',required=True)
    ap.add_argument('--phase',default='screen');ap.add_argument('--only',choices=['all','independent','learned'],default='all')
    a=ap.parse_args();result=build(json.loads(Path(a.runtime).read_text()),a.phase,a.only)
    Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result['counts']))
