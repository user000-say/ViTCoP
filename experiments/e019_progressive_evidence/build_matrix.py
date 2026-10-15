"""Generate distinct model configurations AND task jobs from actual runtime.

48 configurations x four tasks if all near-work schedules are feasible.
No synthetic schedule is silently substituted for an unavailable project config.
"""
import argparse
import hashlib
import json
from pathlib import Path
from schedules import variants

TASKS=['GQA','POPE','MME','MMBench']


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def runtime_digest(runtime):
    return digest({k:v for k,v in runtime.items() if k!='estimated_job_seconds'})


def job_digest(job):
    return digest({k:v for k,v in job.items() if k not in ('estimated_seconds','job_hash','job_id')})


def selectors():
    return {
        'reference':{'mode':'reference','alpha':0.,'lambda':0.,'swaps':0},
        'current_exchange':{'mode':'exchange','alpha':0.,'lambda':.15,'swaps':4},
        'temporal_a25':{'mode':'exchange','alpha':.25,'lambda':.15,'swaps':4},
        'temporal_a50':{'mode':'exchange','alpha':.5,'lambda':.15,'swaps':4},
        'temporal_a75':{'mode':'exchange','alpha':.75,'lambda':.15,'swaps':4},
        'previous_only':{'mode':'exchange','alpha':1.,'lambda':.15,'swaps':4},
        'temporal_l05':{'mode':'exchange','alpha':.5,'lambda':.05,'swaps':4},
        'temporal_l30':{'mode':'exchange','alpha':.5,'lambda':.30,'swaps':4},
        'temporal_b2':{'mode':'exchange','alpha':.5,'lambda':.15,'swaps':2},
        'scalar_ema':{'mode':'mean','alpha':.5,'lambda':0.,'swaps':0},
        'current_mean':{'mode':'mean','alpha':0.,'lambda':0.,'swaps':0},
        'score_topk':{'mode':'score','alpha':0.,'lambda':0.,'swaps':0},
        'random_swap':{'mode':'random','alpha':0.,'lambda':0.,'swaps':4},
    }


def build(runtime,phase='screen'):
    if runtime.get('schema')!='e019_runtime_v1':raise ValueError('wrong runtime schema')
    if runtime.get('kind') not in ('real','synthetic'):raise ValueError('runtime kind missing')
    for field in ['checkpoint_hash','code_commit','prompt_hash','input_pipeline_hash','cache_policy','query_rule']:
        if not runtime.get('source',{}).get(field):raise ValueError('source identity missing: '+field)
    for task in TASKS:
        if not runtime.get('metric_keys',{}).get(task):raise ValueError('freeze metric keys')
        panel=runtime.get('panels',{}).get(phase,{}).get(task,{})
        if not panel.get('path') or not panel.get('ids_hash'):raise ValueError('freeze exact panel IDs for '+task)
    schedules,annotations,skipped=variants(runtime['model_depth'],runtime['initial_visual_tokens'],runtime['schedules'])
    for name,expected in runtime.get('expected_visual_work',{}).items():
        if annotations[name]['visual_token_layers']!=expected:
            raise ValueError('actual completed-block schedule disagrees with exported work: '+name)
    configs={}; rules=selectors()
    def add(arm,schedule,selector,block):
        identity={'stage1':arm,'schedule_name':schedule,'events':schedules[schedule],
                  'selector_name':selector,'selector':rules[selector],
                  'application':'final_event_only','query_groups':4}
        key=digest(identity)
        if key not in configs:
            configs[key]={'config_id':arm+'__'+schedule+'__'+selector,
                          'block':block,'config_hash':key,**identity,
                          'budget':annotations[schedule]}
    for arm in ['BASE','COVER']:
        for name in schedules:add(arm,name,'reference','A_schedule')
        for selector in rules:
            if selector!='reference':add(arm,'EARLY',selector,'B_selector')
        for name in ['LATE_WMATCH','FIRST_SHIFT_WMATCH','THREE_WMATCH']:
            if name in schedules:
                for selector in ['current_exchange','temporal_a50']:
                    add(arm,name,selector,'C_joint')
    schedule_order={name:i for i,name in enumerate(schedules)}
    selector_order={name:i for i,name in enumerate(rules)}
    def priority(config):
        name,selector=config['schedule_name'],config['selector_name']
        if selector=='reference' and name in ('ONE','EARLY','LATE'):tier=0
        elif name=='EARLY' and selector in ('current_exchange','temporal_a50'):tier=1
        elif selector=='reference':tier=2
        elif config['block']=='C_joint':tier=3
        else:tier=4
        return tier,schedule_order[name],selector_order[selector],config['stage1']
    # Pair BASE/COVER before spending the whole screen budget on one entrance.
    ordered=sorted(configs.values(),key=priority); jobs=[]
    for config in ordered:
        for task in TASKS:
            job={'phase':phase,'kind':runtime['kind'],'task':task,'config':config,
                 'panel':runtime['panels'][phase][task],'source':runtime['source'],
                 'metric_key':runtime['metric_keys'][task],
                 'estimated_seconds':runtime.get('estimated_job_seconds',{}).get(task,180)}
            job['job_hash']=job_digest(job)
            job['job_id']=config['config_id']+'__'+task
            jobs.append(job)
    return {'schema':'e019_matrix_v1','phase':phase,'kind':runtime['kind'],
            'runtime_hash':runtime_digest(runtime),'configurations':ordered,'jobs':jobs,
            'skipped_schedules':skipped,'counts':{'configurations':len(ordered),'task_jobs':len(jobs)}}


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('runtime'); ap.add_argument('--out',required=True)
    ap.add_argument('--phase',default='screen')
    args=ap.parse_args()
    result=build(json.loads(Path(args.runtime).read_text()),args.phase)
    Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result['counts']))
