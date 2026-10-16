"""R4 uses task-specific official scores, never an average of incomplete tasks."""
import json
from collections import defaultdict
from build_matrix import TASKS

DENOM={'GQA':61.9,'POPE':85.8,'MME':1852.,'MMBench':64.1}


def aggregate(matrix,state):
    rows=defaultdict(dict);config={c['config_id']:c for c in matrix['configurations']}
    for job in matrix['jobs']:
        result=state.get('results',{}).get(job['job_id'])
        if result and job['task'] in TASKS:
            rows[job['config']['config_id']][job['task']]=float(result['metric_value'])
    output=[]
    for key,metrics in rows.items():
        row={'config_id':key,'metrics':metrics,'complete':all(t in metrics for t in TASKS)}
        if row['complete']:row['R4']=25*sum(metrics[t]/DENOM[t] for t in TASKS)
        row['config']=config[key];output.append(row)
    output.sort(key=lambda r:r.get('R4',-1),reverse=True)
    lookup={r['config_id']:r for r in output}
    for row in output:
        arm=row['config']['stage1'];ref=lookup.get(arm+'__FIRST_SHIFT__reference')
        if ref and ref['complete'] and row['complete']:
            row['delta_R4_vs_same_arm_FIRST_SHIFT']=row['R4']-ref['R4']
            row['task_delta_vs_FIRST_SHIFT']={t:row['metrics'][t]-ref['metrics'][t] for t in TASKS}
        next_ref=lookup.get(arm+'__FIRST_NEXT_WMATCH__reference')
        if next_ref and next_ref['complete'] and row['complete']:
            row['delta_R4_vs_same_arm_FIRST_NEXT']=row['R4']-next_ref['R4']
        global_ref=lookup.get('COVER__FIRST_NEXT_WMATCH__reference')
        if global_ref and global_ref['complete'] and row['complete']:
            row['delta_R4_vs_COVER_FIRST_NEXT']=row['R4']-global_ref['R4']
    return {'rows':output,'completed_four_task_configs':sum(r['complete'] for r in output),
            'phase':matrix['phase'],'development_only':matrix['phase']=='screen'}


def promote(runtime,matrices,states,top=4):
    from build_matrix import from_configs
    complete=[];allconfigs={}
    for m,s in zip(matrices,states):
        complete.extend(r for r in aggregate(m,s)['rows'] if r['complete'])
        allconfigs.update({c['config_id']:c for c in m['configurations']})
    complete.sort(key=lambda r:r['R4'],reverse=True)
    anchors=[a+'__'+sch+'__reference' for sch in ('FIRST_SHIFT','EARLY') for a in ('COVER','BASE')]
    chosen=[allconfigs[k] for k in anchors if k in allconfigs]
    added=[]
    def pick(predicate):
        for row in complete:
            c=row['config']
            if predicate(c) and c['config_id'] not in {x['config_id'] for x in chosen+added}:
                added.append(c);return
    # Predeclared diversity: both arms' best learned module, then best
    # training-free alternative and one strongest remaining challenger.
    for arm in ('COVER','BASE'):
        pick(lambda c:c['stage1']==arm and c['selector']['kind']=='continuation')
    pick(lambda c:c['selector']['kind'] not in ('continuation','reference'))
    while len(added)<top:
        n=len(added);pick(lambda c:True)
        if len(added)==n:break
    chosen+=added[:top]
    return from_configs(runtime,chosen,'confirm')


if __name__=='__main__':
    import argparse
    from pathlib import Path
    a=argparse.ArgumentParser();a.add_argument('matrix');a.add_argument('state');a.add_argument('--out',required=True)
    x=a.parse_args();result=aggregate(json.loads(Path(x.matrix).read_text()),json.loads(Path(x.state).read_text()))
    Path(x.out).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(result['completed_four_task_configs'])
