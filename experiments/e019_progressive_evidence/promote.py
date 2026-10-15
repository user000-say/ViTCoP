"""Promote complementary development candidates, freeze before confirmation."""
import argparse
import json
from pathlib import Path
from build_matrix import build
from summarize import aggregate


def promote(screen,state,runtime,top=4):
    if top<0:raise ValueError('top must be nonnegative')
    rows=aggregate(screen,state)
    candidates=[r for r in rows if not(r['schedule'] in ('ONE','EARLY','LATE') and r['selector']=='reference')]
    if not candidates:raise ValueError('no completed non-reference configurations to promote')
    selections=[]; reasons={}
    rankers=[('highest_R4',sorted(candidates,key=lambda r:-r['R4'])),
             ('best_worst_task_delta',sorted(candidates,key=lambda r:-r.get('worst_normalized_task_delta',-1e9))),
             ('selector_at_fixed_EARLY',sorted([r for r in candidates if r['schedule']=='EARLY'],key=lambda r:-r['R4'])),
             ('near_work_schedule',sorted([r for r in candidates if abs(r['relative_work_difference'])<=.005 and r['schedule']!='EARLY'],key=lambda r:-r['R4']))]
    for reason,ranked in rankers:
        if len(selections)>=top:break
        for row in ranked:
            if row['config_id'] not in selections:
                selections.append(row['config_id']);reasons[row['config_id']]=reason;break
    for row in sorted(candidates,key=lambda r:-r['R4']):
        if len(selections)>=top:break
        if row['config_id'] not in selections:
            selections.append(row['config_id']);reasons[row['config_id']]='remaining_R4_rank'
    anchors=['BASE__EARLY__reference','COVER__EARLY__reference',
             'COVER__LATE__reference','BASE__ONE__reference']
    selected=set(selections+anchors)
    confirm=build(runtime,'confirm')
    confirm['configurations']=[c for c in confirm['configurations'] if c['config_id'] in selected]
    confirm['jobs']=[j for j in confirm['jobs'] if j['config']['config_id'] in selected]
    confirm['counts']={'configurations':len(confirm['configurations']),'task_jobs':len(confirm['jobs'])}
    confirm['promotion']={'candidate_ids':selections,'anchors':anchors,'reasons':reasons,
                          'basis':'development panel only; includes tradeoff candidates, not proven winners'}
    return confirm


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('screen');ap.add_argument('state');ap.add_argument('runtime')
    ap.add_argument('--out',required=True);ap.add_argument('--top',type=int,default=4)
    args=ap.parse_args()
    result=promote(json.loads(Path(args.screen).read_text()),json.loads(Path(args.state).read_text()),
                   json.loads(Path(args.runtime).read_text()),args.top)
    Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result['counts']))
