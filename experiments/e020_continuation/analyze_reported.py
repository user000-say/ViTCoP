"""Audit ONLY user-reported aggregate results. No invented task values/TTFT."""
import json
from pathlib import Path
from budget import profile


def analyze():
    result={'reported_R4':{
        'ONE':{'COVER':95.21,'BASE':95.22},'EARLY':{'COVER':96.10,'BASE':96.10},
        'LATE':{'COVER':96.24,'BASE':95.77},'LATE_WMATCH':{'COVER':95.04,'BASE':94.77},
        'FIRST_SHIFT':{'COVER':96.50,'BASE':95.50},'THREE_WMATCH':{'COVER':96.31,'BASE':96.11}},
        'new_TTFT':None,'new_per_task_scores':None,
        'budget_assumption':{'depth':32,'initial':172,'convention':'zero_based_after_block',
                             'status':'consistent reconstruction; project must verify actual event semantics'}}
    schedules={'EARLY':[[4,102],[12,32]],'LATE':[[4,102],[16,32]],
               'LATE_WMATCH':[[4,78],[16,32]],'FIRST_SHIFT':[[2,116],[12,32]],
               'THREE_WMATCH':[[4,103],[8,101],[12,32]]}
    result['conditional_budget_profiles']={k:profile(32,172,v,'zero_based_after_block') for k,v in schedules.items()}
    result['first_shift_entrance_schedule_interaction_R4_pp']=(96.50-96.10)-(95.50-96.10)
    result['limitations']=['Rounding may move deltas by 0.01 pp.',
        'No paired significance claim without actual prediction records.',
        'Fixed schedules do not establish per-input adaptive allocation.',
        'The new comparison baseline is COVER FIRST_SHIFT, not merely EARLY.']
    return result


if __name__=='__main__':
    r=analyze();Path('reported_analysis.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
    print(json.dumps({k:v['visual_token_layers'] for k,v in r['conditional_budget_profiles'].items()}))
