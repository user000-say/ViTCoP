"""Only the user's pasted aggregates; remote status MD was not read here."""
import json
from math import fsum
from pathlib import Path

DENOM={'GQA':61.9,'POPE':85.8,'MME':1852.,'MMBench':64.1}
METRICS={
 'COVER_EARLY':{'GQA':57.57,'POPE':84.42,'MME':1767.5,'MMBench':62.54},
 'BASE_EARLY':{'GQA':57.18,'POPE':84.03,'MME':1769.9,'MMBench':63.14},
 'COVER_FIRST_SHIFT':{'GQA':57.88,'POPE':84.61,'MME':1768.5,'MMBench':63.06},
 'BASE_FIRST_SHIFT':{'GQA':57.14,'POPE':84.54,'MME':1715.5,'MMBench':63.14},
 'COVER_THREE':{'GQA':57.62,'POPE':84.50,'MME':1767.2,'MMBench':62.97},
 'BASE_THREE':{'GQA':57.12,'POPE':84.17,'MME':1774.3,'MMBench':62.97}}


def contributions(a,b):return {t:25*(METRICS[a][t]-METRICS[b][t])/DENOM[t] for t in DENOM}


def analyze():
    gap=contributions('COVER_FIRST_SHIFT','BASE_FIRST_SHIFT')
    old=contributions('COVER_EARLY','BASE_EARLY')
    interaction={t:gap[t]-old[t] for t in DENOM}
    return {'source':'Pasted user report 2026-10-16 plus explicit historical EARLY task values. Not server file access.',
            'task_values':METRICS,'R4_recomputed_from_displayed_values':{k:25*fsum(v[t]/DENOM[t] for t in DENOM) for k,v in METRICS.items()},
            'COVER_FIRST_vs_EARLY_R4_contributions':contributions('COVER_FIRST_SHIFT','COVER_EARLY'),
            'BASE_FIRST_vs_EARLY_R4_contributions':contributions('BASE_FIRST_SHIFT','BASE_EARLY'),
            'COVER_vs_BASE_FIRST_R4_contributions':gap,'entrance_schedule_interaction_contributions':interaction,
            'MME_fraction_of_FIRST_gap':gap['MME']/fsum(gap.values()),
            'MME_fraction_of_interaction':interaction['MME']/fsum(interaction.values()),
            'new_best_reported':{'config':'COVER_FIRST_NEXT_WMATCH_REF','R4':96.67,'per_task_values':None,'TTFT':None},
            'BASE_SHIFT_control_deltas':{'E019CUR':-.06,'E019TMP':-.15,'RANDR2':.13,'RANDR4':.31,'RANDG4R1':.36,'RANDG4R2':.44,'RDRIFT':.54,'KPLUS8':.84},
            'COVER_SHIFT_control_deltas':{'E019CUR':.01,'E019TMP':-.27,'CSCORE':0.},
            'learning_module_tested':False,'reported_learned_jobs':0,
            'limitations':['Rounded metrics need not reproduce the last reported 0.01 exactly.',
              'R4 equality does not establish selected-set or prediction equality.',
              'Different random settings are not independent RNG seeds.',
              'No new TTFT, significance, saturation or state-adaptive-budget claim.']}


if __name__=='__main__':
    r=analyze();Path('analysis_20261016.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
    print(json.dumps(r['COVER_vs_BASE_FIRST_R4_contributions']))
