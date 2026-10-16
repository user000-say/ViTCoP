"""Recompute MME category acc+acc_plus and paired error transitions."""
import argparse
import json
from pathlib import Path


def compare(a,b):
    if len({str(r['id']) for r in a})!=len(a) or len({str(r['id']) for r in b})!=len(b):raise ValueError('duplicate MME question IDs')
    a={str(r['id']):r for r in a};b={str(r['id']):r for r in b}
    if a.keys()!=b.keys():raise ValueError('MME question IDs differ')
    categories={}
    for key,aa in a.items():
        bb=b[key]
        if aa['category']!=bb['category'] or str(aa['image_id'])!=str(bb['image_id']):raise ValueError('MME identity/category changed')
        if aa['correct'] not in (0,1) or bb['correct'] not in (0,1):raise ValueError('binary official correctness required')
        categories.setdefault(aa['category'],[]).append((aa,bb))
    rows=[]
    for category,pairs in categories.items():
        images={};flips={'wrong_to_right':0,'right_to_wrong':0}
        for aa,bb in pairs:
            images.setdefault(str(aa['image_id']),[]).append((aa['correct'],bb['correct']))
            flips['wrong_to_right']+=aa['correct']==0 and bb['correct']==1
            flips['right_to_wrong']+=aa['correct']==1 and bb['correct']==0
        if any(len(v)!=2 for v in images.values()):raise ValueError('MME requires complete question pairs')
        score=lambda side:100*sum(x[side] for vals in images.values() for x in vals)/(2*len(images))+100*sum(all(x[side] for x in vals) for vals in images.values())/len(images)
        sa,sb=score(0),score(1)
        rows.append({'category':category,'images':len(images),'questions':len(pairs),
                     'baseline_score':sa,'candidate_score':sb,'delta':sb-sa,
                     'R4_pp_contribution':25*(sb-sa)/1852.,**flips})
    rows.sort(key=lambda r:abs(r['delta']),reverse=True)
    return {'categories':rows,'baseline_MME':sum(r['baseline_score'] for r in rows),
            'candidate_MME':sum(r['candidate_score'] for r in rows),
            'delta_MME':sum(r['delta'] for r in rows),'R4_pp_contribution':sum(r['R4_pp_contribution'] for r in rows)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('candidate');p.add_argument('--out',required=True)
    a=p.parse_args()
    def load(path):
        value=json.loads(Path(path).read_text());return value['MME'] if isinstance(value,dict) else value
    r=compare(load(a.baseline),load(a.candidate));Path(a.out).write_text(json.dumps(r,ensure_ascii=False,indent=2))
    print(json.dumps({'delta_MME':r['delta_MME'],'R4_pp_contribution':r['R4_pp_contribution']}))
