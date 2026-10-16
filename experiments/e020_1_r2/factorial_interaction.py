"""Paired 2x2 entrance x schedule interaction, with official task rescoring.

Four prediction JSONs share question identities. One image weight is used for
all four arms. This measures an aggregate interaction, not its mechanism.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from paired_stats import prepare
from build_matrix import TASKS
from summarize import DENOM


def interaction(base_a,cover_a,base_b,cover_b,*,pope_metric,replicates=2000,seed=20):
    datasets=[base_a,cover_a,base_b,cover_b]
    for task in TASKS:
        maps=[{str(r['id']):r for r in d[task]} for d in datasets]
        if any(len(m)!=len(d[task]) for m,d in zip(maps,datasets)):raise ValueError('duplicate question IDs')
        if any(m.keys()!=maps[0].keys() for m in maps[1:]):raise ValueError('unpaired 2x2 question sets')
        for key in maps[0]:
            for m in maps[1:]:
                x,y=maps[0][key],m[key]
                if str(x['image_id'])!=str(y['image_id']):raise ValueError('image identity differs')
                if task=='POPE' and x['label']!=y['label']:raise ValueError('POPE labels differ')
                if task=='MME' and x['category']!=y['category']:raise ValueError('MME categories differ')
    prepared=[prepare(d,pope_metric) for d in datasets]
    ids=prepared[0][0]
    if any(item[0]!=ids for item in prepared):raise ValueError('global image identities differ')
    def evaluate(w):
        scores=[fn(w) for _,fn in prepared]
        return {t:(scores[3][t]-scores[2][t])-(scores[1][t]-scores[0][t]) for t in TASKS+['R4']}
    point=evaluate(np.ones(len(ids)));rng=np.random.default_rng(seed);samples=[]
    for _ in range(replicates):
        try:samples.append(list(evaluate(rng.poisson(1.,len(ids))).values()))
        except ValueError:continue
    if len(samples)<max(20,replicates//2):raise ValueError('too few valid image bootstrap replicates')
    samples=np.asarray(samples)
    return {'interaction_cover_minus_base_schedule_B_minus_A':point,
            'paired_image_95pct_interval':{t:np.quantile(samples[:,j],[.025,.975]).tolist() for j,t in enumerate(point)},
            'R4_contribution_by_task':{t:25*point[t]/DENOM[t] for t in TASKS},
            'image_clusters':len(ids),'valid_replicates':len(samples),
            'limits':'Conditional on these configurations and task mix. Does not identify why the interaction occurs or correct selection bias.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base_a');p.add_argument('cover_a');p.add_argument('base_b');p.add_argument('cover_b')
    p.add_argument('--pope-metric',choices=['f1','accuracy'],required=True);p.add_argument('--out',required=True)
    a=p.parse_args();files=(a.base_a,a.cover_a,a.base_b,a.cover_b)
    r=interaction(*(json.loads(Path(f).read_text()) for f in files),pope_metric=a.pope_metric)
    Path(a.out).write_text(json.dumps(r,ensure_ascii=False,indent=2))
