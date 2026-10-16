"""Gold-only diagnostic ceiling for two ACTUAL prediction files. Not a policy.

Helps decide whether sample-wise schedule/selector complementarity exists at
all. Recomputes POPE F1 and MME paired score through paired_stats.prepare.
No same-latency/average-cost or deployability claim follows from this oracle.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from paired_stats import prepare
from build_matrix import TASKS


def compare(a,b,pope_metric='f1'):
    oracle={};overlap={}
    for task in TASKS:
        if len({str(r['id']) for r in a[task]})!=len(a[task]) or len({str(r['id']) for r in b[task]})!=len(b[task]):raise ValueError('duplicate question IDs')
        aa={str(r['id']):r for r in a[task]};bb={str(r['id']):r for r in b[task]}
        if aa.keys()!=bb.keys():raise ValueError('strategy question sets differ for '+task)
        counts={'both_correct':0,'only_A_correct':0,'only_B_correct':0,'both_wrong':0};rows=[]
        for key,x in aa.items():
            y=bb[key]
            if str(x['image_id'])!=str(y['image_id']):raise ValueError('image identity differs')
            if task=='POPE':
                if x['label']!=y['label']:raise ValueError('POPE truth differs')
                cx,cy=int(x['prediction']==x['label']),int(y['prediction']==y['label'])
            else:cx,cy=int(x['correct']),int(y['correct'])
            if task=='MME' and x['category']!=y['category']:raise ValueError('MME category differs')
            name='both_correct' if cx and cy else ('only_A_correct' if cx else ('only_B_correct' if cy else 'both_wrong'))
            counts[name]+=1;rows.append(copy.deepcopy(y if cy>cx else x))
        overlap[task]={'questions':len(aa),**counts};oracle[task]=rows
    def score(data):
        ids,fn=prepare(data,pope_metric);return fn(np.ones(len(ids)))
    sa,sb,so=score(a),score(b),score(oracle)
    return {'A':sa,'B':sb,'gold_querywise_oracle':so,'overlap':overlap,
            'oracle_R4_minus_best_fixed_R4':so['R4']-max(sa['R4'],sb['R4']),
            'limits':'Uses gold correctness per question. Does not establish that an online feature can predict the choice; costs may differ and are not matched here.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('A');p.add_argument('B');p.add_argument('--out',required=True)
    p.add_argument('--pope-metric',choices=['f1','accuracy'],required=True);a=p.parse_args()
    r=compare(json.loads(Path(a.A).read_text()),json.loads(Path(a.B).read_text()),a.pope_metric)
    Path(a.out).write_text(json.dumps(r,ensure_ascii=False,indent=2));print(r['oracle_R4_minus_best_fixed_R4'])
