"""Paired image-cluster Poisson bootstrap, with task-specific rescoring.

Input JSON maps task -> list of raw per-question records. Fields:
id, image_id, correct (0/1) for GQA/MMBench/MME; category for MME;
label and prediction (0/1) for POPE. Optional global_image_id couples resampling
across tasks containing the same image. Missing scores/records are errors.
MME requires both questions of each image/category and recomputes acc+acc_plus.
POPE F1 is recomputed from confusion counts, not mean per-example F1.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from summarize import DENOM
from build_matrix import TASKS


def prepare(data,pope_metric):
    if pope_metric not in ('f1','accuracy'):raise ValueError('freeze POPE metric explicitly')
    prepared={};global_ids=set()
    for task in TASKS:
        records=data[task]
        if not records or len({str(r['id']) for r in records})!=len(records):
            raise ValueError('missing/duplicate question records for '+task)
        for r in records:
            global_ids.add(str(r.get('global_image_id',task+':'+str(r['image_id']))))
    universe=sorted(global_ids);lookup={s:i for i,s in enumerate(universe)}
    for task in TASKS:
        rows=sorted(data[task],key=lambda r:str(r['id']))
        ix=np.asarray([lookup[str(r.get('global_image_id',task+':'+str(r['image_id'])))] for r in rows])
        for r in rows:
            fields=('label','prediction') if task=='POPE' else ('correct',)
            if any(r[f] not in (0,1) for f in fields):raise ValueError('binary official scoring fields required')
        if task=='POPE':
            y=np.asarray([r['label'] for r in rows]);p=np.asarray([r['prediction'] for r in rows])
            prepared[task]=(ix,{'tp':y*p,'fp':(1-y)*p,'fn':y*(1-p),'ok':(y==p).astype(float)})
        elif task=='MME':
            groups={}
            for r,i in zip(rows,ix):groups.setdefault((r['category'],str(r['image_id'])),[]).append((r['correct'],i))
            if any(len(g)!=2 or g[0][1]!=g[1][1] for g in groups.values()):
                raise ValueError('MME requires two questions per same image/category')
            cats={}
            for (cat,_),group in groups.items():
                cats.setdefault(cat,[]).append((group[0][1],sum(a for a,_ in group)/2,
                                               float(all(a for a,_ in group))))
            prepared[task]={cat:np.asarray(values) for cat,values in cats.items()}
        else:prepared[task]=(ix,np.asarray([r['correct'] for r in rows],dtype=float))
    def score(weights):
        out={}
        for task in TASKS:
            if task=='MME':
                value=0.
                for values in prepared[task].values():
                    w=weights[values[:,0].astype(int)]
                    if w.sum()==0:raise ValueError('empty resampled MME category')
                    value+=100*(w@(values[:,1]+values[:,2]))/w.sum()
                out[task]=value
            else:
                ix,values=prepared[task];w=weights[ix]
                if w.sum()==0:raise ValueError('empty resampled task')
                if task=='POPE' and pope_metric=='f1':
                    tp=w@values['tp'];den=2*tp+w@values['fp']+w@values['fn']
                    out[task]=100*2*tp/den if den else 0.
                elif task=='POPE':out[task]=100*(w@values['ok'])/w.sum()
                else:out[task]=100*(w@values)/w.sum()
        out['R4']=25*sum(out[t]/DENOM[t] for t in TASKS)
        return out
    return universe,score


def compare(baseline,candidate,*,pope_metric,replicates=2000,seed=20):
    for t in TASKS:
        a={str(r['id']):r for r in baseline[t]};b={str(r['id']):r for r in candidate[t]}
        if a.keys()!=b.keys():raise ValueError('paired question set differs for '+t)
        for key in a:
            if str(a[key]['image_id'])!=str(b[key]['image_id']):raise ValueError('paired image identity differs')
            if t=='POPE' and a[key]['label']!=b[key]['label']:raise ValueError('ground truth differs')
            if t=='MME' and a[key]['category']!=b[key]['category']:raise ValueError('MME category differs')
    ids,score_a=prepare(baseline,pope_metric);ids_b,score_b=prepare(candidate,pope_metric)
    if ids!=ids_b:raise ValueError('global image identity differs across arms')
    point_a,point_b=score_a(np.ones(len(ids))),score_b(np.ones(len(ids)))
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(replicates):
        weights=rng.poisson(1.,len(ids))
        try:a,b=score_a(weights),score_b(weights)
        except ValueError:continue
        samples.append([b[t]-a[t] for t in TASKS+['R4']])
    if len(samples)<max(20,replicates//2):raise ValueError('too few valid bootstrap replicates')
    samples=np.asarray(samples)
    return {'baseline':point_a,'candidate':point_b,'delta':{t:point_b[t]-point_a[t] for t in point_a},
            'paired_95pct_interval':{t:np.percentile(samples[:,j],[2.5,97.5]).tolist() for j,t in enumerate(TASKS+['R4'])},
            'valid_replicates':len(samples),'image_clusters':len(ids),
            'note':'Conditional on the chosen pair; use held-out confirmation for selected winners. Not an unconditional post-selection p-value.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('candidate');p.add_argument('--out',required=True)
    p.add_argument('--pope-metric',choices=['f1','accuracy'],required=True);p.add_argument('--replicates',type=int,default=2000)
    a=p.parse_args();r=compare(json.loads(Path(a.baseline).read_text()),json.loads(Path(a.candidate).read_text()),
                             pope_metric=a.pope_metric,replicates=a.replicates)
    Path(a.out).write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r['delta']))
