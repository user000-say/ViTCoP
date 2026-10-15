"""Held-out, same-prefix, one-step candidate diagnostics; NOT ACC or rollout.

The best sampled action bounds headroom only in the sampled proposal catalogue,
not among all possible retained sets. Never sum isolated action gains to claim a
multi-round policy gain, since the intervening deletion context changes.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scorer_numpy import PairScorer,image_split


def diagnose(files,models,allow_synthetic=False):
    x,y,records=[],[],[];seen=set()
    for file in files:
        with np.load(file,allow_pickle=False) as d:
            if str(d['kind'])!='real' and not allow_synthetic:raise ValueError('real paired data required')
            for pair,label,record in zip(d['pairs'],d['labels'],d['records']):
                r=json.loads(str(record))
                key=(r['prefix_hash'],tuple(r['donor_patch_ids']),tuple(r['incoming_patch_ids']))
                if key in seen:continue
                seen.add(key);x.append(pair);y.append(label);records.append(r)
    if not records:raise ValueError('no paired records')
    x,y=np.asarray(x),np.asarray(y)
    images=np.asarray([str(r['image_id']) for r in records]);_,_,test=image_split(images)
    selected=np.flatnonzero(test)
    if not len(selected):raise ValueError('no image-disjoint test actions')
    scores={'current_score_difference':x[:,0,0]-x[:,1,0],
            'rank_drift_difference':x[:,0,10]-x[:,1,10]}
    for name in ('full','current','no_context','shuffled'):
        path=Path(models)/(name+'.npz')
        if path.is_file():
            model,_=PairScorer.load(path);scores[name]=model.predict(x)
    groups=defaultdict(list)
    for i in selected:groups[records[i]['prefix_hash']].append(i)
    report={'test_images':len(set(images[test])),'test_actions':int(test.sum()),
            'test_prefix_states':len(groups),'single_action_label_mean':float(y[test].mean()),
            'positive_label_fraction':float((y[test]>0).mean()),'methods':{}}
    oracle=[max(0.,float(y[ix].max())) for ix in groups.values()]
    report['sampled_one_step_oracle_positive_gain']=float(np.mean(oracle))
    for name,p in scores.items():
        gains=[];selected_actions=[]
        for ix in groups.values():
            j=ix[int(np.argmax(p[ix]))]
            take=p[j]>0
            gains.append(float(y[j]) if take else 0.);selected_actions.append(j if take else None)
        accepted=[j for j in selected_actions if j is not None]
        report['methods'][name]={
            'mean_one_step_real_nll_gain':float(np.mean(gains)),
            'regret_vs_sampled_positive_oracle':float(np.mean(np.asarray(oracle)-gains)),
            'accepted_prefix_fraction':len(accepted)/len(groups),
            'negative_real_gain_among_accepted':float((y[accepted]<0).mean()) if accepted else None,
            'sign_accuracy_actions':float(((p[test]>0)==(y[test]>0)).mean())}
    report['entrance_schedule']={}
    for arm in sorted(set(r['stage1'] for r in records)):
        for schedule in sorted(set(r['schedule_name'] for r in records)):
            ix=[i for i in selected if records[i]['stage1']==arm and records[i]['schedule_name']==schedule]
            if ix:report['entrance_schedule'][arm+'__'+schedule]={'actions':len(ix),'positive_fraction':float((y[ix]>0).mean()),
                                                               'mean_action_nll_gain':float(y[ix].mean())}
    identity=[records[i]['identity_fixed_gain'] for i in selected if 'identity_fixed_gain' in records[i]]
    report['identity_fixed_diagnostic']={'n':len(identity),'mean_gain':float(np.mean(identity)) if identity else None}
    report['limits']='One-step sampled candidate ranking only. No new ACC, TTFT, statistical significance, or multi-round oracle is inferred.'
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('files',nargs='+');p.add_argument('--models',required=True)
    p.add_argument('--out',required=True);p.add_argument('--allow-synthetic',action='store_true')
    a=p.parse_args();result=diagnose(a.files,a.models,a.allow_synthetic)
    Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(result['test_prefix_states'])
