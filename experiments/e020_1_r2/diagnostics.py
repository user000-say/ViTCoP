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
from policy_protocol import action_identity,state_identity
from policy_calibration import complete_groups,image_summary


def diagnose(files,models,allow_synthetic=False,group_size=1,shortlist=16,candidate_limit=16):
    x,y,records,noop_records=[],[],[],[];seen=set()
    for file in files:
        with np.load(file,allow_pickle=False) as d:
            if str(d['kind'])!='real' and not allow_synthetic:raise ValueError('real paired data required')
            if 'noop_records' in d:noop_records.extend(json.loads(str(r)) for r in d['noop_records'])
            for pair,label,record in zip(d['pairs'],d['labels'],d['records']):
                r=json.loads(str(record))
                key=action_identity(r)
                if key in seen:continue
                seen.add(key);x.append(pair);y.append(label);records.append(r)
    if not records:raise ValueError('no paired records')
    x,y=np.asarray(x),np.asarray(y)
    images=np.asarray([str(r['image_id']) for r in records]);_,_,test=image_split(images)
    eligible=np.asarray([len(r['donor_patch_ids'])==group_size and r.get('proposal_shortlist',shortlist)==shortlist
                         and r.get('candidate_limit',candidate_limit)==candidate_limit for r in records])
    selected=np.flatnonzero(test & eligible)
    if not len(selected):raise ValueError('no image-disjoint test actions')
    scores={'current_score_difference':x[:,0,0]-x[:,1,0],
            'rank_drift_difference':x[:,0,10]-x[:,1,10]}
    for name in ('full','current','no_context','shuffled'):
        path=Path(models)/(name+'.npz')
        if path.is_file():
            model,_=PairScorer.load(path);scores[name]=model.predict(x)
    groups=defaultdict(list)
    for i in selected:groups[state_identity(records[i])].append(i)
    report={'test_images':len(set(images[test])),'test_actions':int(test.sum()),
            'test_prefix_states':len(groups),'single_action_label_mean':float(y[selected].mean()),
            'positive_label_fraction':float((y[selected]>0).mean()),'methods':{}}
    report.update(proposal_group_size=group_size,proposal_shortlist=shortlist,proposal_candidate_limit=candidate_limit,
                  evaluated_actions=len(selected),evaluated_images=len(set(images[selected].tolist())))
    oracle=[max(0.,float(y[ix].max())) for ix in groups.values()]
    report['sampled_one_step_oracle_positive_gain']=float(np.mean(oracle))
    report['complete_deployment_catalogues']=len(complete_groups(records,test & eligible))
    report['random_catalogue_expected_gain']=float(np.mean([y[ix].mean() for ix in groups.values()]))
    report['no_swap_gain']=0.
    for name,p in scores.items():
        threshold=0.;enabled=True
        path=Path(models)/(name+'.npz')
        if path.is_file():
            _,meta=PairScorer.load(path)
            threshold=meta.get('decision_threshold',0.);enabled=meta.get('accept_enabled',True)
        gains=[];selected_actions=[];image_rows=[]
        for ix in groups.values():
            ix=sorted(ix,key=lambda i:records[i].get('catalogue_index',records[i].get('action_index',i)))
            j=ix[int(np.argmax(p[ix]))]
            take=bool(enabled and p[j]>threshold)
            gain=float(y[j]) if take else 0.
            gains.append(gain);selected_actions.append(j if take else None)
            image_rows.append({'image_id':str(records[j]['image_id']),'gain':gain})
        accepted=[j for j in selected_actions if j is not None]
        raw_gains=[]
        for ix in groups.values():
            ix=sorted(ix,key=lambda i:records[i].get('catalogue_index',records[i].get('action_index',i)))
            j=ix[int(np.argmax(p[ix]))]
            raw_gains.append(float(y[j]) if p[j]>0 else 0.)
        report['methods'][name]={
            'mean_one_step_real_nll_gain':float(np.mean(gains)),
            'raw_zero_threshold_mean_gain':float(np.mean(raw_gains)),
            'image_cluster_summary':image_summary(image_rows),
            'frozen_threshold':threshold,'accept_enabled':enabled,
            'regret_vs_sampled_positive_oracle':float(np.mean(np.asarray(oracle)-gains)),
            'accepted_prefix_fraction':len(accepted)/len(groups),
            'negative_real_gain_among_accepted':float((y[accepted]<0).mean()) if accepted else None,
            'sign_accuracy_actions':float(((p[selected]>0)==(y[selected]>0)).mean())}
    report['entrance_schedule']={}
    for arm in sorted(set(r['stage1'] for r in records)):
        for schedule in sorted(set(r['schedule_name'] for r in records)):
            ix=[i for i in selected if records[i]['stage1']==arm and records[i]['schedule_name']==schedule]
            if ix:report['entrance_schedule'][arm+'__'+schedule]={'actions':len(ix),'positive_fraction':float((y[ix]>0).mean()),
                                                               'mean_action_nll_gain':float(y[ix].mean())}
    identity=[records[i]['identity_fixed_gain'] for i in selected if 'identity_fixed_gain' in records[i]]
    report['identity_fixed_diagnostic']={'n':len(identity),'mean_gain':float(np.mean(identity)) if identity else None}
    actual_noise=[abs(r['nll_difference']) for r in noop_records if r.get('actually_recomputed')]
    report['noop_replay_noise']={'actually_recomputed':len(actual_noise),
                                'absolute_p95':float(np.percentile(actual_noise,95)) if actual_noise else None,
                                'absolute_max':max(actual_noise) if actual_noise else None,
                                'fraction_action_gain_above_noise_p95':float((np.abs(y[selected])>np.percentile(actual_noise,95)).mean()) if actual_noise else None}
    report['limits']='One-step sampled candidate ranking only. No new ACC, TTFT, statistical significance, or multi-round oracle is inferred.'
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('files',nargs='+');p.add_argument('--models',required=True)
    p.add_argument('--out',required=True);p.add_argument('--allow-synthetic',action='store_true')
    p.add_argument('--group-size',type=int,default=1);p.add_argument('--shortlist',type=int,default=16)
    p.add_argument('--candidate-limit',type=int,default=16)
    a=p.parse_args();result=diagnose(a.files,a.models,a.allow_synthetic,a.group_size,a.shortlist,a.candidate_limit)
    Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(result['test_prefix_states'])
