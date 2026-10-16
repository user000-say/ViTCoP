"""Validation-only abstention calibration on COMPLETE deployed catalogues.

The bootstrap lower bounds help choose a conservative development policy; they
are not a significance claim after threshold search. Independent test/ACC is
still required. No-op is always available, but predicted acceptance can hurt.
"""
from collections import defaultdict
import numpy as np
from policy_protocol import DEFAULT_POLICY, POLICY_SCHEMA, state_identity


def complete_groups(records, eligible=None):
    grouped=defaultdict(list)
    for i,r in enumerate(records):
        if eligible is not None and not eligible[i]:continue
        if r.get('policy_schema')!=POLICY_SCHEMA:continue
        if (r.get('group_size')!=1 or r.get('proposal_shortlist')!=16
                or r.get('candidate_limit')!=16):continue
        grouped[state_identity(r)].append(i)
    complete={}
    for key,ix in grouped.items():
        sizes={records[i].get('catalogue_size') for i in ix}
        index={records[i].get('catalogue_index') for i in ix}
        if len(sizes)==1 and None not in sizes:
            n=next(iter(sizes))
            if len(ix)==n and index==set(range(n)):complete[key]=ix
    return complete


def state_policy(prediction, labels, records, groups, threshold, enabled=True):
    rows=[]
    for key,ix in groups.items():
        # Candidate order is catalogue index, matching deployment tie-breaking.
        ix=sorted(ix,key=lambda i:records[i]['catalogue_index'])
        j=ix[int(np.argmax(prediction[ix]))]
        accepted=bool(enabled and prediction[j]>threshold)
        rows.append({'state_id':key,'image_id':str(records[j]['image_id']),
                     'stage1':records[j]['stage1'],'schedule_name':records[j]['schedule_name'],
                     'chosen_index':j if accepted else None,'accepted':accepted,
                     'gain':float(labels[j]) if accepted else 0.,
                     'oracle_gain':max(0.,float(np.max(labels[ix])))})
    return rows


def image_summary(rows,replicates=400,seed=20):
    images=defaultdict(list)
    for r in rows:images[r['image_id']].append(float(r['gain']))
    values=np.asarray([np.mean(v) for _,v in sorted(images.items())])
    if not len(values):return {'images':0,'mean_gain':None,'paired_image_95pct_interval':None}
    rng=np.random.default_rng(seed)
    boot=np.mean(values[rng.integers(len(values),size=(replicates,len(values)))],axis=1)
    return {'images':len(values),'mean_gain':float(values.mean()),
            'paired_image_95pct_interval':np.quantile(boot,[.025,.975]).tolist()}


def calibrate(model,x,y,image_ids,records,validation,noise_values,*,min_images=12):
    groups=complete_groups(records,validation)
    images={str(records[ix[0]]['image_id']) for ix in groups.values()}
    base={'status':'INSUFFICIENT_COMPLETE_VALIDATION','validation_images':len(images),
          'validation_states':len(groups),'deployment_policy':DEFAULT_POLICY,
          'selection_split':'validation_only','min_validation_images':min_images,
          'decision_threshold':0.,'accept_enabled':False}
    noise=np.asarray(noise_values,dtype=float)
    if not len(noise):
        base['status']='MISSING_GENUINE_NOOP';return base
    if len(images)<min_images:return base
    floor=float(np.max(np.abs(noise)))
    p=model.predict(x)
    maxima=np.asarray([p[ix].max() for ix in groups.values()])
    thresholds=sorted(set([floor]+[max(floor,float(v)) for v in np.quantile(maxima,[.25,.5,.75,.9])]))
    choices=[]
    for threshold in thresholds:
        rows=state_policy(p,y,records,groups,threshold)
        summary=image_summary(rows)
        accepted_images=len({r['image_id'] for r in rows if r['accepted']})
        choices.append({'threshold':threshold,'accepted_images':accepted_images,
                        'acceptance_fraction':float(np.mean([r['accepted'] for r in rows])),**summary})
    valid=[r for r in choices if r['accepted_images']>=4 and r['paired_image_95pct_interval'][0]>0]
    best=max(valid,key=lambda r:r['paired_image_95pct_interval'][0]) if valid else None
    base.update(status='CALIBRATED_ACCEPT' if best else 'CALIBRATED_ABSTAIN',
                decision_threshold=best['threshold'] if best else floor,
                accept_enabled=bool(best),noop_absolute_max=floor,threshold_candidates=choices,
                caveat='Validation threshold search is model selection, not a confirmatory confidence interval.')
    return base
