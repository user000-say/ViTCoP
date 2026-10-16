"""ACC-only selector telemetry and exact endpoint-change matched controls.

Never enable file I/O or CPU tensor transfer inside TTFT measurements. The
inference method does not consume gold labels. Trace keys identify the original
sample/prompt, NOT the model prediction. Random controls match final set changes,
not a nominal maximum number of internal exchange attempts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sample_seed(request_key,seed=20):
    return int(hashlib.sha256((str(seed)+'|'+str(request_key)).encode()).hexdigest()[:15],16)


def make_trace(*,request_key,config_id,source_hash,baseline_ids,selected_ids,
               eligible_ids,hook_called,scorer_sha256=None):
    base,chosen,eligible=map(lambda x:list(map(int,x)),(baseline_ids,selected_ids,eligible_ids))
    if len(set(base))!=len(base) or len(set(chosen))!=len(chosen) or len(set(eligible))!=len(eligible):
        raise ValueError('selector trace repeats native patch identity')
    if len(base)!=len(chosen) or not set(base+chosen)<=set(eligible):
        raise ValueError('trace changed budget or introduced an unavailable patch')
    return {'request_key':str(request_key),'config_id':config_id,'source_hash':source_hash,
            'baseline_patch_ids':base,'selected_patch_ids':chosen,'eligible_patch_ids':eligible,
            'hook_called':bool(hook_called),'scorer_sha256':scorer_sha256,
            'net_changes':len(set(chosen)-set(base))}


def append_trace(path,record):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n')


def load_targets(path,expected_config_id,source_hash):
    result={}
    with Path(path).open() as file:
        for line in file:
            r=json.loads(line)
            if r['config_id']!=expected_config_id or r['source_hash']!=source_hash:
                raise ValueError('matched-control trace source/config mismatch')
            key=r['request_key']
            if key in result and result[key]!=r:raise ValueError('conflicting duplicate request trace')
            result[key]=r
    if not result:raise ValueError('empty matched-control trace')
    return result


def matched_random_numpy(scores,baseline,patch_ids,target,request_key,shortlist=16,seed=20,candidate_limit=None):
    s=np.asarray(scores);selected=np.asarray(baseline,dtype=int);ids=np.asarray(patch_ids)
    if target['request_key']!=str(request_key) or set(ids[selected].tolist())!=set(target['baseline_patch_ids']) or set(ids.tolist())!=set(target['eligible_patch_ids']):
        raise ValueError('matched endpoint does not share the actual baseline prefix selection')
    m=int(target['net_changes']);n=len(s);mask=np.zeros(n,dtype=bool);mask[selected]=True
    if candidate_limit is not None:
        if m not in (0,1):raise ValueError('one-step catalogue control can match zero or one exchange')
        if not m:return np.sort(selected)
        from exchange_numpy import proposals
        d,i=proposals(s,selected,shortlist=shortlist,candidate_limit=candidate_limit)
        rng=np.random.default_rng(sample_seed(request_key,seed));j=int(rng.integers(len(d)))
        mask[d[j]]=False;mask[i[j]]=True
        return np.flatnonzero(mask)
    donors=selected[np.argsort(s[selected],kind='stable')[:shortlist]]
    outsiders=np.flatnonzero(~mask);outsiders=outsiders[np.argsort(-s[outsiders],kind='stable')[:shortlist]]
    if not 0<=m<=min(len(donors),len(outsiders)):raise ValueError('matched change count infeasible')
    rng=np.random.default_rng(sample_seed(request_key,seed))
    if m:
        mask[rng.choice(donors,m,replace=False)]=False
        mask[rng.choice(outsiders,m,replace=False)]=True
    return np.flatnonzero(mask)


def matched_random_torch(scores,baseline,patch_ids,target,request_key,shortlist=16,seed=20,candidate_limit=None):
    import torch
    if candidate_limit is not None:
        selected=matched_random_numpy(scores.detach().cpu().numpy(),baseline.detach().cpu().numpy(),
            patch_ids.detach().cpu().numpy(),target,request_key,shortlist,seed,candidate_limit)
        return torch.as_tensor(selected,device=baseline.device,dtype=baseline.dtype)
    # This is a diagnostic control with an offline trace dependency. The CPU
    # identity check is included in its own timing; never attribute it to the
    # deployed learned scorer, which does not read this trace.
    actual=patch_ids[baseline].detach().cpu().tolist()
    if str(request_key)!=target['request_key'] or set(actual)!=set(target['baseline_patch_ids']) or set(patch_ids.detach().cpu().tolist())!=set(target['eligible_patch_ids']):
        raise ValueError('matched random input differs from reference request')
    m=int(target['net_changes']);n=scores.numel();mask=torch.zeros(n,device=scores.device,dtype=torch.bool)
    mask[baseline]=True
    donors=baseline[scores[baseline].argsort(stable=True)[:shortlist]]
    outside=scores.masked_fill(mask,float('-inf')).argsort(descending=True,stable=True)[:min(shortlist,n-len(baseline))]
    if not 0<=m<=min(donors.numel(),outside.numel()):raise ValueError('matched change count infeasible')
    generator=torch.Generator(device=scores.device).manual_seed(sample_seed(request_key,seed))
    # Fixed-size updates and ranking restore original sequence order.
    d=donors[torch.randperm(len(donors),device=scores.device,generator=generator)[:m]]
    i=outside[torch.randperm(len(outside),device=scores.device,generator=generator)[:m]]
    pos=(baseline[:,None]==d[None,:]).long().argmax(0) if m else d
    return baseline.scatter(0,pos,i).sort().values


def jitter_numpy(scores,baseline,request_key,shortlist=16,sigma=.02,seed=20):
    from features import ranks
    s=np.asarray(scores);b=np.asarray(baseline,dtype=int);mask=np.zeros(len(s),dtype=bool);mask[b]=True
    d=b[np.argsort(s[b],kind='stable')[:shortlist]]
    i=np.flatnonzero(~mask);i=i[np.argsort(-s[i],kind='stable')[:shortlist]]
    pool=np.sort(np.r_[d,i]);rng=np.random.default_rng(sample_seed(request_key,seed))
    noisy=ranks(s)[pool]+rng.normal(0,sigma,len(pool))
    mask[d]=False;mask[pool[np.argsort(-noisy,kind='stable')[:len(d)]]]=True
    return np.flatnonzero(mask)


def summarize(path):
    groups={}
    for line in Path(path).read_text().splitlines():
        r=json.loads(line);groups.setdefault(r['config_id'],[]).append(r)
    return {k:{'requests':len(rows),'hook_called_fraction':float(np.mean([r['hook_called'] for r in rows])),
               'identical_set_fraction':float(np.mean([r['net_changes']==0 for r in rows])),
               'mean_net_changes':float(np.mean([r['net_changes'] for r in rows])),
               'scorer_hashes':sorted(set(r['scorer_sha256'] for r in rows if r.get('scorer_sha256')))}
            for k,rows in groups.items()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('--out',required=True)
    a=p.parse_args();Path(a.out).write_text(json.dumps(summarize(a.trace),indent=2))
