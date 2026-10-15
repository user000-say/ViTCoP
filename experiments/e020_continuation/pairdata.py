"""Real paired suffix collection; VLM/prefix plumbing is project-owned.

Adapter factory(config) -> object:
  snapshots() yields state objects described in validate_snapshot;
  clone(state) returns independent branch including cache, RoPE, positions/masks;
  prefix_digest(state) hashes checkpoint state before/after branches;
  run_suffix(branch, keep_patch_ids, forced_final_ids=None) -> mapping with
     answer_nll (mean over SAME teacher-forced answer tokens),
     prefix_hash, cache_policy, answer_tokens_hash, answer_token_count.
No gold output may have entered snapshot features. snapshots MUST be before the
current gather. Future-policy reranking is allowed in policy labels; identical
final IDs and purged visual KV are required for identity-fixed diagnostics.
next_after is the scheduled FINAL compression boundary, not necessarily the
immediately next intermediate pruning event.
"""
import argparse
import importlib
import json
from pathlib import Path
import numpy as np
from features import TOKEN_DIM, FEATURE_SCHEMA, pair_inputs
from exchange_numpy import proposals


def validate_snapshot(s):
    needed = ('image_id','question_id','stage1','schedule_name','source_hash','cache_policy','features','scores',
              'selected','patch_ids','after','depth','next_after','final_k')
    for key in needed:
        if key not in s: raise ValueError('snapshot missing '+key)
    x = np.asarray(s['features']); ids = np.asarray(s['patch_ids'])
    if x.ndim != 2 or x.shape[1] != TOKEN_DIM or ids.shape != (len(x),) or len(set(ids.tolist())) != len(ids):
        raise ValueError('invalid features or original patch IDs')
    if not np.isfinite(x).all() or len(s['scores']) != len(x):
        raise ValueError('nonfinite or misaligned features')
    if s.get('uses_gold_features',False): raise ValueError('gold cannot enter prompt-only features')
    return s


def checked_branch(adapter,s,keep,prefix_hash,forced=None):
    # Adapter must clone; never mutate a shared prefix between paired branches.
    out = adapter.run_suffix(adapter.clone(s),np.asarray(keep),forced_final_ids=forced)
    if not np.isfinite(float(out['answer_nll'])) or out['prefix_hash'] != prefix_hash:
        raise ValueError('invalid branch NLL or mismatched actual prefix')
    if out['cache_policy'] != s['cache_policy']:
        raise ValueError('cache policy changed across paired branches')
    if not out.get('answer_tokens_hash') or out.get('answer_token_count',0) < 1:
        raise ValueError('same teacher-forced answer token identity required')
    if forced is not None:
        if not out.get('visual_kv_purged_at_horizon',False):
            raise ValueError('identity diagnostic must actually purge old visual KV at horizon')
        if out.get('forced_final_patch_ids')!=np.asarray(forced).tolist():
            raise ValueError('identity diagnostic did not keep the requested common final IDs')
    if adapter.prefix_digest(s) != prefix_hash:
        raise ValueError('branch mutated the saved prefix')
    return out


def collect(adapter,out,*,actions_per_state=6,max_states=256,group_sizes=(1,4),
            shortlist=24,seed=20,identity_every=16,checkpoint_every=16):
    out = Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    rows,labels,records = [],[],[]; rng = np.random.default_rng(seed)
    def save():
        if rows:
            tmp = out.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,pairs=np.asarray(rows),labels=np.asarray(labels),
                                records=np.asarray([json.dumps(r,sort_keys=True) for r in records]),
                                schema=FEATURE_SCHEMA,kind=getattr(adapter,'kind','real'))
            tmp.replace(out)
    try:
        for state_index,s in enumerate(adapter.snapshots()):
            if state_index >= max_states: break
            validate_snapshot(s); x=np.asarray(s['features']); selected=np.asarray(s['selected'],dtype=int)
            ids=np.asarray(s['patch_ids']); prefix=adapter.prefix_digest(s)
            base=checked_branch(adapter,s,ids[selected],prefix)
            candidates=[]
            for g in group_sizes:
                d,i=proposals(s['scores'],selected,group_size=g,shortlist=shortlist)
                if len(d):
                    take=rng.choice(len(d),size=min(actions_per_state,len(d)),replace=False)
                    candidates.extend((d[j],i[j]) for j in take)
            if candidates:
                take=rng.choice(len(candidates),size=min(actions_per_state,len(candidates)),replace=False)
            else: take=[]
            for action_index,j in enumerate(take):
                donor,incoming=candidates[j]
                alt=np.setdiff1d(selected,donor); alt=np.sort(np.r_[alt,incoming])
                branch=checked_branch(adapter,s,ids[alt],prefix)
                if branch['answer_tokens_hash'] != base['answer_tokens_hash'] or branch['answer_token_count'] != base['answer_token_count']:
                    raise ValueError('answer tokenization/normalization differs between branches')
                context={k:int(s[k]) for k in ('after','depth','next_after','final_k')}
                inp=pair_inputs(x,selected,donor[None,:],incoming[None,:],**context)[0]
                record={k:s[k] for k in ('image_id','question_id','stage1','schedule_name','source_hash','cache_policy')}
                record.update(context,prefix_hash=prefix,baseline_patch_ids=ids[selected].tolist(),
                              donor_patch_ids=ids[donor].tolist(),incoming_patch_ids=ids[incoming].tolist(),
                              baseline_nll=float(base['answer_nll']),alternative_nll=float(branch['answer_nll']),
                              answer_tokens_hash=base['answer_tokens_hash'],answer_token_count=base['answer_token_count'],
                              state_index=state_index,action_index=action_index,label_kind='policy',kind=getattr(adapter,'kind','real'))
                rows.append(inp); labels.append(base['answer_nll']-branch['answer_nll']); records.append(record)
                if 'correct' in base and 'correct' in branch:
                    record['baseline_correct']=float(base['correct'])
                    record['alternative_correct']=float(branch['correct'])
                if identity_every and len(rows)%identity_every == 0:
                    common=np.intersect1d(selected,alt)
                    if len(common) < s['final_k']: raise ValueError('not enough common final patches')
                    score=np.asarray(s['scores']); common=common[np.argsort(-score[common],kind='stable')[:s['final_k']]]
                    forced=ids[np.sort(common)]
                    # This diagnostic requires actual global visual-KV purge at
                    # cutoff on BOTH branches; otherwise label it cache-inclusive.
                    if not getattr(adapter,'supports_identity_kv_purge',False):
                        record['identity_diagnostic']='not_supported_cache_inclusive_policy_only'
                    else:
                        ia=checked_branch(adapter,s,ids[selected],prefix,forced)
                        ib=checked_branch(adapter,s,ids[alt],prefix,forced)
                        if ia['answer_tokens_hash']!=ib['answer_tokens_hash'] or ia['answer_token_count']!=ib['answer_token_count']:
                            raise ValueError('identity diagnostic answer tokens differ')
                        record['identity_fixed_gain']=float(ia['answer_nll']-ib['answer_nll'])
                        record['identity_final_patch_ids']=forced.tolist()
                if len(rows)%checkpoint_every == 0: save()
    finally: save()
    return {'paired_actions':len(rows),'file':str(out),'states_limit':max_states}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--adapter',required=True);ap.add_argument('--config',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--states',type=int,default=256)
    ap.add_argument('--actions',type=int,default=6);ap.add_argument('--seed',type=int,default=20)
    args=ap.parse_args();mod,fac=args.adapter.split(':')
    adapter=getattr(importlib.import_module(mod),fac)(json.loads(Path(args.config).read_text()))
    print(json.dumps(collect(adapter,args.out,max_states=args.states,actions_per_state=args.actions,seed=args.seed)))
