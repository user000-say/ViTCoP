"""Project hook dispatch. Call only at FIRST pruning event for E020 configs."""
import torch
from module_torch import token_features, select
from evidence_torch import select as coverage_select


@torch.no_grad()
def refine(spec,hook,scores,response,baseline_indices,patch_ids,*,after,depth,
           next_after,final_k,hidden=None,encoder_rank=None,patch_xy=None,generator=None,
           request_key=None):
    kind=spec['kind']
    if kind=='reference':return baseline_indices.sort().values
    if kind=='random' and request_key is not None:
        from selector_audit import sample_seed
        generator=torch.Generator(device=scores.device).manual_seed(sample_seed(request_key,spec.get('seed',20)))
    if kind=='matched_random':
        from selector_audit import matched_random_torch
        if request_key is None or not hasattr(hook,'matched_targets'):raise ValueError('matched control requires request identity and validated reference trace')
        return matched_random_torch(scores,baseline_indices,patch_ids,hook.matched_targets[str(request_key)],
                                    request_key,shortlist=spec['shortlist'],seed=spec.get('seed',20),
                                    candidate_limit=spec.get('candidate_limit'))
    if kind=='rank_jitter':
        from module_torch import ranks
        from selector_audit import sample_seed
        if request_key is None:raise ValueError('rank jitter needs stable sample identity')
        n,k=scores.numel(),baseline_indices.numel();mask=torch.zeros(n,device=scores.device,dtype=torch.bool);mask[baseline_indices]=True
        donor=baseline_indices[scores[baseline_indices].argsort(stable=True)[:spec['shortlist']]]
        outside=scores.masked_fill(mask,float('-inf')).argsort(descending=True,stable=True)[:min(spec['shortlist'],n-k)]
        pool=torch.cat([donor,outside]).sort().values
        rng=torch.Generator(device=scores.device).manual_seed(sample_seed(request_key,spec.get('seed',20)))
        perturbed=ranks(scores.float())[pool]+spec['sigma']*torch.randn(len(pool),device=scores.device,generator=rng)
        chosen=pool[perturbed.argsort(descending=True,stable=True)[:len(donor)]]
        positions=(baseline_indices[:,None]==donor[None,:]).long().argmax(0)
        return baseline_indices.scatter(0,positions,chosen).sort().values
    needs_history=(kind=='rank_drift' or (kind=='coverage_exchange' and spec.get('alpha',0)>0)
                   or (kind=='continuation' and spec.get('model')=='full'))
    previous=hook.aligned_previous(patch_ids) if needs_history else None
    if kind=='current_score':return scores.argsort(descending=True,stable=True)[:len(baseline_indices)].sort().values
    if kind=='rank_drift':
        if previous is None:raise ValueError('capture previous prompt response at an earlier boundary')
        from module_torch import ranks
        now=ranks(response.mean(0));old=ranks(previous.mean(0))
        s=spec['current_weight']*now+(1-spec['current_weight'])*(now-old)
        return s.argsort(descending=True,stable=True)[:len(baseline_indices)].sort().values
    if kind=='coverage_exchange':
        return coverage_select(scores,response,baseline_indices,previous=previous,alpha=spec['alpha'],
                               lam=spec['lambda'],swaps=spec['rounds'])
    x=token_features(scores,response,previous=previous,hidden=hidden,projection=hook.projection,
                     encoder_rank=encoder_rank,patch_xy=patch_xy)
    if kind not in ('continuation','random'):raise ValueError('unknown selector kind')
    if kind=='continuation' and hook.model is None:raise ValueError('trained model required')
    if kind=='continuation':
        from policy_protocol import validate_model_policy
        policy=validate_model_policy(hook.model.metadata)
        for key in ('rounds','group_size','shortlist','candidate_limit'):
            if spec.get(key)!=policy[key]:raise ValueError('deployment differs from calibrated policy: '+key)
        if spec.get('threshold')!=hook.model.metadata['decision_threshold']:
            raise ValueError('ACC threshold must be frozen from validation, not changed per benchmark')
        if spec.get('accept_enabled')!=hook.model.metadata['accept_enabled']:
            raise ValueError('calibrated abstention state mismatch')
    if kind=='continuation' and needs_history and previous is None:
        raise ValueError('history-enabled model requires captured previous prompt response')
    with torch.autocast(device_type=x.device.type,enabled=False):
        return select(x,scores.float(),baseline_indices,hook.model,after=after,depth=depth,
                      next_after=next_after,final_k=final_k,
                      rounds=0 if kind=='continuation' and not spec['accept_enabled'] else spec['rounds'],
                      group_size=spec['group_size'],shortlist=spec['shortlist'],threshold=spec.get('threshold',0.),
                      mode='random' if kind=='random' else 'learned',generator=generator,
                      candidate_limit=spec.get('candidate_limit'))
