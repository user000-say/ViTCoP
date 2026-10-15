"""Project hook dispatch. Call only at FIRST pruning event for E020 configs."""
import torch
from module_torch import token_features, select
from evidence_torch import select as coverage_select


@torch.no_grad()
def refine(spec,hook,scores,response,baseline_indices,patch_ids,*,after,depth,
           next_after,final_k,hidden=None,encoder_rank=None,patch_xy=None,generator=None):
    kind=spec['kind']
    if kind=='reference':return baseline_indices.sort().values
    previous=hook.aligned_previous(patch_ids)
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
    if kind=='continuation' and hook.model.metadata.get('mode')!='current' and previous is None:
        raise ValueError('history-enabled model requires captured previous prompt response')
    with torch.autocast(device_type=x.device.type,enabled=False):
        return select(x,scores.float(),baseline_indices,hook.model,after=after,depth=depth,
                      next_after=next_after,final_k=final_k,rounds=spec['rounds'],
                      group_size=spec['group_size'],shortlist=spec['shortlist'],threshold=spec.get('threshold',0.),
                      mode='random' if kind=='random' else 'learned',generator=generator)
