"""Deployable single-sample selector. Torch is optional for offline tools.

Install beside the successful project runner. Call refine BEFORE gathering
visual hidden states at a pruning event. Existing text tokens, patch order,
position IDs, RoPE and KV-cache deletion remain the caller's responsibility.
The core changes WHICH real patches survive; cardinality and boundaries do not
change. Run CUDA parity and end-to-end timing in the actual project environment.
"""
import numpy as np
import torch
from torch import nn
from features import INPUT_DIM, TOKEN_DIM, FEATURE_SCHEMA


def ranks(x):
    order = x.argsort(stable=True)
    return torch.zeros_like(x).scatter(0,order,torch.arange(len(x),device=x.device,dtype=x.dtype)/max(len(x)-1,1))


def rows(a,n):
    if a.ndim != 2 or a.shape[1] != n or not 1 <= a.shape[0] <= 4:
        raise ValueError('need 1..4 prompt-only response groups with current patch columns')
    a = a.float(); a = a/a.sum(1,keepdim=True).clamp_min(1e-12)
    return torch.cat([a,a.new_zeros((4-a.shape[0],n))],0)


@torch.no_grad()
def _token_features(scores,response,*,previous=None,hidden=None,projection=None,
                   encoder_rank=None,patch_xy=None):
    s = scores.float(); n = s.numel(); a = rows(response,n)
    p = rows(previous,n) if previous is not None else torch.zeros_like(a)
    x = s.new_zeros((n,TOKEN_DIM)); now = ranks(a.mean(0))
    x[:,0] = ranks(s); x[:,1:5] = (a*n).T; x[:,5] = now
    x[:,6:10] = (p*n).T
    if previous is not None: x[:,10] = now-ranks(p.mean(0))
    if hidden is not None:
        if projection is None or projection.shape != (hidden.shape[1],8):
            raise ValueError('persisted decoder projection [D,8] required')
        h = hidden.float()
        x[:,11:19] = (h/h.norm(dim=1,keepdim=True).clamp_min(1e-12))@projection.float()
    if encoder_rank is not None: x[:,19] = encoder_rank.float()
    if patch_xy is not None: x[:,20:22] = patch_xy.float()
    return x


@torch.no_grad()
def token_features(scores,response,**kwargs):
    with torch.autocast(device_type=scores.device.type,enabled=False):
        return _token_features(scores,response,**kwargs)


class PairScorer(nn.Module):
    def __init__(self,path):
        super().__init__()
        import json
        with np.load(path,allow_pickle=False) as data:
            self.metadata = json.loads(str(data['metadata']))
            if self.metadata['schema'] != FEATURE_SCHEMA:
                raise ValueError('checkpoint feature schema mismatch')
            for key in ('mean','std','mask','w','b','v','target_scale'):
                self.register_buffer(key,torch.tensor(data[key].copy(),dtype=torch.float32))

    def forward(self,pairs):
        z = (pairs-self.mean)/self.std*self.mask
        value = torch.tanh(z@self.w+self.b)@self.v
        return (value[:,0]-value[:,1])*self.target_scale


@torch.no_grad()
def select(x,scores,selected,model,*,after,depth,next_after,final_k,rounds=2,
           group_size=1,shortlist=16,threshold=0.,protected=None,
           mode='learned',generator=None):
    if rounds < 0 or group_size < 1 or shortlist < group_size:
        raise ValueError('invalid exchange settings')
    if not 0 <= after < next_after <= depth:
        raise ValueError('future interaction horizon required')
    n,k = x.shape[0],selected.numel(); selected = selected.sort().values
    protect = torch.zeros(n,device=x.device,dtype=torch.bool) if protected is None else protected
    for _ in range(rounds):
        mask = torch.zeros(n,device=x.device,dtype=torch.bool); mask[selected] = True
        # Fixed-size ranking avoids CUDA nonzero()'s dynamic-output sync. The
        # protected tail is masked at candidate level rather than copied to CPU.
        di = selected[scores[selected].masked_fill(protect[selected],float('inf')).argsort(stable=True)[:shortlist]]
        ii = scores.masked_fill(mask,float('-inf')).argsort(descending=True,stable=True)[:min(shortlist,n-k)]
        nd,ni = di.numel()//group_size,ii.numel()//group_size
        if not nd or not ni: break
        dg = di[:nd*group_size].reshape(nd,group_size).repeat_interleave(ni,dim=0)
        ig = ii[:ni*group_size].reshape(ni,group_size).repeat(nd,1)
        valid = ~protect[dg].any(1)
        context = (x[selected].sum(0)[None,:]-x[dg].sum(1))/max(k-group_size,1)
        const = x.new_tensor([k/n,group_size/n,after/depth,(next_after-after)/depth,final_k/n,n/576])
        context = torch.cat([context,const.expand(len(dg),-1)],1)
        pairs = torch.stack([torch.cat([x[ig].mean(1),context],1),
                             torch.cat([x[dg].mean(1),context],1)],1)
        if mode == 'learned':
            gain = model(pairs).masked_fill(~valid,float('-inf'))
            best = gain.argmax(); accept = gain[best]>threshold
        elif mode == 'random':
            lottery = torch.rand(len(dg),device=x.device,generator=generator).masked_fill(~valid,-1.)
            best = lottery.argmax(); accept = valid[best]
        else: raise ValueError('unknown exchange mode')
        # Fixed-cardinality, device-side accepted/rejected update. Timing must
        # still include ranking, feature construction, attention extraction and
        # original-patch alignment, not merely this small network's matmuls.
        donor,incoming = dg[best],ig[best]
        index = (selected[:,None] == donor[None,:]).to(torch.int64).argmax(0)
        selected = selected.scatter(0,index,torch.where(accept,incoming,donor)).sort().values
    return selected


class ContinuationHook:
    """Per request state; do not share between concurrent requests or samples."""
    def __init__(self,model=None,projection=None,debug=True):
        self.model,self.projection,self.debug = model,projection,debug
        self.reset()

    def reset(self):
        self.previous = None; self.previous_ids = None

    @torch.no_grad()
    def capture(self,response,original_patch_ids):
        self.previous = response.detach().float().clone()
        self.previous_ids = original_patch_ids.detach().clone()

    def aligned_previous(self,patch_ids):
        if self.previous is None: return None
        ordered,order = self.previous_ids.sort()
        positions = torch.searchsorted(ordered,patch_ids)
        if self.debug:
            valid = positions < ordered.numel()
            if not bool(valid.all()) or not bool((ordered[positions] == patch_ids).all()):
                raise ValueError('previous rows do not contain current patch IDs')
        return self.previous[:,order[positions]]

    @torch.no_grad()
    def refine(self,scores,response,baseline_indices,patch_ids,*,after,depth,next_after,
               final_k,hidden=None,encoder_rank=None,patch_xy=None,rounds=2,
               group_size=1,shortlist=16,threshold=0.,mode='learned'):
        if mode == 'reference' or rounds == 0:
            return baseline_indices.sort().values
        if mode == 'learned' and self.model is None:
            raise ValueError('missing trained scorer: never use random weights as fallback')
        previous=self.aligned_previous(patch_ids)
        if mode=='learned' and self.model.metadata.get('mode')!='current' and previous is None:
            raise ValueError('history-enabled model requires its actual previous prompt response')
        if self.debug:
            if patch_ids.numel() != scores.numel() or patch_ids.unique().numel() != patch_ids.numel():
                raise ValueError('preserve unique native patch identity')
            if baseline_indices.unique().numel() != baseline_indices.numel():
                raise ValueError('baseline repeats visual patches')
        x = token_features(scores,response,previous=previous,
                           hidden=hidden,projection=self.projection,encoder_rank=encoder_rank,patch_xy=patch_xy)
        with torch.autocast(device_type=x.device.type,enabled=False):
            return select(x,scores.float(),baseline_indices,self.model,after=after,depth=depth,
                          next_after=next_after,final_k=final_k,rounds=rounds,
                          group_size=group_size,shortlist=shortlist,threshold=threshold,mode=mode)
