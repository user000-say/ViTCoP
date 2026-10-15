"""Device-side exchange backend; verify parity on the real server first.

Fixed number of vectorized exchange attempts, no .item()/.cpu() in the hot path.
Caller supplies aligned rows and valid baseline indices, keeps FP32 and disables
autocast around this block. Memory/copy/optional QK recomputation must be timed.
"""
import torch


@torch.no_grad()
def select(scores,current,baseline_indices,*,previous=None,alpha=.5,lam=.15,
           swaps=4,mode='exchange',eps=.01,generator=None):
    raw=scores.float()
    s=raw-raw.min().clamp_max(0)
    s=torch.where(s.sum()>1e-12,s/s.sum().clamp_min(1e-12),torch.ones_like(s)/s.numel())
    c=current.float(); c=c/c.sum(1,keepdim=True).clamp_min(1e-12)
    if alpha:
        if previous is None:raise ValueError('aligned previous response required')
        p=previous.float(); p=p/p.sum(1,keepdim=True).clamp_min(1e-12)
        u=(1-alpha)*c+alpha*p; u=u/u.sum(1,keepdim=True).clamp_min(1e-12)
    else:u=c
    w=(u.sum(1)>1e-12).float(); w=w/w.sum().clamp_min(1)
    n=s.numel(); k=baseline_indices.numel()
    mask=torch.zeros(n,dtype=torch.bool,device=s.device); mask[baseline_indices]=True
    if mode=='score':return s.argsort(descending=True,stable=True)[:k].sort().values
    if mode=='mean':
        return (w@u).argsort(descending=True,stable=True)[:k].sort().values
    if mode=='random':
        inside=torch.where(mask)[0]; outside=torch.where(~mask)[0]
        b=min(swaps,k,n-k)
        mask[inside[torch.randperm(k,device=s.device,generator=generator)[:b]]]=False
        mask[outside[torch.randperm(n-k,device=s.device,generator=generator)[:b]]]=True
        return torch.where(mask)[0]
    if mode!='exchange':raise ValueError('invalid mode')
    if not lam or not swaps or k==n:return baseline_indices.sort().values
    # Each iteration keeps the selected array length k; rejected attempts are
    # device-side no-ops. Baseline sequence is restored at the end by index sort.
    inside=baseline_indices.sort().values
    for _ in range(swaps):
        mask.zero_(); mask[inside]=True
        outside=torch.where(~mask)[0]
        coverage=u[:,inside].sum(1)
        after=coverage[:,None,None]-u[:,inside][:,:,None]+u[:,outside][:,None,:]
        gain=(s[outside][None,:]-s[inside][:,None]
              +lam*(w[:,None,None]*((eps+after).clamp_min(eps).log()
                                    -(eps+coverage).log()[:,None,None])).sum(0))
        best=gain.flatten().argmax(); donor=best//outside.numel(); candidate=best%outside.numel()
        old=inside[donor]; new=torch.where(gain.flatten()[best]>1e-12,outside[candidate],old)
        # Avoid a Python boolean on a CUDA tensor.
        inside=inside.scatter(0,donor.reshape(1),new.reshape(1)).sort().values
    return inside
