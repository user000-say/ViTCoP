"""Thin hook for an existing progressive-pruning project, PyTorch required.

Capture existing query-group attention at PENULTIMATE pruning, then call refine
at FINAL pruning. This never inserts a token, changes K or updates hidden values.
Question-group identity must be fixed from original prompt token positions.
"""
import torch
from .e019_evidence_torch import select


class EvidenceHook:
    def __init__(self,config,debug_validate=True):
        self.config=config;self.debug_validate=debug_validate
        self.prior=None;self.patch_ids=None

    @torch.no_grad()
    def capture(self,response,patch_ids):
        order=patch_ids.argsort(stable=True)
        self.patch_ids=patch_ids[order].detach().clone()
        self.prior=response[:,order].float().detach().clone()
        if self.debug_validate:
            if self.patch_ids.numel()!=torch.unique(self.patch_ids).numel():
                raise ValueError('duplicate original patch identities')

    @torch.no_grad()
    def refine(self,scores,current_response,current_patch_ids,baseline_indices,sample_seed=0):
        cfg=self.config
        if cfg['mode']=='reference':return baseline_indices
        if cfg['mode']=='score':return scores.argsort(descending=True,stable=True)[:baseline_indices.numel()].sort().values
        previous=None
        if cfg['alpha']:
            if self.prior is None:raise ValueError('temporal mode needs penultimate capture')
            locations=torch.searchsorted(self.patch_ids,current_patch_ids)
            safe=locations.clamp_max(self.patch_ids.numel()-1)
            if self.debug_validate:
                valid=(locations<self.patch_ids.numel()) & (self.patch_ids[safe]==current_patch_ids)
                if not bool(valid.all()):raise ValueError('current patch missing from captured prefix')
                if self.prior.shape[0]!=current_response.shape[0]:raise ValueError('question groups changed')
            previous=self.prior[:,safe]
        if self.debug_validate:
            if baseline_indices.numel()!=torch.unique(baseline_indices).numel():
                raise ValueError('baseline selector returned duplicate indices')
            if scores.numel()!=current_patch_ids.numel() or current_response.shape[1]!=scores.numel():
                raise ValueError('current scores/response/patch identity mismatch')
        generator=None
        if cfg['mode']=='random':
            generator=torch.Generator(device=scores.device);generator.manual_seed(int(sample_seed))
        with torch.autocast(device_type=scores.device.type,enabled=False):
            return select(scores,current_response,baseline_indices,previous=previous,
                          alpha=cfg['alpha'],lam=cfg['lambda'],swaps=cfg['swaps'],
                          mode=cfg['mode'],generator=generator)
