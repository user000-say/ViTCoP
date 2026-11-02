"""Fail-closed bridge to ORIGINAL BELA implementation.

There is no legacy NeXT source in the supplied archive. We MUST call the real
1.5-7B Dual COVER and QA-COVER, never silently substitute a new algorithm.
"""
from dataclasses import dataclass
import importlib
import inspect
import torch


@dataclass(frozen=True)
class BelaConfig:
    stage1: int
    cut12: int
    cut20: int
    slots: int = 32
    qa_swaps: int = 8
    donor: int = 16
    drop: int = 12
    seed: int = 17
    mid_layer: int = 9
    pen_layer: int = -2
    require_true_qa: bool = True

    def check(self):
        assert 0 < self.cut20 <= self.cut12 <= self.stage1
        assert self.qa_swaps <= self.stage1 and self.qa_swaps >= 0
        assert self.slots > 0
        return self


ARMS = {
    'A0': BelaConfig(320,91,46),
    'A1': BelaConfig(320,100,40),
    'A2': BelaConfig(320,82,52),
    'A3': BelaConfig(288,136,48),
    'A4': BelaConfig(256,160,64),
}


def import_fn(spec):
    if ':' not in spec:
        raise ValueError('Use module:function')
    mod, name = spec.rsplit(':',1)
    fn = getattr(importlib.import_module(mod),name)
    if not callable(fn):
        raise TypeError(f'Not callable: {spec}')
    return fn


def _verified_ids(ids, n, k, label, device):
    ids = torch.as_tensor(ids,device=device,dtype=torch.long).flatten()
    if len(ids) != k or len(ids.unique()) != k or (len(ids) and (ids.min()<0 or ids.max()>=n)):
        raise AssertionError(f'{label}: wrong count, duplicate, or out-of-range patch IDs')
    return ids


def run_original_bela(mid_patches, pen_patches, question_embeddings, cfg: BelaConfig,
                      native_dual_fn, native_qa_fn, question_ids=None, extra_context=None):
    """Exact legacy selection semantics; user supplies two thin wrappers.

    native_dual_fn(*, mid,pen,keep,slots,seed,context)->flat patch IDs
    native_qa_fn(*, mid,pen,question,question_ids,initial_keep,max_swaps,
                   donor,drop,seed,context)->flat patch IDs

    The bridge does NOT implement an ersatz QA-COVER; wrapper MUST invoke the
    SAME historical qa_core coverage-delta and bounded-swap implementation.
    """
    cfg.check()
    if native_dual_fn is None:
        raise RuntimeError('BELA_REQUIRED_LEGACY_DUAL: supply the original 1.5-7B Dual COVER')
    if cfg.qa_swaps and cfg.require_true_qa and native_qa_fn is None:
        raise RuntimeError('BELA_REQUIRED_LEGACY_QA: qa_swaps=8 but QA donor-swap callable missing')
    if question_embeddings is None and cfg.qa_swaps and cfg.require_true_qa:
        raise RuntimeError('BELA_QUESTION_MISSING: QA B8 requires real question representation')
    if mid_patches.ndim!=2 or pen_patches.ndim!=2 or mid_patches.shape[0]!=pen_patches.shape[0]:
        raise AssertionError('Dual-depth feature input shape is invalid')
    n, device = mid_patches.shape[0],mid_patches.device
    if n<cfg.stage1:
        raise AssertionError(f'Only {n} native patches but stage1 requires {cfg.stage1}')
    context = extra_context or {}
    base = _verified_ids(native_dual_fn(mid=mid_patches,pen=pen_patches,keep=cfg.stage1,
                         slots=cfg.slots,seed=cfg.seed,context=context),n,cfg.stage1,'Dual COVER',device)
    if cfg.qa_swaps == 0:
        return base, {'swaps':0,'dual_ids':base.tolist(),'qa_ids':base.tolist()}
    qa = _verified_ids(native_qa_fn(mid=mid_patches,pen=pen_patches,question=question_embeddings,
                question_ids=question_ids,initial_keep=base,max_swaps=cfg.qa_swaps,
                donor=cfg.donor,drop=cfg.drop,seed=cfg.seed,context=context),
                n,cfg.stage1,'QA-COVER',device)
    swaps = len(set(qa.tolist()) - set(base.tolist()))
    if swaps > cfg.qa_swaps:
        raise AssertionError(f'QA changed {swaps} entries but B={cfg.qa_swaps}')
    return qa, {'swaps':swaps, 'dual_ids':base.tolist(),'qa_ids':qa.tolist()}
