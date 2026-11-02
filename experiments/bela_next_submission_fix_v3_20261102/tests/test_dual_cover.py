import pytest
import torch
from bela_legacy_cover.e013_stage1_support import e013_choose
from bela_next_forward.dual_cover import dual_cover_one_crop
from bela_next_forward.bela_next_compat import _assign_slots, original_dual
from bela_next_budget.crops import crop_quota_from_ids
from types import SimpleNamespace


def sample(n=576):
    torch.manual_seed(49)
    mid = torch.randn(n,32)
    pen = torch.randn(n,32)
    scores = torch.rand(n) + torch.arange(n).float() * 1e-5
    return mid,pen,scores


@pytest.mark.parametrize('k,b', [(64,8), (64,16), (64,32), (256,32), (32,8), (172,32)])
def test_matches_historical_e013(k,b):
    m,p,s=sample()
    original=e013_choose(torch.cat([m,p],-1),s,'REPLACE','COVER',b,core_k=k)[0]
    port=dual_cover_one_crop(m,p,k,slots=b,cls_scores=s)
    assert torch.equal(original,port), (k,b,original[:10],port[:10])


def test_unpack_permutation_invariant():
    m,p,s=sample()
    torch.manual_seed(10)
    perm=torch.randperm(576)
    native=dual_cover_one_crop(m,p,64,slots=8,cls_scores=s)
    changed=dual_cover_one_crop(m[perm],p[perm],64,slots=8,
                                 cls_scores=s[perm],local_patch_ids=perm)
    assert set(native.tolist()) == set(perm[changed].tolist())


def test_unpad_original_coordinates_change_selection():
    m,p,s=sample()
    chosen = torch.randperm(576)[:480]
    out=dual_cover_one_crop(m[chosen],p[chosen],60,slots=10,
                             cls_scores=s[chosen],local_patch_ids=chosen)
    assert out.unique().numel()==60
    assert chosen[out].min() >= 0 and chosen[out].max() <576


def test_proxy_is_explicit_only():
    m,p,s=sample()
    with pytest.raises(RuntimeError, match='REAL_CLS_REQUIRED'):
        dual_cover_one_crop(m,p,64,slots=8)
    valid=dual_cover_one_crop(m,p,64,slots=8,saliency_mode='norm_proxy')
    assert valid.numel()==64


@pytest.mark.parametrize('n,k,b', [(20,20,32),(20,19,32),(10,2,32),(576,320,32)])
def test_no_duplicates_bounds(n,k,b):
    m,p,s=sample(n)
    out=dual_cover_one_crop(m,p,k,slots=b,cls_scores=s)
    assert out.numel()==k
    assert out.unique().numel()==k
    assert int(out.min())>=0 and int(out.max())<n

@pytest.mark.parametrize('bad', [torch.tensor([0]*576), torch.arange(576)+1])
def test_illegal_crop_indices(bad):
    m,p,s=sample()
    with pytest.raises(ValueError):
        dual_cover_one_crop(m,p,64,cls_scores=s,local_patch_ids=bad)


def test_global_slots_respect_budget():
    alloc,used=_assign_slots([64]*5,[576]*5,32)
    assert sum(alloc)==32 and used==32 and max(alloc) <32
    alloc2,used2=_assign_slots([2,2,2],[3,3,3],32)
    assert sum(alloc2)==3 and used2==3


def test_real_crop_selection_tracks_ids():
    m,p,s=sample()
    m=torch.cat([m,m],0);p=torch.cat([p,p],0);s=torch.cat([s,s],0)
    class Layout:
        patch_mask=torch.ones(1152,dtype=torch.bool)
        origin_crop=torch.cat([torch.zeros(576,dtype=torch.long),torch.ones(576,dtype=torch.long)])
        origin_local_patch=torch.arange(576).repeat(2)
    ctx={'native_layout': Layout(), 'packed_cls_scores':s, 'saliency_mode':'cls_attention'}
    out=original_dual(mid=m,pen=p,keep=128,slots=32,seed=17,context=ctx)
    assert out.unique().numel()==128
    assert ctx['dual_quotas']==[64,64]
    assert sum(ctx['dual_slots_per_crop'])==32
    assert (out<576).sum()==64 and (out>=576).sum()==64


def test_crop_quota_respects_capacity():
    q=crop_quota_from_ids([576,480,130,576,96],320,policy='uniform')
    assert sum(q)==320 and all(x<=c for x,c in zip(q,[576,480,130,576,96]))
