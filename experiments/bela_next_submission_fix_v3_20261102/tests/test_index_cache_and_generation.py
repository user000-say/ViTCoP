import sys
import types
import pytest
import torch
from bela_next_forward.stage2 import (
    Stage2State, q_attn_scores_from_qk, prune_cache, select_and_prune
)
from bela_next_forward.apply_llava_next import (
    _rewrite_image_placeholders, _verify_native_placeholders,
    _remap_text_start, _generation_parameters, _arms
)
from bela_next_forward.native_stage1 import _question_ids_from_input


def test_QATTN_matches_full_softmax():
    torch.manual_seed(30)
    q=torch.randn(1,4,17,8)
    k=torch.randn(1,4,17,8)
    pos=torch.tensor([1,2,3,6,7,8,10])
    score=q_attn_scores_from_qk(q,k,pos,11)
    logits=(q.float() @ k.float().transpose(-1,-2))/8**.5
    causal=torch.ones(17,17,dtype=torch.bool).triu(1)
    probs=logits.masked_fill(causal,-torch.inf).softmax(-1)
    expect=probs[0,:,11:,:][:,:,pos].mean((0,1))
    assert torch.allclose(score,expect,atol=1e-6)


@pytest.mark.parametrize('idx,query_start', [(torch.tensor([30]),11),(torch.tensor([1,1]),11),(torch.tensor([1]),17)])
def test_qattn_invalid_indices(idx,query_start):
    q=torch.randn(1,4,17,8)
    with pytest.raises((RuntimeError,ValueError)):
        q_attn_scores_from_qk(q,q,idx,query_start)


class FakeLayer:
    def __init__(self,k,v):
        self.keys=k
        self.values=v
        self.is_initialized=True

class FakeCache:
    def __init__(self,layers):
        self.layers=layers
    def get_seq_length(self):
        return self.layers[0].keys.shape[-2]


def stub_transformers_cache(monkeypatch):
    tr=types.ModuleType('transformers')
    ca=types.ModuleType('transformers.cache_utils')
    ca.DynamicCache=FakeCache
    monkeypatch.setitem(sys.modules,'transformers',tr)
    monkeypatch.setitem(sys.modules,'transformers.cache_utils',ca)


def test_cache_exact_gather_and_failed_old_length(monkeypatch):
    stub_transformers_cache(monkeypatch)
    cache=FakeCache([FakeLayer(torch.arange(10).view(1,1,10,1),torch.arange(10).view(1,1,10,1)) for _ in range(3)])
    prune_cache(cache,torch.tensor([0,4,7,9]),3,10)
    assert cache.get_seq_length()==4
    assert cache.layers[2].keys.flatten().tolist()==[0,4,7,9]
    with pytest.raises(RuntimeError,match='CACHE_LENGTH'):
        prune_cache(cache,torch.tensor([0,1,2]),3,10)


def test_two_cuts_identity_and_rope(monkeypatch):
    stub_transformers_cache(monkeypatch)
    state=Stage2State(torch.tensor([1,2,4,5]),(2,1),(12,20),patch_uids=torch.tensor([10,20,40,50]))
    h=torch.randn(1,9,16)
    pos=torch.arange(9).unsqueeze(0)
    cache=FakeCache([FakeLayer(torch.arange(9).view(1,1,9,1),torch.arange(9).view(1,1,9,1)) for _ in range(12)])
    q=torch.randn(1,4,9,8)
    h,pos,cache,state=select_and_prune(state=state,hidden_states=h,position_ids=pos,
        past_key_values=cache,decoder_layer=None,position_embeddings=None,
        layer_1based=12,k_target=2,layers_filled=12,text_start=6,q=q,k=q)
    assert h.shape[1]==7 and cache.get_seq_length()==7
    assert pos[0].tolist()==sorted(pos[0].tolist())
    assert state.patch_uids.numel()==2 and state.patch_uids.unique().numel()==2
    # New higher decoder layers populate cache to the cut length.
    for _ in range(8):
        cache.layers.append(FakeLayer(torch.arange(7).view(1,1,7,1),torch.arange(7).view(1,1,7,1)))
    q=torch.randn(1,4,7,8)
    h,pos,cache,state=select_and_prune(state=state,hidden_states=h,position_ids=pos,
        past_key_values=cache,decoder_layer=None,position_embeddings=None,
        layer_1based=20,k_target=1,layers_filled=20,text_start=int(state.patch_pos.max())+1,q=q,k=q)
    assert h.shape[1]==6 and cache.get_seq_length()==6
    assert state.patch_uids.numel()==1 and int(state.patch_uids[0]) in {10,20,40,50}
    assert len(state.events)==2 and state.events[0]['seq_after']==7 and state.events[1]['seq_after']==6
    assert pos[0].tolist()==sorted(pos[0].tolist())


def test_image_placeholder_rewrite_and_no_text_deleted():
    ids=torch.tensor([[11,77,77,77,13,14]])
    mask=torch.tensor([[0,1,1,1,1,1]])
    output,am,start=_rewrite_image_placeholders(ids,mask,77,2)
    assert output.tolist()==[[11,77,77,13,14]]
    assert am.tolist()==[[0,1,1,1,1]]
    assert start==1
    with pytest.raises(RuntimeError,match='NONCONTIGUOUS'):
        _rewrite_image_placeholders(torch.tensor([[77,5,77]]),None,77,1)


class FakeTokenizer:
    bos_token_id=1
    def __call__(self,text,add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[ord(c)+100 for c in text])
    def decode(self,ids,skip_special_tokens=False,clean_up_tokenization_spaces=False):
        return ''.join(chr(x-100) for x in ids)


def test_qa_slices_original_text_not_template():
    tok=FakeTokenizer()
    ids=torch.tensor([1]+[ord(x)+100 for x in 'USER: ']+[77]+[ord(x)+100 for x in ' What color? ASSISTANT:'])
    out=_question_ids_from_input(ids,77,tok)
    assert tok.decode(out.tolist())=='What color?'
    with pytest.raises(RuntimeError,match='BOUNDARY'):
        _question_ids_from_input(torch.tensor([1]+[ord(x)+100 for x in 'What color?']),77,tok)


def test_fixed_W_budget_and_generation_options():
    arms=_arms();arm=arms['NEXT7_T160_A0_PAPER']
    assert 12*arm['stage_k'][0]+8*arm['stage_k'][1]+12*arm['stage_k'][2]==5120
    dummy=types.SimpleNamespace(generation_config=types.SimpleNamespace(max_new_tokens=None,max_length=None,
                                   do_sample=False,num_beams=1,temperature=1.,top_p=1.,use_cache=True,
                                   eos_token_id=2,repetition_penalty=1.))
    input_ids=torch.ones(1,10,dtype=torch.long)
    assert _generation_parameters(dummy,input_ids,{'max_new_tokens':300})==(300,{2})
    with pytest.raises(RuntimeError,match='UNSUPPORTED_GENERATION_OPTIONS'):
        _generation_parameters(dummy,input_ids,{'max_new_tokens':10,'logits_processor':object()})
    with pytest.raises(RuntimeError,match='Only greedy'):
        _generation_parameters(dummy,input_ids,{'max_new_tokens':10,'do_sample':True})
    with pytest.raises(RuntimeError,match='repetition_penalty'):
        _generation_parameters(dummy,input_ids,{'max_new_tokens':10,'repetition_penalty':1.4})


def test_stale_text_start_newline_not_query():
    # p=patch, N=newline, T=text; cut must not pretend final newline is a text query.
    # old [prefix,p0,p1,p2,N,p3,N,T0,T1] -> [prefix,p0,p3,N,T0,T1]
    keep=torch.tensor([0,1,5,6,7,8])
    assert _remap_text_start(keep,7,9)==4  # N at index 3, true text starts at 4
    keep2=torch.tensor([0,2,3,4,5])  # second-stage full physical keep
    assert _remap_text_start(keep2,4,6)==3
    with pytest.raises(RuntimeError,match='TEXT_REMAP_INVALID_KEEP'):
        _remap_text_start(torch.tensor([0,8,8]),7,9)
    with pytest.raises(RuntimeError,match='TEXT_REMAP_NO_TEXT'):
        _remap_text_start(torch.tensor([0,1,5,6]),7,9)


def test_native_placeholder_parity_before_rewrite():
    ids=torch.tensor([[1,44,77,77,77,2]])
    assert _verify_native_placeholders(ids,torch.ones_like(ids),77,3)==2
    with pytest.raises(RuntimeError,match='NATIVE_PLACEHOLDER_MISMATCH'):
        _verify_native_placeholders(ids,torch.ones_like(ids),77,4)
    with pytest.raises(RuntimeError,match='NONCONTIGUOUS_IMAGE_PLACEHOLDERS'):
        _verify_native_placeholders(torch.tensor([[77,3,77,77]]),None,77,3)
    with pytest.raises(RuntimeError,match='IMAGE_PLACEHOLDER_ATTENTION_MASK'):
        _verify_native_placeholders(ids,torch.tensor([[1,1,1,0,1,1]]),77,3)


def test_qattn_padding_mask_matches_full_reference():
    torch.manual_seed(47)
    q=torch.randn(1,3,13,7)
    k=torch.randn(1,3,13,7)
    patch=torch.tensor([3,4,5,6,7])
    mask=torch.tensor([[0,0,1,1,1,1,1,1,1,1,1,1,1]])
    got=q_attn_scores_from_qk(q,k,patch,text_start=9,valid_token_mask=mask)
    full=(q.float() @ k.float().transpose(-1,-2))*(7**-.5)
    causal=(torch.arange(13)[None,:]<=torch.arange(13)[:,None])
    allow=(causal & mask[0].bool().view(1,-1))
    full=full.masked_fill(~allow[None,None,:,:],-torch.inf)
    reference=full.softmax(-1)[0,:,9:,patch].mean((0,1))
    assert torch.allclose(got,reference,atol=1e-6)
    with pytest.raises(RuntimeError, match='Q_ATTN_MASK_SHAPE'):
        q_attn_scores_from_qk(q,k,patch,9,valid_token_mask=torch.ones(13))
    mask2=mask.clone();mask2[0,9:]=0
    with pytest.raises(RuntimeError,match='Q_ATTN_NO_VALID_TEXT'):
        q_attn_scores_from_qk(q,k,patch,9,valid_token_mask=mask2)
    mask3=mask.clone();mask3[0,3]=0
    with pytest.raises(RuntimeError,match='Q_ATTN_PADDED_PATCH'):
        q_attn_scores_from_qk(q,k,patch,9,valid_token_mask=mask3)


def test_paper_five_crop_protocol_does_not_silently_change_budget():
    from bela_next_forward.native_stage1 import _require_paper_crop_protocol
    _require_paper_crop_protocol({'sampling_policy':'locked_paper_five_crop'}, 5)
    with pytest.raises(RuntimeError, match='PAPER_CROP_PROTOCOL_MISMATCH'):
        _require_paper_crop_protocol({'sampling_policy':'locked_paper_five_crop'}, 3)
    _require_paper_crop_protocol({'sampling_policy':'variable_resolution_diagnostic'}, 3)


def test_qa_requires_exact_runtime_tokenizer_instance():
    from types import SimpleNamespace
    from bela_next_forward.native_stage1 import _tokenizer_for
    exact = object()
    assert _tokenizer_for(SimpleNamespace(_tokenizer=exact)) is exact
    with pytest.raises(RuntimeError, match='QA_TOKENIZER_IDENTITY_REQUIRED'):
        _tokenizer_for(SimpleNamespace(config=SimpleNamespace(_name_or_path='arbitrary')))
