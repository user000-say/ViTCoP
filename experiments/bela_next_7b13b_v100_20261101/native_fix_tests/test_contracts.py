"""CPU regression suite: packing/QA/physical pruning/KV/generation.
No network, no checkpoint weights. This does NOT certify remote V100 behavior.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from bela_next_fix.native_layout import map_native_packing,structural_keep_mask,PackingMismatch
from bela_next_fix.legacy_bridge import BelaConfig,ARMS,run_original_bela
from bela_next_fix.progressive_hooks import ProgressivePruningHooks,InvalidForward
from bela_next_fix.runner import greedy_native_bela,prepare_native_bela_prefill


class MiniNative(torch.nn.Module):
    # Native spatial unpad geometry analogue: base, HR row-major with newline.
    # rows = 2, columns = 4; 3 crops 2x2 = 12 Patch, 2 newline.
    def __init__(self):
        super().__init__()
    def forward(self,feats,sizes,vision_feature_select_strategy='default',image_newline=None):
        packed=[]
        lens=[]
        for im in feats:
            if im.shape[0]!=3 or im.shape[1]!=4: raise ValueError('bad crop shape')
            # 3 crops: one base; two HR images side by side.
            hr=torch.cat((im[1].reshape(2,2,-1),im[2].reshape(2,2,-1)),dim=1)
            z=[im[0]]
            for r in hr:
                z.append(r)
                if image_newline is not None: z.append(image_newline.to(r)[None])
            f=torch.cat(z,dim=0)
            packed.append(f);lens.append(len(f))
        return torch.cat(packed), torch.tensor(lens,device=packed[0].device)


class FakeTower(torch.nn.Module):
    def forward(self,pixels,output_hidden_states=True):
        n=len(pixels)
        base=torch.arange(n*5*8,dtype=torch.float32).reshape(n,5,8)*0.001
        return SimpleNamespace(hidden_states=tuple(base + float(i)/100 for i in range(12)))


class FakeCache:
    def __init__(self): self.key_cache=[];self.value_cache=[]
    def get_seq_length(self):return self.key_cache[0].shape[-2] if self.key_cache else 0
    def update(self,k,v,idx,**kwargs):
        if len(self.key_cache)<=idx:
            self.key_cache.extend([None]*(idx+1-len(self.key_cache)))
            self.value_cache.extend([None]*(idx+1-len(self.value_cache)))
        old=self.key_cache[idx]
        self.key_cache[idx]=k if old is None else torch.cat((old,k),dim=-2)
        self.value_cache[idx]=v if old is None else torch.cat((self.value_cache[idx],v),dim=-2)
        return self.key_cache[idx], self.value_cache[idx]


class FakeDecoder(torch.nn.Module):
    def __init__(self,idx):
        super().__init__();self.idx=idx
    def forward(self,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,
                output_attentions=False,use_cache=False,cache_position=None,position_embeddings=None):
        b,n,d=hidden_states.shape
        if attention_mask is not None:
            assert attention_mask.shape[-2]==n and attention_mask.shape[-1]>=n
        assert position_embeddings[0].shape[-2]==n
        if past_key_value is not None:
            kv=torch.ones((1,2,n,4))* (self.idx+1)
            past_key_value.update(kv,kv,self.idx)
        h=hidden_states+0.0001
        if output_attentions:
            m=torch.arange(n)
            attn=1/(1+torch.abs(m[:,None]-m[None,:]).float())
            attn=attn[None,None,:,:].expand(1,2,n,n)
            return h,attn
        return (h,)


class FakeLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model=SimpleNamespace(layers=torch.nn.ModuleList([FakeDecoder(i) for i in range(32)]))
        self.config=SimpleNamespace(_attn_implementation='eager',eos_token_id=3)
        self.emb=torch.nn.Embedding(200,8)
        torch.manual_seed(91)
        self.head=torch.nn.Linear(8,30)
    def forward(self,input_ids=None,inputs_embeds=None,attention_mask=None,position_ids=None,
                past_key_values=None,cache_position=None,use_cache=True,output_attentions=False,return_dict=True):
        h=inputs_embeds if inputs_embeds is not None else self.emb(input_ids)
        n=h.shape[1]
        pos=position_ids if position_ids is not None else torch.arange(n)[None,:]
        emb=(pos[:,:,None].float().expand(-1,-1,4),pos[:,:,None].float().expand(-1,-1,4))
        mask=torch.zeros((1,1,n,n))
        if past_key_values is None: past_key_values=FakeCache()
        for layer in self.model.layers:
            h=layer(h,attention_mask=mask,position_ids=pos,past_key_value=past_key_values,
                    output_attentions=False,use_cache=use_cache,cache_position=cache_position,
                    position_embeddings=emb)[0]
        return SimpleNamespace(logits=self.head(h),past_key_values=past_key_values)


class FakeNext(torch.nn.Module):
    def __init__(self):
        super().__init__();self.vision_tower=FakeTower();self.language_model=FakeLM()
        self.config=SimpleNamespace(image_token_index=99,vision_feature_layer=-2,vision_feature_select_strategy='default')
        self.image_newline=torch.nn.Parameter(torch.randn(8))
        self.pack_image_features=MiniNative()
    def get_input_embeddings(self):return self.language_model.emb
    def get_image_features(self,pixel_values,image_sizes,vision_feature_layer,vision_feature_select_strategy):
        r=self.vision_tower(pixel_values,output_hidden_states=True)
        return [r.hidden_states[vision_feature_layer][:,1:,:]]


def example_inputs():
    # image placeholder count == native features 14, text on both ends
    return {'input_ids':torch.tensor([[7,9]+[99]*14+[10,11]]),
            'attention_mask':torch.ones((1,18),dtype=torch.long),
            'pixel_values':torch.zeros(3,3,2,2),
            'image_sizes':torch.tensor([[2,4]])}


def dual_fn(**kw):return torch.arange(kw['keep'])

def qa_fn(**kw):
    ids=kw['initial_keep'].clone()
    n=kw['mid'].shape[0]
    if n>len(ids):ids[-1]=len(ids)
    return ids


class Tests(unittest.TestCase):
    def test_arithmetic_all_arms(self):
        for cfg in ARMS.values():
            self.assertEqual(12*cfg.stage1+8*cfg.cut12+12*cfg.cut20,5120)
            self.assertTrue(cfg.check())

    def test_native_mapping_and_lines(self):
        f=MiniNative();a=torch.randn(3,4,8)
        native,le=f([a],torch.tensor([[2,4]]),image_newline=torch.ones(8))
        layout=map_native_packing(f,[a],torch.tensor([[2,4]]))
        self.assertEqual(layout.patch_count,12)
        self.assertEqual(len(layout.patch_ids),14)
        self.assertEqual(int(layout.newline_mask.sum()),2)
        self.assertEqual(layout.patch_ids[4:9].tolist(),[4,5,8,9,-1])
        self.assertTrue(torch.equal(le,layout.image_lens))
        self.assertEqual(int(structural_keep_mask(layout,[0,1,4,5]).sum()),5)
        with self.assertRaises(PackingMismatch):structural_keep_mask(layout,[0,0])
        with self.assertRaises(PackingMismatch):structural_keep_mask(layout,[12])

    def test_qa_mandatory_and_bounded(self):
        cfg=BelaConfig(6,4,2,qa_swaps=2)
        x=torch.randn(12,8)
        with self.assertRaises(RuntimeError):run_original_bela(x,x,x,cfg,dual_fn,None)
        with self.assertRaises(RuntimeError):run_original_bela(x,x,None,cfg,dual_fn,qa_fn)
        ids,log=run_original_bela(x,x,x,cfg,dual_fn,qa_fn)
        self.assertEqual(len(ids),6);self.assertEqual(log['swaps'],1)
        def malicious(**kw):return torch.arange(6)+6
        with self.assertRaises(AssertionError):run_original_bela(x,x,x,cfg,dual_fn,malicious)

    def test_dense_sidecar_invariant(self):
        model=FakeNext();inputs=example_inputs()
        a=prepare_native_bela_prefill(model,inputs,BelaConfig(6,4,2,qa_swaps=2),None,None,enabled=False)
        self.assertEqual(a.native_patch_count,12)
        self.assertEqual(a.native_newlines,2)
        self.assertEqual(a.embeds.shape[1],18)
        self.assertEqual(a.position_ids[0].tolist(),list(range(18)))
        a_ids,a_log=greedy_native_bela(model,inputs,BelaConfig(6,4,2,qa_swaps=2),enabled=False,
                    progressive=False,max_new_tokens=4)
        self.assertEqual(a_log['visual_patch_trajectory'],[12])
        self.assertEqual(len(a_ids[0]),4)

    def test_qa_full_pipeline_physically_prunes(self):
        model=FakeNext();cfg=BelaConfig(6,4,2,qa_swaps=2)
        inputs=example_inputs()
        prep=prepare_native_bela_prefill(model,inputs,cfg,dual_fn,qa_fn)
        self.assertEqual(int(prep.patch_mask.sum()),6)
        self.assertEqual(prep.qa_log['swaps'],1)
        self.assertTrue(int(prep.stage1_newlines)>=0)
        self.assertGreater(prep.position_ids.max().item(),len(prep.patch_mask)-1)
        answer,trace=greedy_native_bela(model,inputs,cfg,dual_fn,qa_fn,max_new_tokens=4)
        self.assertEqual(trace['visual_patch_trajectory'],[6,4,2])
        self.assertEqual(trace['prune_log'][0]['layer'],12)
        self.assertEqual(trace['prune_log'][1]['layer'],20)
        self.assertTrue(trace['position_preserved_after_pruning'])
        self.assertTrue(trace['qa_log']['legacy_qa_called'])
        self.assertTrue(1 <= answer.shape[1] <= 4)

    def test_strict_fail_closed(self):
        model=FakeNext()
        with self.assertRaises(RuntimeError):greedy_native_bela(model,example_inputs(),
                     BelaConfig(6,4,2,qa_swaps=2),dual_fn,None)
        model.language_model.config._attn_implementation='sdpa'
        with self.assertRaises(InvalidForward):greedy_native_bela(model,example_inputs(),
                     BelaConfig(6,4,2,qa_swaps=2),dual_fn,qa_fn)
        model.language_model.config._attn_implementation='eager'
        bad=example_inputs();bad['input_ids']=torch.cat((bad['input_ids'][:,:-3],bad['input_ids'][:,-2:]),dim=1)
        with self.assertRaises(InvalidForward):greedy_native_bela(model,bad,
                     BelaConfig(6,4,2,qa_swaps=2),dual_fn,qa_fn)


if __name__=='__main__':unittest.main(verbosity=2)
