"""Synthetic unfused CLIP fixture only; real CLIP/server parity remains required."""
import unittest
try:
    import torch
except ImportError:
    torch = None
from elva.clip_stage1 import CLIPStage1, encode_projected
from make_catalog import build_catalog


@unittest.skipIf(torch is None,'PyTorch absent locally; execute real forward hook checks on server')
class CLIPHookTests(unittest.TestCase):
    def build(self):
        from torch import nn
        class Attention(nn.Module):
            def __init__(self):
                super().__init__();self.num_heads=2;self.scale=4**-.5
                self.q_proj=nn.Linear(8,8);self.k_proj=nn.Linear(8,8);self.v_proj=nn.Linear(8,8)
            def forward(self,hidden_states,attention_mask=None,causal_attention_mask=None):
                q=self.q_proj(hidden_states).reshape(1,577,2,4).transpose(1,2)
                k=self.k_proj(hidden_states).reshape(1,577,2,4).transpose(1,2)
                v=self.v_proj(hidden_states).reshape(1,577,2,4).transpose(1,2)
                a=((q*self.scale)@k.transpose(-1,-2)).softmax(-1)
                self.last_scores=a[0,:,0,1:].mean(0)
                self.last_scores=self.last_scores/self.last_scores.sum()
                return (a@v).transpose(1,2).reshape(1,577,8)
        class Block(nn.Module):
            def __init__(self):
                super().__init__();self.self_attn=Attention();self.norm=nn.LayerNorm(8)
            def forward(self,x):
                return (x+self.self_attn(self.norm(x)),)
        class Tower(nn.Module):
            def __init__(self):
                super().__init__();self.blocks=nn.ModuleList([Block() for _ in range(10)])
                self.calls=0
            def forward(self,x):
                self.calls+=1
                for i,b in enumerate(self.blocks):
                    x=b(x)[0]
                    if i==8:self.pen=x.detach().clone()
                return self.pen[:,1:]
        torch.manual_seed(3)
        return Tower().eval()

    def test_single_forward_source_scores_raw_payload_and_real_projector(self):
        from torch import nn
        tower=self.build();x=torch.randn(1,577,8);ids=torch.arange(576)
        config=next(c for c in build_catalog() if 'W1024_V3_' in c['id'])
        projector=nn.Linear(8,5).eval()
        projected,selected=encode_projected(vision_tower=tower,projector=projector,images=x,
            blocks=tower.blocks,original_patch_ids=ids,config=config,l9_completed=4)
        self.assertEqual(tower.calls,1)
        self.assertEqual(projected.shape,(1,60,5))
        torch.testing.assert_close(selected['scores'],.5*(tower.blocks[8].self_attn.last_scores+
                                                         tower.blocks[3].self_attn.last_scores),rtol=1e-5,atol=1e-7)
        torch.testing.assert_close(selected['payload'],tower.pen[0,1:][selected['keep']],rtol=0,atol=0)
        torch.testing.assert_close(projected,projector(selected['payload'].unsqueeze(0)),rtol=0,atol=0)
        self.assertFalse(any(len(b._forward_hooks) for b in tower.blocks))

    def test_pen_only_has_no_early_capture_and_bad_payload_is_rejected(self):
        tower=self.build();x=torch.randn(1,577,8);ids=torch.arange(576)
        config=next(c for c in build_catalog() if 'W1024_V1_' in c['id'])
        with CLIPStage1(tower.blocks,config,l9_completed=4) as hooks:
            payload=tower(x)
            self.assertEqual(set(hooks.features),{9})
            with self.assertRaises(ValueError):hooks.select(payload+1,ids)
            selected=hooks.select(payload,ids)
            self.assertFalse(selected['capture_meta']['early_captured'])
            with self.assertRaises(RuntimeError):hooks.select(payload,ids)

    def test_hooks_removed_on_forward_failure_and_training_rejected(self):
        tower=self.build()
        config=next(c for c in build_catalog() if 'W1024_V1_' in c['id'])
        with self.assertRaises(RuntimeError):
            with CLIPStage1(tower.blocks,config,l9_completed=4):
                raise RuntimeError('FIXTURE native forward failure')
        for b in tower.blocks:
            self.assertFalse(b._forward_hooks)
            self.assertFalse(b.self_attn.q_proj._forward_hooks)
        with self.assertRaises(RuntimeError):CLIPStage1(tower.train().blocks,config,l9_completed=4).__enter__()


if __name__=='__main__':unittest.main()
