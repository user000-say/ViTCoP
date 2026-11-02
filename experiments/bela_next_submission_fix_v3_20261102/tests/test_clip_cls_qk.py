import torch
from torch import nn
from types import SimpleNamespace
from bela_next_forward.native_stage1 import cls_scores_from_clip_qk


class FakeCLIPLayer(nn.Module):
    def __init__(self,dim=32,heads=4):
        super().__init__()
        self.layer_norm1=nn.LayerNorm(dim)
        self.self_attn=nn.Module()
        self.self_attn.num_heads=heads
        self.self_attn.head_dim=dim//heads
        self.self_attn.scale=(dim//heads)**-0.5
        self.self_attn.q_proj=nn.Linear(dim,dim)
        self.self_attn.k_proj=nn.Linear(dim,dim)


def test_cls_probe_equals_eager_attention_oracle():
    torch.manual_seed(123)
    layers=nn.ModuleList([FakeCLIPLayer() for _ in range(4)])
    model=nn.Module()
    model.vision_model=nn.Module()
    model.vision_model.encoder=nn.Module()
    model.vision_model.encoder.layers=layers
    model.eval()
    hidden=[torch.randn(2,577,32) for _ in range(5)]
    probe=cls_scores_from_clip_qk(model,hidden,pen_index=-2)
    layer=layers[-2]
    h=layer.layer_norm1(hidden[-3]); b,s,d=h.shape
    q=layer.self_attn.q_proj(h).view(b,s,4,8).transpose(1,2)
    k=layer.self_attn.k_proj(h).view(b,s,4,8).transpose(1,2)
    eager=((q.float() @ k.float().transpose(-1,-2))*layer.self_attn.scale).softmax(-1)
    ref=eager[:,:,0,1:].mean(1)
    assert torch.allclose(probe,ref,atol=1e-6)
    assert probe.shape==(2,576)
