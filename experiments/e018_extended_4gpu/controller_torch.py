"""On-device inference backend. Requires real-server PyTorch parity/smoke tests.

No .cpu(), .numpy(), .item() or hidden full-attention request in the hot path.
Weights are trained offline; one prediction at the final pruning checkpoint.
"""
import json
import numpy as np
import torch
from torch import nn


class FrozenController(nn.Module):
    def __init__(self, checkpoint, *, allow_synthetic=False):
        super().__init__()
        with np.load(checkpoint, allow_pickle=False) as z:
            self.metadata = json.loads(str(z['metadata']))
            if self.metadata.get('schema') != 'cf_defer_v1':
                raise ValueError('wrong controller schema')
            if self.metadata.get('kind') != 'real' and not allow_synthetic:
                raise ValueError('synthetic checkpoints are for tests only')
            for key in ['mean', 'std', 'w1', 'b1', 'w2', 'b2', 'ymean', 'ystd']:
                self.register_buffer(key, torch.as_tensor(np.array(z[key]), dtype=torch.float32))
        self.lam = float(self.metadata['lambda'])

    @torch.no_grad()
    def forward(self, features):
        z = ((features.float() - self.mean) / self.std).clamp(-10, 10)
        return ((torch.tanh(z @ self.w1 + self.b1) @ self.w2 + self.b2).squeeze(-1)
                * self.ystd + self.ymean)

    @torch.no_grad()
    def decide(self, features, positive_extra_ms):
        # extra_ms must come from the same frozen cost table used in calibration.
        return self(features) > self.lam * positive_extra_ms


@torch.no_grad()
def region_summary(features, region_ids, selected, proj, groups=16):
    # Validate index range and shapes once at adapter setup. Keep execution on
    # device; current implementation intentionally targets single-image batch=1.
    x = features.float()
    z = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12) @ proj.float()
    g = region_ids.long()
    total = torch.bincount(g, minlength=groups).float()
    count = torch.bincount(g[selected], minlength=groups).float()
    sums = torch.zeros((groups, z.shape[1]), device=x.device, dtype=torch.float32)
    second = torch.zeros_like(sums)
    sums.index_add_(0, g[selected], z[selected])
    second.index_add_(0, g[selected], z[selected].square())
    denom = count[:, None].clamp_min(1)
    return torch.cat([(count / total.clamp_min(1))[:, None],
                      sums / denom, second / denom], dim=1).flatten()


@torch.no_grad()
def decision_features(query_hidden, current_visual, visual_attention, q_proj,
                      v_proj, *, target_k, completed_blocks, total_blocks, original_n):
    q, v, a = query_hidden.float(), current_visual.float(), visual_attention.float()
    qz = q / q.norm().clamp_min(1e-12) @ q_proj.float()
    vz = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12) @ v_proj.float()
    weights = a / a.sum().clamp_min(1e-12)
    n = v.shape[0]
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum() / max(float(np.log(n)), 1)
    extras = torch.stack([entropy, weights.max(), weights.topk(target_k).values.sum(),
                          a.new_tensor(n / original_n),
                          a.new_tensor(completed_blocks / total_blocks),
                          a.new_tensor(target_k / n), a.sum()])
    return torch.cat([qz, vz.mean(0), vz.square().mean(0), extras])


@torch.no_grad()
def build_inputs(base, history, retained, mode, stride=17):
    # Accept [B,D]; the same dropped count/location fields enter all real controls.
    if mode == 'history':
        extra = history
    elif mode == 'retained':
        extra = retained.clone()
        extra[:, ::stride] = history[:, ::stride]
    elif mode == 'geometry':
        extra = torch.zeros_like(history)
        extra[:, ::stride] = history[:, ::stride]
    elif mode == 'zero':
        extra = torch.zeros_like(history)
    else:
        raise ValueError('unknown mode')
    return torch.cat([base, extra], dim=1)
