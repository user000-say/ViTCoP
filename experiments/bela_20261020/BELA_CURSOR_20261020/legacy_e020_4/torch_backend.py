"""CUDA implementation. No hidden-state CPU transfer; audit tensors stay on device.

NumPy is the reference. Debug checks synchronize and must be disabled for
production timing after real-feature validation. PyTorch is a server dependency.
"""
import math
import torch
import torch.nn.functional as F


def cosine_cache(features):
    f = F.normalize(features.float(), dim=-1, eps=1e-12)
    return (f @ f.T).clamp(-1, 1)


def recycle(hidden, keep, affinity=None, coords=None, groups=None, weights=None,
            alpha=.5, cosine_floor=.8, radius=2., assignment="semantic",
            preserve_norm=False, seed=0, debug=False):
    if hidden.ndim != 2 or min(hidden.shape) == 0 or not hidden.is_floating_point():
        raise ValueError("hidden must be floating [N,D]")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("invalid alpha")
    n = len(hidden)
    if keep.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        raise ValueError("keep must contain integer active-local indices")
    keep = keep.to(device=hidden.device, dtype=torch.long)
    if keep.ndim != 1 or not 0 < len(keep) <= n:
        raise ValueError("invalid keep shape")
    if debug:
        if keep.unique().numel() != keep.numel() or int(keep.min()) < 0 or int(keep.max()) >= n:
            raise ValueError("invalid keep indices")
        if not bool(torch.isfinite(hidden).all()):
            raise ValueError("nonfinite hidden")
    base = hidden.index_select(0, keep)
    if alpha == 0 or len(keep) == n:
        return base, {"off": True}
    if not math.isfinite(cosine_floor) or not -1 <= cosine_floor <= 1:
        raise ValueError("invalid cosine floor")
    if assignment not in ("semantic", "spatial", "random"):
        raise ValueError("unknown assignment")
    mask = torch.ones(n, device=hidden.device, dtype=torch.bool)
    mask[keep] = False
    # N and K are already known: avoid CUDA nonzero's dynamic-size sync.
    donors = torch.argsort(mask.to(torch.int8), stable=True)[len(keep):]
    if affinity is None or affinity.shape != (n, n) or affinity.device != hidden.device:
        raise ValueError("on-device [N,N] affinity required")
    sim = affinity.index_select(0, donors).index_select(1, keep).float()
    eligible = sim >= cosine_floor
    distance = None
    if radius is not None or assignment == "spatial":
        if coords is None or coords.shape != (n, 2) or coords.device != hidden.device:
            raise ValueError("on-device patch coordinates required")
        distance = (coords[donors, None].float() - coords[keep][None].float()).square().sum(-1)
    if radius is not None:
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("invalid radius")
        eligible &= distance <= radius * radius
    if groups is not None:
        if groups.shape != (n,) or groups.device != hidden.device:
            raise ValueError("invalid crop groups")
        eligible &= groups[donors, None] == groups[keep][None]
    if assignment == "semantic":
        score = sim
    elif assignment == "spatial":
        score = -distance
    else:
        generator = torch.Generator(device=hidden.device).manual_seed(int(seed))
        score = torch.rand(sim.shape, device=hidden.device, generator=generator)
    receiver = score.masked_fill(~eligible, -torch.inf).argmax(-1)
    valid = eligible.any(-1)
    w = torch.ones(n, device=hidden.device) if weights is None else weights.float()
    if w.shape != (n,) or w.device != hidden.device:
        raise ValueError("invalid weights")
    if debug and (not bool(torch.isfinite(w).all()) or bool((w < 0).any())):
        raise ValueError("invalid nonnegative weights")
    fallback = w.sum() <= 0
    w = torch.where(fallback, torch.ones_like(w), w)
    dw = w[donors] * valid
    mass = torch.zeros(len(keep), device=hidden.device)
    total = torch.zeros((len(keep), hidden.shape[1]), device=hidden.device)
    mass.index_add_(0, receiver, dw)
    total.index_add_(0, receiver, hidden[donors].float() * dw[:, None])
    center = total / mass[:, None].clamp_min(1e-12)
    fraction = alpha * mass / (w[keep] + mass).clamp_min(1e-12)
    anchors = base.float()
    out = anchors + fraction[:, None] * (center - anchors)
    if preserve_norm:
        size = out.norm(dim=-1, keepdim=True)
        restored = out * anchors.norm(dim=-1, keepdim=True) / size.clamp_min(1e-12)
        out = torch.where(size > 1e-12, restored, anchors)
    return out.to(hidden.dtype), {"off": False, "donors": donors.detach(),
        "receiver": torch.where(valid, receiver, -1).detach(),
        "donor_mass": mass.detach(), "anchor_mass": w[keep].detach(),
        "assigned": valid.sum().detach(), "positive_weight_donors": (dw > 0).sum().detach(),
        "uniform_weight_fallback": fallback.detach(),
        "max_displacement": (out - anchors).norm(dim=-1).max().detach()}


def pooled_qk(query, visual_key, reader_weights=None, causal_bounds=None):
    if causal_bounds is None or len(causal_bounds) != 2 or not all(isinstance(x, int) for x in causal_bounds):
        raise ValueError("CPU-known actual reader/visual position bounds required")
    if not 0 <= causal_bounds[1] < causal_bounds[0]:
        raise ValueError("reader must be causally after all visual keys")
    if query.ndim != 3 or visual_key.ndim != 3 or query.shape[0] != visual_key.shape[0] or query.shape[2] != visual_key.shape[2]:
        raise ValueError("matched [H,R,D] and [H,N,D] required")
    if min(query.shape) == 0 or min(visual_key.shape) == 0:
        raise ValueError("empty QK")
    q = query.float()
    w = torch.ones(q.shape[1], device=q.device) if reader_weights is None else reader_weights.float()
    if w.shape != (q.shape[1],) or w.device != q.device:
        raise ValueError("invalid reader weights")
    if reader_weights is not None and (not bool(torch.isfinite(w).all()) or bool((w < 0).any()) or float(w.sum()) <= 0):
        raise ValueError("reader weights must be finite, nonnegative and nonzero")
    pooled = (q * (w / w.sum().clamp_min(1e-12))[None, :, None]).sum(1)
    return torch.einsum("hd,hnd->n", pooled, visual_key.float()) / (q.shape[0] * math.sqrt(q.shape[2]))


def key_energy(visual_key):
    if visual_key.ndim != 3 or min(visual_key.shape) == 0:
        raise ValueError("expected [H,N,D]")
    return visual_key.float().square().sum((0, 2))


def topk(scores, keep):
    if scores.ndim != 1 or not isinstance(keep, int) or not 0 < keep <= len(scores):
        raise ValueError("invalid scores/K")
    return torch.argsort(scores, descending=True, stable=True)[:keep].sort().values


def random_keep(tokens, keep, seed, device):
    if not 0 < keep <= tokens:
        raise ValueError("invalid K")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randperm(tokens, device=device, generator=generator)[:keep].sort().values


def inject_visual(compacted_hidden, retained_visual_positions, recycled_visual, inplace=True):
    """Replace only visual rows AFTER the original gather. Inference only.

    The caller supplies actual retained positions from its existing runtime;
    no contiguous image span, position reset or cache rewrite is assumed.
    """
    if compacted_hidden.requires_grad:
        raise ValueError("this integration helper is for frozen inference")
    if compacted_hidden.ndim not in (2, 3) or (compacted_hidden.ndim == 3 and compacted_hidden.shape[0] != 1):
        raise ValueError("batch-one hidden states required")
    positions = retained_visual_positions
    if positions.ndim != 1 or positions.dtype != torch.long or positions.device != compacted_hidden.device:
        raise ValueError("actual on-device visual positions required")
    if recycled_visual.shape != (len(positions), compacted_hidden.shape[-1]) or recycled_visual.device != compacted_hidden.device:
        raise ValueError("recycled features and retained visual positions disagree")
    target = compacted_hidden if inplace else compacted_hidden.clone()
    if target.ndim == 2:
        target.index_copy_(0, positions, recycled_visual.to(target.dtype))
    else:
        target.index_copy_(1, positions, recycled_visual.to(target.dtype).unsqueeze(0))
    return target
