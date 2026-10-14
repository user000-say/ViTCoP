"""Original QED-RS idea as an OFFLINE diagnostic, not an absorption guarantee.

Use a question-by-visual attention matrix AFTER full legal-prefix softmax.
Return statistics and score; not wired into E018's main online retirement policy.
"""
import torch
import torch.nn.functional as F


def qed_rs(visual_attention, eps=1e-8):
    if visual_attention.ndim != 2 or bool((visual_attention < 0).any()):
        raise ValueError("Nonnegative [question, visual] attention required")
    joint = visual_attention.float() / visual_attention.float().sum().clamp_min(eps)
    w, r = joint.sum(1), joint.sum(0)
    conditional = joint / w[:, None].clamp_min(eps)
    mi = (joint * (conditional.clamp_min(eps).log() - r[None].clamp_min(eps).log())).sum().clamp_min(0)
    entropy_w = -(w * w.clamp_min(eps).log()).sum()
    role_diversity = torch.where(entropy_w > eps, mi / entropy_w.clamp_min(eps), torch.zeros_like(mi)).clamp(0, 1)
    posterior = joint / r[None].clamp_min(eps)
    centered = (posterior - w[:, None]) / w[:, None].clamp_min(eps).sqrt()
    z = F.normalize(centered.T, dim=-1, eps=eps)
    sim = (z @ z.T).clamp(0, 1)
    higher = r[None] > r[:, None]
    rho = sim.masked_fill(~higher, -1).max(1).values.clamp_min(0)
    return dict(score=r * (1 - role_diversity * rho), relevance=r, role_diversity=role_diversity,
                role_redundancy=rho, mutual_information=mi, question_entropy=entropy_w)
