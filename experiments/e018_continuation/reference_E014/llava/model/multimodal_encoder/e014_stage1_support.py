"""E014 Adaptive Evidence Exchange at a fixed Stage1 budget.

No model weights or decoder modifications.  The output contains exactly 172
*eligible* CLIP patches and no transient/context-only tokens.

Research caution: COVER proxy is the formula stated in E013 research notes,
NOT a claim that it is bit-for-bit identical to the unprovided E013 helper.
FIXED_COVER delegates to that helper and is the faithful legacy comparator.
"""
import torch
import torch.nn.functional as F

from .e014_config import E014_BUDGETS, E014Config

STAGE1_K = 172


def _check_inputs(tokens, attn):
    if tokens.ndim != 2 or attn.ndim != 1:
        raise ValueError("[E014] tokens [T,D] and attention [T] required")
    if tokens.shape[0] != attn.shape[0] or tokens.shape[0] != 576:
        raise ValueError("[E014] frozen LLaVA CLIP geometry requires 576 patches")
    if tokens.device != attn.device:
        raise ValueError("[E014] token/attention device mismatch")


def _ids_added_removed(selected, core):
    # Called only in the exact E013 comparator; torch.isin is not used
    # in the fast AEE path.
    is_in_core = torch.isin(selected, core)
    is_in_selected = torch.isin(core, selected)
    return selected[~is_in_core], core[~is_in_selected]


def _cover_gain(tokens, attn, residual_ids):
    """E013-notes COVER proxy: a_i * [1-max(0, cos(x_i, global_mean))].

    Both score terms use the same raw CLS-attention scale.  This does NOT
    establish that addition gain and eviction risk are calibrated quantities.
    """
    x = tokens.float()
    r = x.index_select(0, residual_ids)
    mu = x.mean(dim=0, keepdim=True)
    cos = F.cosine_similarity(r, mu, dim=-1, eps=1e-8).clamp(min=0., max=1.)
    return attn.float().index_select(0, residual_ids) * (1. - cos)


def _eviction_risk(tokens, attn, core_ids):
    """L_j = a_j [1-max_{k in Core, k!=j} clamp(cos(j,k),0,1)].

    Important: mask diagonal to -inf BEFORE row maximum.  Simply subtracting
    an identity matrix is incorrect when off-diagonal similarities are < 0.
    """
    core = F.normalize(
        tokens.float().index_select(0, core_ids), p=2, dim=-1, eps=1e-8
    )
    sim = core @ core.transpose(0, 1)
    sim.fill_diagonal_(float("-inf"))
    nearest = sim.max(dim=-1).values.clamp(min=0., max=1.)
    return attn.float().index_select(0, core_ids) * (1. - nearest)


def _adaptive_budget(gain, risk, attn, cfg):
    """Choose one of {0,8,16,24,32} from GPU prefix sums.

    A SINGLE .item() synchronizes GPU->CPU to obtain dynamic tensor lengths.
    Do not ignore this cost in Stage1/TTFT latency measurements.
    """
    max_b = E014_BUDGETS[-1]
    top_gain = torch.topk(gain, k=max_b, largest=True).values
    low_risk = torch.topk(risk, k=max_b, largest=False).values
    zero = top_gain.new_zeros((1,))
    prefix_g = torch.cat((zero, top_gain.cumsum(0)), dim=0)
    prefix_l = torch.cat((zero, low_risk.cumsum(0)), dim=0)
    budgets = torch.tensor(E014_BUDGETS, device=gain.device, dtype=torch.long)
    utility = (
        prefix_g.index_select(0, budgets)
        - cfg.lam * prefix_l.index_select(0, budgets)
        - cfg.penalty * budgets * attn.float().mean()
    )
    idx = int(utility.argmax().item())
    return E014_BUDGETS[idx], utility[idx]


def e014_choose(tokens, attn, cfg: E014Config, request_index: int = -1):
    """Return (selected_ids, original_core_ids, added_ids, removed_ids, stats).

    AEE/proxy variants return patch IDs in original spatial order;
    FIXED_COVER preserves the legacy E013 helper's output ordering byte-for-byte.
    All returned token IDs refer to CLIP patch coordinates [0,T).
    stats contain a Python int and 0-D device tensors, not implicit CPU syncs.
    """
    _check_inputs(tokens, attn)
    device = tokens.device
    core = torch.topk(attn, k=STAGE1_K, largest=True).indices.sort().values
    z = attn.new_zeros((), dtype=torch.float32)
    empty = core[:0]

    if cfg.variant == "BASE":
        return core, core, empty, empty, dict(
            budget=0, utility=z, gain_sum=z, risk_sum=z, score_type="BASE"
        )

    if cfg.variant == "FIXED_COVER":
        # Preserve E013 selection/ordering exactly rather than silently
        # substituting a similar-looking COVER formula.
        try:
            from .e013_stage1_support import e013_choose
        except ImportError as e:
            raise RuntimeError(
                "[E014] FIXED_COVER needs existing "
                "llava/model/multimodal_encoder/e013_stage1_support.py; "
                "it was not included in the two uploaded sources"
            ) from e
        selected, _transient, original = e013_choose(
            tokens, attn, "REPLACE", "COVER", cfg.b, cfg.seed, request_index
        )
        if selected.numel() != STAGE1_K:
            raise RuntimeError("[E014] E013 comparator returned !=172 tokens")
        # Do NOT reorder what the original helper returned: changing token
        # order would alter Llama position IDs and spoil the E013 identity gate.
        added, removed = _ids_added_removed(selected, original)
        if (original.numel() != STAGE1_K or
                added.numel() != cfg.b or removed.numel() != cfg.b or
                selected.unique().numel() != STAGE1_K):
            raise RuntimeError(
                "[E014] legacy FIXED_COVER exchange cardinality mismatch; "
                "do not treat this as a valid exact E013 control"
            )
        return selected, original, added, removed, dict(
            budget=cfg.b, utility=z, gain_sum=z, risk_sum=z,
            score_type="E013_EXACT", comparator=True
        )

    # A zero-swap fixed control must do no coverage/cosine work.
    # Keeping its overhead at the BASE level makes the b=0 comparison fair.
    if cfg.variant in ("FIXED_COVER_PROXY", "FIXED_RISK") and cfg.b == 0:
        return core, core, empty, empty, dict(
            budget=0, utility=z, gain_sum=z, risk_sum=z,
            score_type="COVER_PROXY"
        )

    # Construct the original 404 residual patches, rather than using the
    # residual relative to Top(172-b).  This freezes the candidate pool.
    idx = torch.arange(tokens.shape[0], device=device, dtype=torch.long)
    mask = torch.ones(tokens.shape[0], device=device, dtype=torch.bool)
    mask[core] = False
    residual = idx[mask]
    gain = _cover_gain(tokens, attn, residual)

    need_risk = cfg.variant in ("AEE", "ADAPTIVE_B_ONLY", "FIXED_RISK")
    risk = _eviction_risk(tokens, attn, core) if need_risk else None

    if cfg.variant in ("AEE", "ADAPTIVE_B_ONLY"):
        b, utility = _adaptive_budget(gain, risk, attn, cfg)
    else:
        b, utility = cfg.b, z

    if b == 0:
        return core, core, empty, empty, dict(
            budget=0, utility=utility, gain_sum=z, risk_sum=z,
            score_type="COVER_PROXY"
        )

    # Fixed original-Core eviction (E013's Top(172-b) counterpart) versus
    # risk-based eviction: keep those with higher eviction scores.
    if cfg.variant in ("AEE", "FIXED_RISK"):
        removal_score = risk
    else:
        removal_score = attn.float().index_select(0, core)

    removal_order = torch.argsort(removal_score, descending=False, stable=True)
    removed = core.index_select(0, removal_order[:b])
    retained = core.index_select(0, removal_order[b:])
    add_order = torch.argsort(gain, descending=True, stable=True)
    added = residual.index_select(0, add_order[:b])

    selected = torch.cat((retained, added), dim=0).sort().values
    if selected.numel() != STAGE1_K:
        raise RuntimeError("[E014] output token-count !=172")
    gain_sum = gain.index_select(0, add_order[:b]).sum()
    risk_sum = (
        risk.index_select(0, removal_order[:b]).sum()
        if risk is not None else z
    )

    return selected, core, added, removed, dict(
        budget=b, utility=utility, gain_sum=gain_sum, risk_sum=risk_sum,
        score_type="COVER_PROXY"
    )
