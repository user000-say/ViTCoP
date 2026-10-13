"""Regional prototypes score native patches; means are NEVER sent to the LLM.

Fixed variants isolate the reference used for residual novelty. Eviction stays
the CLS tail, the residual pool stays outside the original Top172, and output
is exactly 172 ordered, unique native patch IDs. These variants do not claim
an exact coverage optimum or solve the old joint-eviction problem.

REGION_JOINT additionally evaluates five complete exchange sets with an
explicit region-local proxy. It is a separate experiment, not a replacement
for the fixed controls or a calibrated task-utility estimator.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from .e015_regional_config import BUDGETS, RegionalConfig

VERSION = "E015_REGIONAL_V1_20261013"
PATCH_SIDE = 24
STAGE1_K = 172


def _check(tokens, attn):
    if tokens.ndim != 2 or attn.ndim != 1 or tokens.shape[0] != 576:
        raise ValueError("[E015] requires tokens [576,D], attention [576]")
    if attn.shape[0] != 576 or tokens.shape[1] < 1:
        raise ValueError("[E015] invalid feature or attention shape")
    if tokens.device != attn.device:
        raise ValueError("[E015] token/attention devices differ")
    if not tokens.is_floating_point() or not attn.is_floating_point():
        raise ValueError("[E015] floating-point features and attention required")


def _geometry(device, grid):
    """Row-major CLIP coordinates, not contiguous 1-D blocks."""
    side = PATCH_SIDE // grid
    ids = torch.arange(576, device=device, dtype=torch.long).reshape(24, 24)
    groups = ids.reshape(grid, side, grid, side).permute(0, 2, 1, 3).reshape(grid * grid, side * side)
    flat = groups.reshape(-1)
    group_id = torch.empty(576, device=device, dtype=torch.long)
    group_id.scatter_(0, flat, torch.arange(grid * grid, device=device).repeat_interleave(side * side))
    yy = torch.arange(side, device=device, dtype=torch.float32).repeat_interleave(side)
    xx = torch.arange(side, device=device, dtype=torch.float32).repeat(side)
    center_distance = (yy - (side - 1) / 2).square() + (xx - (side - 1) / 2).square()
    return groups, group_id, center_distance


def _prepare(tokens, attn, cfg):
    x = tokens.float()
    a = attn.float()
    groups, group_id, center_distance = _geometry(tokens.device, cfg.grid)
    shape = (cfg.grid * cfg.grid, groups.shape[1], x.shape[1])
    xn = F.normalize(x, p=2, dim=-1, eps=cfg.eps)
    return {
        "x": x, "a": a, "xn": xn, "groups": groups,
        "group_id": group_id, "center_distance": center_distance,
        "x_group": x.index_select(0, groups.reshape(-1)).reshape(shape),
        "xn_group": xn.index_select(0, groups.reshape(-1)).reshape(shape),
        "a_group": a.index_select(0, groups.reshape(-1)).reshape(groups.shape),
    }


def _reference(p, retained, kind, cfg):
    """All reference anchors come from the ACTUALLY RETAINED set.

    NEAREST is nearest-to-raw-centroid in cosine, not an exact pairwise medoid.
    CENTER is the retained patch nearest the geometric cell center.
    """
    mask = torch.zeros(576, dtype=torch.bool, device=p["x"].device)
    mask[retained] = True
    present = mask.index_select(0, p["groups"].reshape(-1)).reshape(p["groups"].shape)
    count = present.sum(dim=1)
    mean = (p["x_group"] * present.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1).unsqueeze(-1)
    mean_norm = mean.norm(p=2, dim=-1)
    mean_valid = (count > 0) & (mean_norm > cfg.eps)
    mu = F.normalize(mean, p=2, dim=-1, eps=cfg.eps)
    if kind in ("REGION_DUAL_SALIENT", "REGION_MEAN"):
        rep_score = p["a_group"]
    elif kind == "REGION_DUAL_NEAREST":
        rep_score = (p["xn_group"] * mu.unsqueeze(1)).sum(dim=-1)
    elif kind == "REGION_DUAL_CENTER":
        rep_score = -p["center_distance"].unsqueeze(0).expand_as(p["a_group"])
    else:
        raise ValueError("[E015] invalid fixed proposal")
    local = rep_score.masked_fill(~present, float("-inf")).argmax(dim=1)
    rep_ids = p["groups"].gather(1, local.unsqueeze(1)).squeeze(1)
    rep_valid = (count > 0) & (p["x"].index_select(0, rep_ids).norm(p=2, dim=-1) > cfg.eps)
    return mu, mean_valid, rep_ids, rep_valid, count


def _fixed(p, core, ranked_core, residual, b, kind, cfg):
    z = p["a"].new_zeros(())
    if b == 0:
        return core, core[:0], core[:0], z, None
    # Ascending CLS order reproduces legacy proxy eviction, including ties:
    # low score first, then smaller original patch ID first.
    retained = ranked_core[b:]
    removed = ranked_core[:b]
    if kind == "GLOBAL_RETAINED_MEAN":
        mean = p["x"].index_select(0, retained).mean(dim=0)
        mean_valid = mean.norm(p=2) > cfg.eps
        mu = F.normalize(mean, p=2, dim=-1, eps=cfg.eps)
        represented = (p["xn"] * mu).sum(dim=-1).clamp(0., 1.)
        represented = torch.where(mean_valid, represented, torch.zeros_like(represented))
        gain = p["a"] * (1. - represented)
        residual_gain = gain.index_select(0, residual)
        add_order = torch.argsort(residual_gain, descending=True, stable=True)
        added = residual.index_select(0, add_order[:b])
        selected = torch.cat((retained, added)).sort().values
        return selected, added, removed, residual_gain.index_select(0, add_order[:b]).sum(), None
    mu, mean_valid, reps, rep_valid, counts = _reference(p, retained, kind, cfg)
    gid = p["group_id"]
    mean_sim = (p["xn"] * mu.index_select(0, gid)).sum(dim=-1).clamp(0., 1.)
    mean_sim = torch.where(mean_valid.index_select(0, gid), mean_sim, torch.zeros_like(mean_sim))
    if kind == "REGION_MEAN":
        represented = mean_sim
    else:
        anchor = p["xn"].index_select(0, reps).index_select(0, gid)
        rep_sim = (p["xn"] * anchor).sum(dim=-1).clamp(0., 1.)
        rep_sim = torch.where(rep_valid.index_select(0, gid), rep_sim, torch.zeros_like(rep_sim))
        represented = torch.maximum(mean_sim, rep_sim)
    # Empty or zero-norm reference regions are treated as uncovered.
    gain = p["a"] * (1. - represented)
    residual_gain = gain.index_select(0, residual)
    add_order = torch.argsort(residual_gain, descending=True, stable=True)
    added = residual.index_select(0, add_order[:b])
    selected = torch.cat((retained, added)).sort().values
    details = (reps, rep_valid, counts, mean_valid)
    return selected, added, removed, residual_gain.index_select(0, add_order[:b]).sum(), details


def _quality_matrix(p, kind, cfg):
    """Image targets use ALL patches; novelty references use retained patches.

    This distinction is intentional. Quality is local: a target in region g
    can be covered only by selected tokens in region g. No 576x576 graph.
    """
    mu = p["x_group"].mean(dim=1)
    mean_valid = mu.norm(p=2, dim=-1) > cfg.eps
    mu_n = F.normalize(mu, p=2, dim=-1, eps=cfg.eps)
    if kind in ("REGION_DUAL_SALIENT", "REGION_MEAN"):
        local = p["a_group"].argmax(dim=1)
    elif kind == "REGION_DUAL_NEAREST":
        local = (p["xn_group"] * mu_n.unsqueeze(1)).sum(dim=-1).argmax(dim=1)
    elif kind == "REGION_DUAL_CENTER":
        local = (-p["center_distance"]).argmax().expand(cfg.grid * cfg.grid)
    else:
        raise ValueError("[E015] invalid quality proposal")
    reps = p["groups"].gather(1, local.unsqueeze(1)).squeeze(1)
    rep_valid = p["x"].index_select(0, reps).norm(p=2, dim=-1) > cfg.eps
    mass = p["a_group"].clamp_min(0.).sum(dim=1)
    total = mass.sum()
    mass = torch.where(total > cfg.eps, mass / total.clamp_min(cfg.eps), torch.full_like(mass, 1. / mass.numel()))
    regions = torch.arange(cfg.grid * cfg.grid, device=mass.device)
    if kind == "REGION_MEAN":
        proto, target_region = mu_n, regions
        weights = mass * mean_valid
    else:
        mw = cfg.joint_mean_weight * mean_valid
        rw = (1. - cfg.joint_mean_weight) * rep_valid
        scale = (mw + rw).clamp_min(cfg.eps)
        weights = torch.cat((mass * mw / scale, mass * rw / scale))
        proto = torch.cat((mu_n, p["xn"].index_select(0, reps)), dim=0)
        target_region = torch.cat((regions, regions))
    weights = weights / weights.sum().clamp_min(cfg.eps)
    sim = (proto @ p["xn"].transpose(0, 1)).clamp(0., 1.)
    sim = sim.masked_fill(target_region.unsqueeze(1) != p["group_id"].unsqueeze(0), 0.)
    return sim, weights


def _quality(sim, weights, selected):
    return (sim.index_select(1, selected).max(dim=1).values * weights).sum()


def e015_choose(tokens, attn, cfg: RegionalConfig, request_index=-1):
    _check(tokens, attn)
    cfg.validate()
    # Same Core construction as the old helper for comparable tie behavior.
    core = torch.topk(attn, k=STAGE1_K, largest=True).indices.sort().values
    z = attn.new_zeros((), dtype=torch.float32)
    # b=0 fixed control does no prototype, complement, or eviction work.
    if cfg.variant != "REGION_JOINT" and cfg.b == 0:
        return core, core, core[:0], core[:0], dict(
            budget=0, utility=z, gain_sum=z, risk_sum=z,
            score_type=VERSION + ":" + cfg.variant,
        )
    ranked_core = core.index_select(0, torch.argsort(attn.float().index_select(0, core), descending=False, stable=True))
    is_residual = torch.ones(576, device=tokens.device, dtype=torch.bool)
    is_residual[core] = False
    # Fixed-size complement avoids the CUDA synchronization of Boolean
    # compaction/nonzero, whose output length is otherwise data dependent.
    residual = torch.arange(576, device=tokens.device).masked_fill(~is_residual, 576).sort().values[:404]
    if cfg.variant == "GLOBAL_RETAINED_MEAN" and not cfg.audit:
        x = tokens.float()
        p = {"x": x, "a": attn.float(), "xn": F.normalize(x, p=2, dim=-1, eps=cfg.eps)}
    else:
        p = _prepare(tokens, attn, cfg)
    kind = cfg.joint_proposal if cfg.variant == "REGION_JOINT" else cfg.variant
    all_quality = None
    if cfg.variant == "REGION_JOINT":
        sim, weights = _quality_matrix(p, kind, cfg)
        base_quality = _quality(sim, weights, core)
        options = [_fixed(p, core, ranked_core, residual, b, kind, cfg) for b in BUDGETS]
        all_quality = torch.stack([_quality(sim, weights, o[0]) for o in options])
        budgets = torch.tensor(BUDGETS, device=tokens.device, dtype=torch.long)
        utility_all = all_quality - base_quality - cfg.joint_penalty * budgets.float() / STAGE1_K
        choice = utility_all.argmax()
        # One explicit synchronization is required by the legacy Python budget
        # metadata contract. Fixed variants have no explicit data .item() call.
        option_index = int(choice.item())
        b = BUDGETS[option_index]
        selected, added, removed, gain_sum, details = options[option_index]
        utility = utility_all[option_index]
    else:
        b = cfg.b
        selected, added, removed, gain_sum, details = _fixed(p, core, ranked_core, residual, b, kind, cfg)
        utility = z
    if selected.numel() != STAGE1_K:
        raise RuntimeError("[E015] selector returned !=172 patches")
    stats = dict(budget=b, utility=utility, gain_sum=gain_sum, risk_sum=z,
                 score_type=VERSION + ":" + cfg.variant)
    if cfg.audit:
        if not bool(torch.isfinite(tokens).all().item()) or not bool(torch.isfinite(attn).all().item()):
            raise RuntimeError("[E015] nonfinite audit input")
        if selected.unique().numel() != STAGE1_K:
            raise RuntimeError("[E015] duplicate audit IDs")
        counts = torch.bincount(p["group_id"].index_select(0, selected), minlength=cfg.grid * cfg.grid)
        payload = dict(version=VERSION, request_index=int(request_index), variant=cfg.variant,
                       proposal=kind, grid=cfg.grid, b=b, stage1_out=STAGE1_K,
                       selected_ids=selected.cpu().tolist(), added_ids=added.cpu().tolist(),
                       removed_ids=removed.cpu().tolist(), selected_region_counts=counts.cpu().tolist(),
                       feature_dim=int(tokens.shape[1]))
        if details is not None:
            reps, rep_valid, ref_counts, mean_valid = details
            payload.update(reference_region_counts=ref_counts.cpu().tolist(),
                           representative_ids=torch.where(rep_valid, reps, torch.full_like(reps, -1)).cpu().tolist(),
                           mean_reference_valid=mean_valid.cpu().tolist())
        if all_quality is not None:
            payload["quality_by_b"] = dict(zip(map(str, BUDGETS), all_quality.cpu().tolist()))
            payload["utility_by_b"] = dict(zip(map(str, BUDGETS), utility_all.cpu().tolist()))
        else:
            # Fixed controls share identical, frozen diagnostic targets.
            # Their selection references still differ as specified above.
            sim, weights = _quality_matrix(p, "REGION_DUAL_SALIENT", cfg)
            payload["quality_targets"] = "REGION_DUAL_SALIENT"
            payload["base_quality"] = float(_quality(sim, weights, core).item())
            payload["selected_quality"] = float(_quality(sim, weights, selected).item())
        print("[E015_REGION_JSON] " + json.dumps(payload, separators=(",", ":")))
        if cfg.dump_dir and 0 <= int(request_index) < cfg.dump_limit:
            destination = Path(cfg.dump_dir)
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / ("clip_%06d.pt" % int(request_index))
            if target.exists():
                raise RuntimeError("[E015] dump exists; use a fresh per-job dump directory: " + str(target))
            torch.save(dict(tokens=tokens.detach().cpu(), attention=attn.detach().cpu(),
                            request_index=int(request_index), version=VERSION), target)
    return selected, core, added, removed, stats
