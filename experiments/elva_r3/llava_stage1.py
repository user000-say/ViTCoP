"""Stage1 selection on the CLIP forward LLaVA already ran.

Payload stays the penultimate patch rows that the existing projector consumes.
Saliency uses this repository's live rule: sum of attention heads, CLS to patches.
That is the historical baseline, and it is not the package mean-then-L1 score.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from elva.cover_binding import NativeCover, file_sha, select_checked
from elva.low_budget_cover import POLICY, callback as pcov_callback
from native_cover import legacy_callback

PKG = Path("/home/xiangyue/ViTCoP/experiments/elva_r3")
E013 = Path("/home/xiangyue/ViTCoP/LLaVA/llava/model/multimodal_encoder/e013_stage1_support.py")
_COVERS = None
# Original-patch priority for sprint B2 ENTRY_ORDER cuts (higher = keep first).
LAST_ENTRY_PRIORITY = None
# QCP needs Stage1 Dual features + entry keep IDs (original patch coordinates).
LAST_ENTRY_KEEP = None
LAST_ENTRY_METRIC = None


def _covers():
    global _COVERS
    if _COVERS is not None:
        return _COVERS
    legacy_files = {
        str(Path("/home/xiangyue/ViTCoP/experiments/elva_r3/native_cover.py").resolve()): file_sha(
            "/home/xiangyue/ViTCoP/experiments/elva_r3/native_cover.py"),
        str(E013.resolve()): file_sha(E013),
    }
    pcov = PKG / "elva/low_budget_cover.py"
    _COVERS = {
        "legacy_native": NativeCover(legacy_callback, dict(
            family="legacy_native", entrypoint="native_cover:legacy_callback",
            # COVER@K0: late-cut 64/72/80 + EASY96 / G1-F3-128 / NORMAL160 / HARD172; slots=32.
            supported_k=[32, 64, 72, 80, 96, 128, 160, 172],
            replacement_slots={
                "32": 8, "64": 32, "72": 32, "80": 32, "96": 32,
                "128": 32, "160": 32, "172": 32,
            },
            source_files=legacy_files)),
        "protected_coverage": NativeCover(pcov_callback, dict(
            family="protected_coverage", entrypoint="elva.low_budget_cover:callback",
            # B12 fixed; K80 is the CVPR W1024 equal-budget entrance candidate.
            supported_k=[60, 80], replacement_slots={"60": 12, "80": 12},
            allowed_slots=[6, 12, 18],
            coverage_policy=POLICY, source_files={str(pcov.resolve()): file_sha(pcov)})),
    }
    return _COVERS


def _sum_cls(attentions, index):
    # attentions[index]: [1, heads, 577, 577] -> [576]
    return attentions[index][0, :, 0, 1:].sum(0).float()


def _set_entry_priority(metric_features, scores, keep: torch.Tensor, slots: int) -> torch.Tensor:
    """Rebuild COVER REPLACE priority on original patch IDs (same scores/features COVER saw).

    Anchors = CLS top (K-slots) → priority = 1e6 + CLS.
    Residual COVER picks → priority = coverage utility.
    Keep members altered by admission get CLS fallback. Others stay -1e9.
    """
    from llava.model.multimodal_encoder.e013_stage1_support import _core_relative_coverage

    global LAST_ENTRY_PRIORITY
    n = int(scores.numel())
    k = int(keep.numel())
    b = int(slots)
    if b < 0 or b > k:
        raise RuntimeError(f"invalid COVER slots={b} for K={k}")
    scores_f = scores.float()
    priority = torch.full((n,), -1e9, device=scores.device, dtype=torch.float32)
    anchors_k = k - b
    anchors = (
        torch.topk(scores_f, k=anchors_k, largest=True, sorted=False).indices
        if anchors_k > 0
        else scores.new_empty(0, dtype=torch.long)
    )
    if anchors_k > 0:
        priority[anchors] = scores_f.index_select(0, anchors) + 1e6
    if b > 0:
        base = torch.topk(scores_f, k=k, largest=True, sorted=False).indices
        base_mask = torch.ones(n, dtype=torch.bool, device=scores.device)
        base_mask[base] = False
        residual = torch.where(base_mask)[0]
        utility = _core_relative_coverage(metric_features, scores_f, anchors, residual)
        keep_set = torch.zeros(n, dtype=torch.bool, device=scores.device)
        keep_set[keep] = True
        for local_i in range(int(residual.numel())):
            pid = int(residual[local_i].item())
            if keep_set[pid]:
                priority[pid] = float(utility[local_i].item())
    # Admission / edge: any keep still unset gets CLS so Stage2 can rank it.
    for pid in keep.detach().tolist():
        if float(priority[pid].item()) <= -1e8:
            priority[pid] = float(scores_f[pid].item())
    LAST_ENTRY_PRIORITY = priority.detach()
    return priority


def _rank_percentile(scores: torch.Tensor) -> torch.Tensor:
    """Average-rank percentile in [0, 1]; ties share the same percentile."""
    n = int(scores.numel())
    if n <= 1:
        return torch.zeros_like(scores)
    order = torch.argsort(scores, descending=False)
    ranks = torch.empty_like(scores, dtype=torch.float32)
    # dense average ranks for ties
    sorted_scores = scores[order]
    ranks_sorted = torch.empty(n, device=scores.device, dtype=torch.float32)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg = 0.5 * (i + j - 1)
        ranks_sorted[i:j] = avg
        i = j
    ranks[order] = ranks_sorted
    return ranks / float(n - 1)


def bounded_admission(
    keep: torch.Tensor,
    early_feat: torch.Tensor,
    early_scores: torch.Tensor,
    late_scores: torch.Tensor,
    cap: int,
) -> tuple[torch.Tensor, dict]:
    """Handoff A: swap up to ``cap`` Dual keep slots for positive L9–late disagreement.

    Payload features stay penultimate; L9 is selection statistics only.
    No permanent survivors; K unchanged. See METHOD_AND_CONTROLS §3.
    """
    cap = int(cap)
    k = int(keep.numel())
    n = int(late_scores.numel())
    if cap <= 0 or k == 0:
        return keep, {"admission_cap": cap, "swapped": 0, "action": "noop"}
    device = keep.device
    s0 = torch.zeros(n, dtype=torch.bool, device=device)
    s0[keep] = True
    r9 = _rank_percentile(early_scores.float())
    r_late = _rank_percentile(late_scores.float())
    d = torch.clamp(r9 - r_late, min=0.0)
    feat = torch.nn.functional.normalize(early_feat.float(), dim=-1, eps=1e-6)
    # max cosine to current keep set
    keep_feat = feat.index_select(0, keep)
    # (n, k) -> max over keep
    cos = torch.clamp(feat @ keep_feat.transpose(0, 1), 0.0, 1.0).max(dim=1).values
    t = d * (1.0 - cos)
    t = torch.where(s0, torch.zeros_like(t), t)
    # protect high positive-disagreement keep members from being donors
    d_keep = d.index_select(0, keep)
    protect_b = min(cap, k)
    if protect_b > 0:
        prot_local = torch.topk(d_keep, k=protect_b, largest=True).indices
        protected = torch.zeros(k, dtype=torch.bool, device=device)
        protected[prot_local] = True
    else:
        protected = torch.zeros(k, dtype=torch.bool, device=device)
    # candidate outsiders with positive priority
    cand_mask = (~s0) & (t > 0)
    cand_ids = torch.where(cand_mask)[0]
    if cand_ids.numel() == 0:
        return keep, {"admission_cap": cap, "swapped": 0, "action": "no_positive_t"}
    cand_scores = t.index_select(0, cand_ids)
    take = min(cap, int(cand_ids.numel()))
    pick_local = torch.topk(cand_scores, k=take, largest=True).indices
    incoming = cand_ids.index_select(0, pick_local)
    # donors: lowest late score among unprotected keep
    late_keep = late_scores.float().index_select(0, keep)
    donor_scores = late_keep.clone()
    donor_scores[protected] = float("inf")
    # need `take` finite donors
    n_donor = min(take, int((~protected).sum().item()))
    if n_donor <= 0:
        return keep, {"admission_cap": cap, "swapped": 0, "action": "no_donor"}
    incoming = incoming[:n_donor]
    donor_local = torch.topk(donor_scores, k=n_donor, largest=False).indices
    donor_ids = keep.index_select(0, donor_local)
    new_keep = keep.clone()
    new_keep[donor_local] = incoming
    new_keep, _ = torch.sort(new_keep)
    if int(new_keep.unique().numel()) != k:
        raise RuntimeError("bounded_admission produced duplicate keep IDs")
    return new_keep, {
        "admission_cap": cap,
        "swapped": int(n_donor),
        "incoming": incoming.detach().cpu().tolist(),
        "donors": donor_ids.detach().cpu().tolist(),
        "action": "swap",
    }


def select_existing_forward(image_forward_outs, tokens_0):
    config = json.loads(Path(os.environ["ELVA_R3_CONFIG"]).read_text())
    hidden = image_forward_outs.hidden_states
    pen = hidden[-2][0, 1:]
    if pen.shape != tokens_0.shape or pen.device != tokens_0.device or not torch.equal(pen, tokens_0):
        raise RuntimeError("ELVA payload is not the pen rows already used as projector input")
    stage = config["method"]["stage1"]
    guide = stage["guide"]
    beta = float(stage.get("beta") or 0.0)
    score_guide = stage.get("score_guide", "same")
    sg = guide if score_guide == "same" else score_guide
    admission_cap = int(stage.get("admission_cap") or 0)
    need_early = (
        guide == "l9" or (guide == "dual" and beta > 0)
        or sg == "l9" or (sg == "dual" and beta > 0)
        or admission_cap > 0
    )
    early = hidden[9][0, 1:] if need_early else None
    pen_scores = _sum_cls(image_forward_outs.attentions, -2)
    early_scores = _sum_cls(image_forward_outs.attentions, 8) if need_early else None
    ids = torch.arange(pen.shape[0], device=pen.device, dtype=torch.long)
    family = stage["cover_family"]
    cover = _covers()[family] if family != "none" else None
    selected = select_checked(
        pen, pen_scores, ids, config,
        early=early, early_scores=early_scores,
        early_ids=ids if need_early else None,
        native_cover=cover, backend="torch")
    keep = selected["keep"]
    audit = selected["stage1_audit"]
    if admission_cap > 0:
        if early is None or early_scores is None:
            raise RuntimeError("admission_cap>0 requires L9 features/scores")
        keep, admit_audit = bounded_admission(
            keep, early, early_scores, pen_scores, admission_cap
        )
        audit = dict(audit)
        audit["admission"] = admit_audit
    # Always record entry COVER priority (B2 ENTRY_ORDER; cheap; others ignore).
    slots = int(audit.get("replacement_slots") or 0)
    _set_entry_priority(selected["metric"], selected["scores"], keep, slots)
    global LAST_ENTRY_KEEP, LAST_ENTRY_METRIC
    LAST_ENTRY_KEEP = keep.detach().long().clone()
    # Dual (or pen) metric features over all 576 patches — QCP similarity source.
    LAST_ENTRY_METRIC = selected["metric"].detach().clone()
    payload = pen.index_select(0, keep)
    if payload.shape[0] != int(keep.numel()) or int(keep.unique().numel()) != int(keep.numel()):
        raise RuntimeError("admission/cover keep is not a unique pen-row index set")
    audit_dir = os.environ.get("ELVA_R3_AUDIT_DIR", "")
    if audit_dir:
        path = Path(audit_dir) / "stage1.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(ids=keep.detach().cpu().tolist(), audit=audit,
                   score_rule="sum_heads_cls_to_patch")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return keep, audit
