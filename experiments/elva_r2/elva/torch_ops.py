"""CUDA-resident Stage1 path. Server parity test is mandatory before ACC.

No attention backend is changed here. Existing COVER stays source-bound.
CPU reference in evidence.py is the tested mathematical specification.
"""
import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def select_stage1(pen, pen_scores, ids, k, *, guide="pen", early=None,
                  early_scores=None, early_ids=None, beta=.5, cover=None,
                  use_cover=False, context=None, frozen_keep=None, debug=True,
                  score_guide="same"):
    if pen.ndim != 2 or min(pen.shape) == 0 or not pen.is_floating_point():
        raise ValueError("floating raw pen [N,D] required")
    n = len(pen)
    if (pen_scores.shape != (n,) or ids.shape != (n,) or ids.dtype != torch.long
            or pen_scores.device != pen.device or ids.device != pen.device
            or not pen_scores.is_floating_point() or type(k) is not int or not 0 < k <= n):
        raise ValueError("aligned source saliency/IDs and valid K required")
    if guide not in ("pen", "l9", "dual") or not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("invalid guide/beta")
    sg = guide if score_guide == "same" else score_guide
    if sg not in ("pen", "l9", "dual"):
        raise ValueError("invalid score guide")
    needs_early = (guide == "l9" or (guide == "dual" and beta > 0)
                   or sg == "l9" or (sg == "dual" and beta > 0))
    if needs_early:
        if (early is None or early.shape != pen.shape or early.device != pen.device
                or not early.is_floating_point() or early_scores is None
                or early_scores.shape != (n,) or early_scores.device != pen.device
                or not early_scores.is_floating_point() or early_ids is None
                or early_ids.shape != ids.shape or early_ids.dtype != torch.long
                or early_ids.device != pen.device):
            raise ValueError("aligned captured early features/scores/IDs required")
    if debug:
        tensors = [pen, pen_scores]+([early, early_scores] if needs_early else [])
        if (any(not bool(torch.isfinite(x).all()) for x in tensors)
                or bool((pen_scores < 0).any()) or ids.unique().numel() != n or bool((ids < 0).any())
                or (needs_early and (not torch.equal(ids, early_ids) or bool((early_scores < 0).any())))):
            raise ValueError("nonfinite/negative/misaligned evidence")
    if guide == "pen" or (guide == "dual" and beta == 0):
        metric, scores = pen, pen_scores
    elif guide == "l9" or beta == 1:
        metric, scores = early, early_scores
    else:
        metric = torch.cat((math.sqrt(1-beta)*F.normalize(pen.float(), dim=-1, eps=1e-12),
                            math.sqrt(beta)*F.normalize(early.float(), dim=-1, eps=1e-12)), dim=-1)
    scores = pen_scores if sg == "pen" or (sg == "dual" and beta == 0) else (
        early_scores if sg == "l9" or beta == 1 else (1-beta)*pen_scores.float()+beta*early_scores.float())
    if frozen_keep is not None:
        keep = frozen_keep
    elif use_cover:
        if cover is None:
            raise RuntimeError("BLOCKED_SOURCE: bind original COVER")
        keep = cover(metric_features=metric.clone(), scores=scores.clone(),
                     patch_ids=ids.clone(), k=k, context=context)
    else:
        keep = torch.argsort(scores, descending=True, stable=True)[:k]
    if not isinstance(keep, torch.Tensor) or keep.shape != (k,) or keep.dtype != torch.long or keep.device != pen.device:
        raise ValueError("native selector must return device-local long [K]")
    if debug and (keep.unique().numel() != k or int(keep.min()) < 0 or int(keep.max()) >= n):
        raise ValueError("invalid/duplicate selected local indices")
    keep = keep.sort().values
    return dict(payload=pen.index_select(0, keep), keep=keep,
                ids=ids.index_select(0, keep), metric=metric, scores=scores)


@torch.no_grad()
def question_visual_mass(logits, allowed, visual_columns):
    if logits.ndim not in (2, 3) or min(logits.shape) == 0 or allowed.dtype != torch.bool:
        raise ValueError("partial [Q,T]/[H,Q,T] logits and boolean visible mask required")
    if allowed.device != logits.device or visual_columns.device != logits.device or visual_columns.dtype != torch.long:
        raise ValueError("device-local masks/column indices required")
    mask = torch.broadcast_to(allowed, logits.shape)
    # Validate once in server preflight; this function deliberately makes its
    # synchronization cost visible in end-to-end timing.
    if (not bool(mask.any(-1).all()) or not bool(torch.isfinite(logits[mask]).all())
            or visual_columns.ndim != 1 or visual_columns.numel() == 0
            or visual_columns.unique().numel() != visual_columns.numel()
            or int(visual_columns.min()) < 0 or int(visual_columns.max()) >= logits.shape[-1]):
        raise ValueError("invalid visible rows/visual indices")
    p = logits.float().masked_fill(~mask, -torch.inf).softmax(-1)
    return p.index_select(-1, visual_columns).mean(dim=tuple(range(logits.ndim-1)))
