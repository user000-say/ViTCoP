"""Isolated E018 inference path for the attached, Transformers-4.37-era LLaVA.

Batch=1, single image, SDPA, unpadded prefill and one-token decode only.
No GT enters this module. The observer exists only for offline prefix capture.
"""
import copy
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from .e018_contract import E018Config, FEATURE_NAMES, SCHEMA

_STAGE1 = ContextVar("e018_stage1", default=None)


def publish_stage1(patch_ids, core_ids, cls_attention, name, exchange_budget):
    ids = patch_ids.detach().long().clone()
    if ids.ndim != 1 or ids.unique().numel() != ids.numel():
        raise ValueError("Stage1 patch IDs must be a unique ordered vector")
    _STAGE1.set(dict(patch_ids=ids, added=~torch.isin(ids, core_ids),
                     cls=cls_attention.detach().float().clone(),
                     stage1=dict(name=str(name), exchange_budget=int(exchange_budget))))


def take_stage1():
    value = _STAGE1.get()
    _STAGE1.set(None)
    if value is None:
        raise RuntimeError("Missing current-request Stage1 provenance. Prepare image or publish saved metadata.")
    return value


def clear_stage1():
    _STAGE1.set(None)


def clone_cache(cache):
    out = DynamicCache()
    for i, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
        out.update(k.detach().clone(), v.detach().clone(), i)
    return out


@dataclass
class State:
    visual_start: int
    visual_ids: torch.Tensor
    visual_positions: torch.Tensor
    positions: torch.Tensor
    question_indices: torch.Tensor
    metadata: dict
    initial_n: int
    original_next: int
    fingerprint: str
    cache_positions: dict = field(default_factory=dict)
    pending: list = field(default_factory=list)
    history: dict = field(default_factory=dict)
    receiver: object = None
    events: list = field(default_factory=list)
    per_layer_n: list = field(default_factory=list)
    forced_keeps: dict = field(default_factory=dict)
    physical_events: int = 0
    physical_layers: list = field(default_factory=list)
    gather_operations: int = 0


class Controller:
    def __init__(self, path):
        self.data = json.loads(Path(path).read_text())
        d = self.data
        if d.get("schema") != SCHEMA or d.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Controller schema/feature names mismatch")
        if not d.get("calibrated") or not math.isfinite(d.get("q_upper", float("inf"))):
            raise ValueError("Controller lacks finite calibration")
        n = len(FEATURE_NAMES)
        width = len(d.get("b1", []))
        valid = (width > 0 and len(d.get("mean", [])) == n and len(d.get("scale", [])) == n
                 and len(d.get("w1", [])) == n and all(len(v) == width for v in d["w1"])
                 and len(d.get("w2", [])) == width and all(len(v) == 1 for v in d["w2"])
                 and len(d.get("b2", [])) == 1 and all(v > 0 for v in d["scale"]))
        if not valid:
            raise ValueError("Invalid controller parameter shapes/scales")
        def flat(v):
            if isinstance(v, list):
                for item in v:
                    yield from flat(item)
            else:
                yield v
        if not all(math.isfinite(v) for k in ("mean", "scale", "w1", "b1", "w2", "b2") for v in flat(d[k])):
            raise ValueError("Nonfinite controller parameters")
        mask = d.get("feature_mask", [1.] * n)
        if len(mask) != n or any(v not in (0, 1) for v in mask):
            raise ValueError("Invalid feature mask")
        self.tensors = {}

    def predict(self, x, fingerprint):
        if self.data.get("fingerprint") != fingerprint:
            raise ValueError("Controller protocol mismatch: weights/Stage1/selector/cache/schedule changed")
        key = str(x.device)
        if key not in self.tensors:
            self.tensors[key] = {k: torch.tensor(self.data[k], device=x.device, dtype=torch.float32)
                                for k in ("mean", "scale", "w1", "b1", "w2", "b2", "q_upper")}
        t = self.tensors[key]
        z = ((x - t["mean"]) / t["scale"]).clamp(-8, 8)
        keep = self.data.get("feature_mask", [1.] * len(FEATURE_NAMES))
        if "mask" not in t:
            t["mask"] = torch.tensor(keep, device=x.device)
        h = torch.tanh((z * t["mask"]) @ t["w1"] + t["b1"])
        pred = (h @ t["w2"] + t["b2"]).squeeze()
        return pred, pred + t["q_upper"]


def question_response(q, k, mask, q_idx, v0, vn, need_redundancy=True):
    """Full legal-prefix softmax, THEN take visual columns. Never visual-only softmax."""
    logits = q.index_select(2, q_idx).float() @ k.float().transpose(-2, -1)
    logits /= math.sqrt(q.shape[-1])
    if mask is None:
        illegal = torch.arange(k.shape[-2], device=q.device)[None, :] > q_idx[:, None]
        logits = logits.masked_fill(illegal[None, None], torch.finfo(logits.dtype).min)
    else:
        logits = logits + mask.index_select(2, q_idx).float()
    visual = logits.softmax(-1).mean(1)[0, :, v0:v0 + vn]
    g = visual.sum(-1)
    p = visual / g[:, None].clamp_min(1e-8)
    w = g / g.sum().clamp_min(1e-8)
    r = (w[:, None] * p).sum(0)
    r = r / r.sum().clamp_min(1e-8)
    z = F.normalize(p.T * w.clamp_min(1e-8).sqrt()[None, :], dim=-1)
    if need_redundancy:
        sim = (z @ z.T).clamp(0, 1)
        higher = r[None, :] > r[:, None]
        rho = sim.masked_fill(~higher, -1).max(-1).values.clamp_min(0)
    else:
        rho = torch.zeros_like(r)
    entropy = -(r * r.clamp_min(1e-8).log()).sum() / math.log(max(vn, 2))
    return dict(g=g, p=p, r=r, rho=rho, entropy=entropy, signatures=z, need_redundancy=need_redundancy,
                q_attn=visual.mean(0), qed=r * (1 - entropy * rho),
                visual=visual)



def _e020_load_path():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3] / "e020_lib"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _e020_from_env():
    """First-event continuation / controls. Mutually exclusive with E019 final refine."""
    if os.environ.get("E020_ENABLE", "False") != "True":
        return None
    kind = os.environ.get("E020_KIND", "reference").strip().lower()
    if kind in {"", "reference", "off", "none"}:
        return None
    _e020_load_path()
    from module_torch import ContinuationHook
    import json
    spec = {
        "kind": kind,
        "model": os.environ.get("E020_MODEL", "current"),
        "group_size": int(os.environ.get("E020_GROUP_SIZE", "1")),
        "rounds": int(os.environ.get("E020_ROUNDS", "2")),
        "shortlist": int(os.environ.get("E020_SHORTLIST", "16")),
        "threshold": float(os.environ.get("E020_THRESHOLD", "0")),
        "alpha": float(os.environ.get("E020_ALPHA", "0")),
        "lambda": float(os.environ.get("E020_LAMBDA", "0.15")),
        "current_weight": float(os.environ.get("E020_CURRENT_WEIGHT", "0.6")),
        "seed": int(os.environ.get("E020_SEED", "20")),
    }
    # Optional learned scorer path; required only for kind=continuation
    model = None
    model_path = os.environ.get("E020_MODEL_PATH", "").strip()
    if kind == "continuation":
        if not model_path:
            raise RuntimeError("E020 continuation requires E020_MODEL_PATH fitted checkpoint")
        from module_torch import PairScorer
        model = PairScorer(model_path)
    debug = os.environ.get("E020_DEBUG", "True") == "True"
    # Fixed random projection created lazily on first hidden use if needed
    hook = ContinuationHook(model=model, projection=None, debug=debug)
    return {"hook": hook, "spec": spec}


def _e019_from_env():
    """Optional final-event EvidenceHook; reference / disabled leaves STATIC path unchanged."""
    if os.environ.get("E019_ENABLE", "False") != "True":
        return None
    mode = os.environ.get("E019_MODE", "reference").strip().lower()
    if mode in {"", "reference", "off", "none"}:
        return None
    from .e019_selector_hook import EvidenceHook
    cfg = {
        "mode": mode,
        "alpha": float(os.environ.get("E019_ALPHA", "0")),
        "lambda": float(os.environ.get("E019_LAMBDA", "0")),
        "swaps": int(os.environ.get("E019_SWAPS", "0")),
    }
    debug = os.environ.get("E019_DEBUG", "True") == "True"
    return EvidenceHook(cfg, debug_validate=debug)


def _group_query_response(visual, groups=4):
    """Average question→vision rows into ≤groups fixed contiguous buckets."""
    q = int(visual.shape[0])
    g = max(1, min(int(groups), q))
    # Contiguous map over original question-token order; short prompts use fewer groups.
    ids = torch.arange(q, device=visual.device)
    ids = (ids * g) // max(q, 1)
    ids = ids.clamp_max(g - 1)
    out = visual.new_zeros((g, visual.shape[1]))
    out.index_add_(0, ids, visual.float())
    counts = torch.bincount(ids, minlength=g).clamp_min(1).to(out.dtype)[:, None]
    return out / counts


class Runtime:
    def __init__(self, model, cfg=None):
        self.model = model
        self.cfg = cfg or E018Config.from_env()
        # COLLECT may replay a previously trained policy before the captured
        # event. GT is still read only AFTER the current state has been frozen.
        self.controller = Controller(self.cfg.controller) if self.cfg.mode == "ONLINE" or (
            self.cfg.mode == "COLLECT" and self.cfg.controller) else None
        self.state = None
        self.observer = None
        self._side = None
        self.last_trace = None
        self._sketch = {}
        self.e019_hook = _e019_from_env()
        self.e019_groups = int(os.environ.get("E019_QUERY_GROUPS", "4"))
        self._e019_sample_seed = int(os.environ.get("E019_SAMPLE_SEED", "0"))
        self.e020 = _e020_from_env()
        # E020 first-event refine disables E019 final refine on the same request.
        if self.e020 is not None:
            self.e019_hook = None

    def _read(self, q, k, value, mask):
        s = self.state
        need = self.cfg.mode in {"COLLECT", "ONLINE"} or self.cfg.selector == "QED" or self.observer is not None
        self._side = question_response(q, k, mask, s.question_indices,
                                       s.visual_start, s.visual_ids.numel(), need)

    def _pick(self, side, count, layer):
        s = self.state
        if layer in s.forced_keeps:
            ids = s.forced_keeps[layer].to(s.visual_ids.device)
            selected = torch.where(torch.isin(s.visual_ids, ids))[0]
            if selected.numel() != count or ids.numel() != count:
                raise RuntimeError("Frozen future survivors absent from current state")
            return selected
        if self.cfg.selector == "RANDOM":
            # Control is keyed by original patch ID, not a changing local coordinate.
            score = ((s.visual_ids * 1103515245 + 12345 + layer * 2654435761) % 2147483647).float()
        else:
            score = side["q_attn" if self.cfg.selector == "Q_ATTN" else "qed"]
        baseline = torch.argsort(score, descending=True, stable=True)[:count].sort().values
        # E020: refine ONLY the FIRST progressive prune (continuation / controls).
        if (self.e020 is not None and self.cfg.events and layer == self.cfg.events[0][0]):
            visual = side.get("visual")
            if visual is None:
                raise RuntimeError("E020 requires question→vision response matrix")
            current = _group_query_response(visual, self.e019_groups)
            _e020_load_path()
            from dispatch import refine as e020_refine
            spec = self.e020["spec"]
            hook = self.e020["hook"]
            next_after = int(os.environ.get(
                "E020_NEXT_AFTER",
                str(self.cfg.events[-1][0] + 1)))  # completed-block horizon
            after = int(layer + 1)  # zero-based layer → completed blocks
            gen = None
            if spec["kind"] == "random":
                gen = torch.Generator(device=score.device)
                seed = int(spec.get("seed", 20))
                if s.visual_ids.numel():
                    seed = int((seed + s.visual_ids.long().sum() % 1000003).item())
                gen.manual_seed(seed)
            # Optional decoder projection for learned/random feature path
            hidden_vis = None
            if spec["kind"] in ("continuation", "random"):
                # Prefer not requiring hidden; random mode works with scores/response only
                # if select gets features — module_torch select needs token_features.
                # For random, dispatch still builds features; hidden optional.
                pass
            refined = e020_refine(
                spec, hook, score, current, baseline, s.visual_ids,
                after=after, depth=len(self.model.layers), next_after=next_after,
                final_k=int(self.cfg.events[-1][1]),
                hidden=hidden_vis, encoder_rank=None, patch_xy=None, generator=gen)
            if refined.numel() != count:
                raise RuntimeError("E020 refine must preserve event K")
            return refined.sort().values
        # E019: refine ONLY the final progressive prune; penultimate only captures.
        if self.e019_hook is not None and self.cfg.events and layer == self.cfg.events[-1][0]:
            visual = side.get("visual")
            if visual is None:
                raise RuntimeError("E019 requires question→vision response matrix")
            current = _group_query_response(visual, self.e019_groups)
            seed = int(self._e019_sample_seed)
            if seed == 0 and s.visual_ids.numel():
                # Stable per-request seed from original patch IDs (not launch order).
                seed = int((s.visual_ids.long().sum() % 2147483647).item())
            refined = self.e019_hook.refine(
                score, current, s.visual_ids, baseline, sample_seed=seed)
            if refined.numel() != count:
                raise RuntimeError("E019 refine must preserve final K")
            return refined.sort().values
        return baseline

    def _features(self, hidden, side, keep, layer, endpoint):
        s, n = self.state, self.state.visual_ids.numel()
        removed = torch.ones(n, device=hidden.device, dtype=torch.bool)
        removed[keep] = False
        drop = torch.where(removed)[0]
        r, rho = side["r"], side["rho"]
        score = side["q_attn" if self.cfg.selector in {"Q_ATTN", "RANDOM"} else "qed"]
        zero = r.new_zeros(())
        receiver = hidden[0].index_select(0, s.question_indices).float().mean(0)
        norm_receiver = F.normalize(receiver, dim=0)
        receiver_drift = zero if s.receiver is None else 1 - F.cosine_similarity(receiver, s.receiver, dim=0)
        drift = zero
        if s.history:
            previous_ids, previous_r = s.history["ids"], s.history["r"]
            match = s.visual_ids[:, None] == previous_ids[None, :]
            old = (match.float() @ previous_r)
            old = old / old.sum().clamp_min(1e-8)
            drift = (r - old).abs().sum() / 2
        key = (hidden.device, receiver.numel())
        if key not in self._sketch:
            gen = torch.Generator(device="cpu").manual_seed(20261014)
            self._sketch[key] = (torch.randint(0, 2, (receiver.numel(), 8), generator=gen).float() * 2 - 1).to(hidden.device)
        sketch = norm_receiver @ self._sketch[key] / math.sqrt(receiver.numel())
        all_ids = s.metadata["patch_ids"].to(hidden.device)
        match = s.visual_ids[:, None] == all_ids[None, :]
        added = (match.float() @ s.metadata["added"].to(hidden.device).float())
        cls = s.metadata["cls"].to(hidden.device)
        cls = cls / cls.sum().clamp_min(1e-8)
        cls = cls.index_select(0, s.visual_ids)
        norms = hidden[0, s.visual_start:s.visual_start + n].float().norm(dim=-1)
        delta = drop.numel() * (endpoint - layer) / (s.initial_n * len(self.model.layers))
        coherence = side["signatures"].index_select(0, drop).mean(0).norm()
        vals = [layer / len(self.model.layers), n / s.initial_n, keep.numel() / s.initial_n,
                drop.numel() / n, delta, endpoint / len(self.model.layers), s.physical_events / 2,
                side["g"].mean(), side["g"].min(), side["entropy"], r[drop].sum(),
                side["g"].mean() * r[drop].sum(), score[drop].max(),
                score[keep].min() - score[drop].max(), rho[drop].mean(), rho[keep].mean(),
                drift, float(bool(s.history)), receiver_drift,
                receiver.norm() / math.sqrt(receiver.numel()), norms[drop].mean() / math.sqrt(hidden.shape[-1]),
                norms[keep].mean() / math.sqrt(hidden.shape[-1]), added.mean(), added[drop].mean(),
                (added[drop] * r[drop]).sum(), cls[drop].sum(), cls[keep].sum(),
                s.metadata["stage1"]["exchange_budget"] / s.initial_n, coherence,
                float(self.cfg.cache_policy == "scrub_all")]
        x = torch.stack([v.float() if torch.is_tensor(v) else r.new_tensor(v) for v in vals] + list(sketch))
        if x.numel() != len(FEATURE_NAMES) or not torch.isfinite(x).all():
            raise RuntimeError("Invalid continuation feature vector")
        return x, delta, receiver

    def _drop_ids(self, hidden, cache, dropped):
        s = self.state
        remove = torch.isin(s.visual_ids, dropped)
        if not bool(remove.any()):
            return hidden
        s.gather_operations += 1
        keep_v = torch.where(~remove)[0]
        old_n = s.visual_ids.numel()
        keep_rows = torch.cat((torch.arange(s.visual_start, device=hidden.device),
                               keep_v + s.visual_start,
                               torch.arange(s.visual_start + old_n, hidden.shape[1], device=hidden.device)))
        dropped_positions = s.visual_positions[remove]
        hidden = hidden.index_select(1, keep_rows)
        s.positions = s.positions.index_select(0, keep_rows)
        # All question rows are after the single image; every text row is retained.
        s.question_indices = s.question_indices - int(remove.sum().item())
        s.visual_ids = s.visual_ids.index_select(0, keep_v)
        s.visual_positions = s.visual_positions.index_select(0, keep_v)
        if self.cfg.cache_policy == "scrub_all":
            for layer, positions in list(s.cache_positions.items()):
                keep_cache = torch.where(~torch.isin(positions, dropped_positions))[0]
                cache.key_cache[layer] = cache.key_cache[layer].index_select(2, keep_cache)
                cache.value_cache[layer] = cache.value_cache[layer].index_select(2, keep_cache)
                s.cache_positions[layer] = positions.index_select(0, keep_cache)
        return hidden

    def _record_physical(self, layer):
        if layer not in self.state.physical_layers:
            self.state.physical_layers.append(layer)
            self.state.physical_events += 1

    def _minimum_future_work(self, layer, candidate_drop, endpoint):
        s = self.state
        pending = [(t, ids.numel()) for t, ids in s.pending] + [(endpoint, candidate_drop.numel())]
        n, work = s.visual_ids.numel(), sum(s.per_layer_n)
        schedule = dict(self.cfg.events)
        for i in range(layer + 1, len(self.model.layers)):
            work += n
            n -= sum(count for t, count in pending if t == i)
            if i in schedule:
                n = min(n, schedule[i])
        return work

    def _event(self, hidden, cache, side, layer, target, endpoint):
        s = self.state
        if target >= s.visual_ids.numel():
            return hidden
        keep = self._pick(side, target, layer)
        if self.cfg.mode in {"COLLECT", "ONLINE"} or self.observer is not None:
            x, delta, receiver = self._features(hidden, side, keep, layer, endpoint)
        else:
            x, receiver = None, None
            delta = (s.visual_ids.numel() - target) * (endpoint - layer) / (s.initial_n * len(self.model.layers))
        mask = torch.ones(s.visual_ids.numel(), dtype=torch.bool, device=hidden.device)
        mask[keep] = False
        dropped = s.visual_ids[mask].clone()
        if self.observer is not None:
            if self.cfg.mode == "ONLINE":
                raise RuntimeError("Offline observer cannot run in ONLINE mode")
            self.observer(dict(layer=layer, endpoint=endpoint, keep=keep.clone(),
                               drop_ids=dropped, features=x, delta_work=delta,
                               hidden=hidden, cache=cache, state=s, side_r=side["r"].clone()))
        continue_it, pred, upper = False, None, None
        if self.cfg.mode == "PROXY":
            continue_it = bool((side["q_attn"][mask].sum() > self.cfg.proxy_threshold).item())
        if self.controller is not None:
            pred_t, upper_t = self.controller.predict(x, s.fingerprint)
            continue_it = bool((upper_t > self.cfg.lam * delta).item())
            if self.cfg.trace:
                pred, upper = float(pred_t.item()), float(upper_t.item())
        if continue_it and self.cfg.max_work and self._minimum_future_work(layer, dropped, endpoint) > self.cfg.max_work:
            continue_it = False
        row = dict(layer=layer, target=target, endpoint=endpoint,
                   action="continue" if continue_it else "retire", delta_work=delta,
                   keep_ids=s.visual_ids[keep].clone(), drop_ids=dropped,
                   predicted=pred, upper=upper)
        s.events.append(row)
        s.history = dict(ids=s.visual_ids.clone(), r=side["r"].detach().clone())
        s.receiver = receiver.detach() if receiver is not None else None
        if continue_it:
            s.pending.append((endpoint, dropped))
            return hidden
        return self._drop_ids(hidden, cache, dropped)

    def _tail(self, hidden, cache, start=0, output_hidden_states=False):
        s, history = self.state, () if output_hidden_states else None
        schedule = dict(self.cfg.events)
        endpoint_map = {layer: (self.cfg.events[j + 1][0] if j + 1 < len(self.cfg.events)
                                 else len(self.model.layers) - 1)
                        for j, (layer, _) in enumerate(self.cfg.events)}
        for layer_idx in range(start, len(self.model.layers)):
            layer = self.model.layers[layer_idx]
            n_before = s.visual_ids.numel()
            s.per_layer_n.append(n_before)
            if output_hidden_states:
                history += (hidden,)
            self._side = None
            layer.self_attn._e006_enabled = False
            layer.self_attn._e010_collect_local_influence = False
            # E020: also read prompt→vision at layer 0 for history capture before first prune.
            e020_capture = (self.e020 is not None and layer_idx == 0
                            and self.cfg.events and self.cfg.events[0][0] > 0)
            layer.self_attn._e018_callback = (
                self._read if (layer_idx in schedule or e020_capture) else None)
            try:
                output = layer(hidden, attention_mask=None, position_ids=s.positions[None],
                               past_key_value=cache, use_cache=True, output_attentions=False)
            finally:
                layer.self_attn._e018_callback = None
            hidden = output[0]
            s.cache_positions[layer_idx] = s.positions.clone()
            # Complete a previous deferral first; restrict the new candidate set.
            due = [ids for t, ids in s.pending if t == layer_idx]
            s.pending = [(t, ids) for t, ids in s.pending if t != layer_idx]
            old_ids = s.visual_ids.clone()
            if due:
                hidden = self._drop_ids(hidden, cache, torch.cat(due))
                if s.visual_ids.numel() != n_before:
                    self._record_physical(layer_idx)
            side = self._side
            if e020_capture and side is not None:
                vis = side.get("visual")
                if vis is None:
                    raise RuntimeError("E020 capture missing visual response at layer 0")
                grouped = _group_query_response(vis, self.e019_groups)
                self.e020["hook"].capture(grouped, s.visual_ids)
            if layer_idx in schedule:
                if side is None:
                    raise RuntimeError("E018 SDPA callback was not executed; wrong attention implementation")
                if s.visual_ids.numel() != old_ids.numel():
                    keep_side = torch.where(torch.isin(old_ids, s.visual_ids))[0]
                    for k in ("r", "rho", "q_attn", "qed", "signatures"):
                        side[k] = side[k].index_select(0, keep_side)
                    side["p"] = side["p"].index_select(1, keep_side)
                    if "visual" in side:
                        side["visual"] = side["visual"].index_select(1, keep_side)
                    # Recompute conditional statistics on the surviving set; retain full-prefix mass.
                    visual = side["p"] * side["g"][:, None]
                    side["g"] = visual.sum(-1)
                    side["p"] = visual / side["g"][:, None].clamp_min(1e-8)
                    side["visual"] = visual
                    side["q_attn"] = visual.mean(0)
                    w = side["g"] / side["g"].sum().clamp_min(1e-8)
                    side["r"] = (w[:, None] * side["p"]).sum(0)
                    side["entropy"] = -(side["r"] * side["r"].clamp_min(1e-8).log()).sum() / math.log(max(s.visual_ids.numel(), 2))
                    z = F.normalize(side["p"].T * w.clamp_min(1e-8).sqrt()[None], dim=-1)
                    side["signatures"] = z
                    if side["need_redundancy"]:
                        sim = (z @ z.T).clamp(0, 1)
                        side["rho"] = sim.masked_fill(~(side["r"][None] > side["r"][:, None]), -1).max(-1).values.clamp_min(0)
                    else:
                        side["rho"] = torch.zeros_like(side["r"])
                    side["qed"] = side["r"] * (1 - side["entropy"] * side["rho"])
                # E019: capture query-group response at penultimate prune only.
                if (self.e019_hook is not None and len(self.cfg.events) >= 2
                        and layer_idx == self.cfg.events[-2][0]):
                    vis = side.get("visual")
                    if vis is None:
                        raise RuntimeError("E019 penultimate capture missing visual response")
                    grouped = _group_query_response(vis, self.e019_groups)
                    self.e019_hook.capture(grouped, s.visual_ids)
                hidden = self._event(hidden, cache, side, layer_idx, schedule[layer_idx], endpoint_map[layer_idx])
            if s.visual_ids.numel() != n_before:
                self._record_physical(layer_idx)
        hidden = self.model.norm(hidden)
        if output_hidden_states:
            history += (hidden,)
        if s.pending:
            raise RuntimeError("Unexecuted continuation endpoint")
        if s.physical_events > 3:
            raise RuntimeError("Unexpected fourth physical pruning event")
        if s.gather_operations > 3:
            raise RuntimeError("Unexpected fourth hidden-state gather")
        # trace() synchronizes tensors with CPU. Never execute it in timing mode.
        self.last_trace = self.trace(cache) if self.cfg.trace or self.cfg.mode == "COLLECT" else None
        if self.cfg.trace:
            path = Path(self.cfg.trace)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(json.dumps(self.last_trace) + "\n")
        return hidden, cache, history

    def trace(self, cache):
        s = self.state
        return dict(schema=SCHEMA, fingerprint=s.fingerprint, stage1=s.metadata["stage1"],
                    mode=self.cfg.mode, cache_policy=self.cfg.cache_policy,
                    per_layer_n=s.per_layer_n, token_work=sum(s.per_layer_n),
                    average_visual=sum(s.per_layer_n) / len(self.model.layers),
                    final_ids=s.visual_ids.detach().cpu().tolist(),
                    original_next=s.original_next, physical_events=s.physical_events,
                    physical_layers=s.physical_layers, gather_operations=s.gather_operations,
                    cache_lengths=[int(k.shape[2]) for k in cache.key_cache],
                    events=[{k: v.detach().cpu().tolist() if torch.is_tensor(v) else v
                             for k, v in row.items()} for row in s.events])

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,
                image_shape=576, token_length_list=(), pre_prompt_length_list=(),
                e006_question_span=None, **unused):
        m = self.model
        actual_attentions = m.config.output_attentions if output_attentions is None else output_attentions
        if m.training or use_cache is False or actual_attentions or getattr(m, "_use_flash_attention_2", False):
            raise ValueError("E018 requires frozen eval, use_cache=True, SDPA and output_attentions=False")
        if not getattr(m, "_use_sdpa", False):
            raise ValueError("E018 requires the patched LlamaSdpaAttention")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids/inputs_embeds")
        hidden = m.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if hidden.shape[0] != 1:
            raise ValueError("E018 supports batch=1 only")
        if attention_mask is not None and (attention_mask.ndim != 2 or not bool(attention_mask.bool().all())):
            raise ValueError("E018 supports an unpadded 2-D all-ones attention mask only")
        legacy = not isinstance(past_key_values, Cache)
        cache = DynamicCache.from_legacy_cache(past_key_values) if legacy else past_key_values
        if not isinstance(cache, DynamicCache):
            raise ValueError("Only DynamicCache/legacy tuple supported")
        is_prefill = cache.get_seq_length() == 0
        output_hidden_states = bool(output_hidden_states if output_hidden_states is not None else m.config.output_hidden_states)
        return_dict = m.config.use_return_dict if return_dict is None else return_dict
        if is_prefill:
            metadata = take_stage1()
            n = int(image_shape)
            if n != metadata["patch_ids"].numel() or len(pre_prompt_length_list) != 1:
                raise ValueError("Stage1 IDs, image_shape and one visual-start entry required")
            self.cfg.validate(len(m.layers), n)
            v0 = int(pre_prompt_length_list[0])
            span = e006_question_span
            if span is None or len(span) != 2:
                raise ValueError("Explicit expanded-multimodal user-question span required")
            a, b = map(int, span)
            if not 0 <= v0 < v0 + n <= a < b <= hidden.shape[1]:
                raise ValueError("Question must follow the single contiguous image block")
            if token_length_list and int(token_length_list[0]) != hidden.shape[1]:
                raise ValueError("token_length_list must describe the expanded prefill")
            pos = torch.arange(hidden.shape[1], device=hidden.device) if position_ids is None else position_ids[0].to(hidden.device)
            if pos.numel() != hidden.shape[1] or not bool((pos[1:] > pos[:-1]).all()):
                raise ValueError("Require strictly increasing original RoPE positions")
            ids = metadata["patch_ids"].to(hidden.device)
            self.state = State(v0, ids, pos[v0:v0+n].clone(), pos,
                               torch.arange(a, b, device=hidden.device), metadata, n,
                               int(pos[-1].item()) + 1,
                               self.cfg.fingerprint(n, len(m.layers), metadata["stage1"]))
            hidden, cache, history = self._tail(hidden, cache, output_hidden_states=output_hidden_states)
            # Compatibility counters for the existing lmms-eval audit path.
            m.e001_last_stage1_visual_tokens = int(n)
            m.e001_last_stage2_visual_tokens = int(self.state.visual_ids.numel())
            m.e001_last_final_visual_tokens = int(self.state.visual_ids.numel())
            first_layer = int(self.cfg.events[0][0]) if self.cfg.events else -1
            m.e006_last = {
                "visual_in": int(n),
                "prefill_count": 1,
                "decode_count": 0,
                "e010_qed_layer": first_layer if first_layer >= 0 else int(os.environ.get("E010_QED_LAYER", "8")),
                "span_source": "explicit",
                "sdpa_preserved": True,
                "e018_mode": self.cfg.mode,
                "e018_events": list(self.cfg.events),
                "e018_final_k": int(self.state.visual_ids.numel()),
                "e018_token_work": int(sum(self.state.per_layer_n)),
            }
        else:
            s = self.state
            if s is None or hidden.shape[1] != 1:
                raise ValueError("Decode requires the matching request state and exactly one new token")
            history = () if output_hidden_states else None
            pos = torch.tensor([[s.original_next]], device=hidden.device, dtype=torch.long)
            if position_ids is not None and not torch.equal(position_ids.to(pos.device), pos):
                raise ValueError("Decode RoPE must use original position, not compacted cache length")
            for i, layer in enumerate(m.layers):
                if output_hidden_states:
                    history += (hidden,)
                layer.self_attn._e018_callback = None
                output = layer(hidden, attention_mask=None, position_ids=pos,
                               past_key_value=cache, use_cache=True, output_attentions=False)
                hidden = output[0]
                s.cache_positions[i] = torch.cat((s.cache_positions[i], pos[0]))
            s.original_next += 1
            hidden = m.norm(hidden)
            if output_hidden_states:
                history += (hidden,)
        result_cache = cache.to_legacy_cache() if legacy else cache
        if not return_dict:
            return tuple(x for x in (hidden, result_cache, history) if x is not None)
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=result_cache,
                                       hidden_states=history, attentions=None)


def e018_forward(model, **kwargs):
    if not hasattr(model, "_e018_runtime"):
        model._e018_runtime = Runtime(model)
    return model._e018_runtime.forward(**kwargs)


def e018_generation_inputs(model, input_ids, past_key_values, attention_mask, inputs_embeds, kwargs):
    """Generation time uses request state, never the scrubbed layer-0 cache length."""
    has_past = past_key_values is not None and (
        past_key_values.get_seq_length() > 0 if isinstance(past_key_values, Cache) else
        bool(past_key_values) and past_key_values[0][0].shape[2] > 0)
    if has_past:
        runtime = model.model._e018_runtime
        pos = torch.tensor([[runtime.state.original_next]], device=input_ids.device)
        return dict(input_ids=input_ids[:, -1:], position_ids=pos, past_key_values=past_key_values,
                    use_cache=True, attention_mask=None)
    return dict(inputs_embeds=inputs_embeds, position_ids=kwargs.get("position_ids"),
                past_key_values=None, use_cache=True, attention_mask=attention_mask) if inputs_embeds is not None else dict(
                input_ids=input_ids, position_ids=kwargs.get("position_ids"), past_key_values=None,
                use_cache=True, attention_mask=attention_mask)
