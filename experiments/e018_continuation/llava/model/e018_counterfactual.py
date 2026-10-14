"""Offline common-prefix continuation interventions. Ground truth stays here."""
import copy
from dataclasses import replace
import torch
from .e018_stage2_support import clone_cache, Runtime


class PrefixCaptured(Exception):
    def __init__(self, snapshot):
        super().__init__("offline prefix captured")
        self.snapshot = snapshot


def capture_at(runtime, prepared, event_layer):
    """Abort at POST-block t, before either candidate treatment. No duplicated prefix."""
    def observer(ctx):
        if ctx["layer"] == event_layer:
            frozen = {k: v for k, v in ctx.items() if k not in {"hidden", "cache", "state"}}
            frozen["hidden"] = ctx["hidden"].detach().clone()
            frozen["cache"] = clone_cache(ctx["cache"])
            frozen["state"] = copy.deepcopy(ctx["state"])
            raise PrefixCaptured(frozen)
    old = runtime.observer
    runtime.observer = observer
    try:
        runtime.forward(**prepared)
    except PrefixCaptured as e:
        return e.snapshot
    finally:
        runtime.observer = old
    raise ValueError("Requested event was not reached")


@torch.no_grad()
def _run_branch(core, head, config, snap, answer_ids, delayed, future_keeps=None):
    r = Runtime(core, replace(config, mode="STATIC", controller="", trace=""))
    r.state = copy.deepcopy(snap["state"])
    if future_keeps:
        r.state.forced_keeps.update({int(k): v.clone() for k, v in future_keeps.items()})
    h, c = snap["hidden"].clone(), clone_cache(snap["cache"])
    layer, end = snap["layer"], snap["endpoint"]
    # Same D and same survivors for both branches. Only retirement time differs.
    r.state.events.append(dict(layer=layer, target=snap["keep"].numel(), endpoint=end,
                               action="continue" if delayed else "retire",
                               delta_work=snap["delta_work"],
                               keep_ids=r.state.visual_ids[snap["keep"]].clone(),
                               drop_ids=snap["drop_ids"].clone(), predicted=None, upper=None))
    receiver = h[0].index_select(0, r.state.question_indices).float().mean(0)
    # History is captured before deletion; answer labels never affect future selection.
    r.state.receiver = receiver
    if "side_r" in snap:
        r.state.history = dict(ids=r.state.visual_ids.clone(), r=snap["side_r"].clone())
    if delayed:
        r.state.pending.append((end, snap["drop_ids"].clone()))
    else:
        h = r._drop_ids(h, c, snap["drop_ids"])
        r._record_physical(layer)
    h, c, _ = r._tail(h, c, start=layer + 1)
    trace = r.last_trace or r.trace(c)
    first_ids = r.state.visual_ids.clone()
    future = {x["layer"]: x["keep_ids"].clone() for x in r.state.events if x["layer"] > layer}
    logits = head(h[:, -1, :]).float()[0]
    losses = [F_nll(logits, int(answer_ids[0]))]
    first_correct = bool(logits.argmax().item() == int(answer_ids[0]))
    for i in range(1, len(answer_ids)):
        out = r.forward(input_ids=torch.tensor([[int(answer_ids[i-1])]], device=h.device),
                        past_key_values=c, use_cache=True, return_dict=True)
        c = out.past_key_values
        logits = head(out.last_hidden_state[:, -1, :]).float()[0]
        losses.append(F_nll(logits, int(answer_ids[i])))
    return dict(nll=float(torch.stack(losses).mean().item()), token_nll=[float(x.item()) for x in losses],
                first_token_correct=first_correct, trace=trace, final_ids=first_ids,
                future_keeps=future, cache_positions=copy.deepcopy(r.state.cache_positions))


def F_nll(logits, token):
    return -torch.log_softmax(logits, dim=-1)[token]


@torch.no_grad()
def continuation_pair(core, head, config, snapshot, answer_ids):
    if not answer_ids:
        raise ValueError("Nonempty tokenized reference answer required")
    now = _run_branch(core, head, config, snapshot, answer_ids, False)
    later = _run_branch(core, head, config, snapshot, answer_ids, True, now["future_keeps"])
    if not torch.equal(now["final_ids"], later["final_ids"]):
        raise RuntimeError("Counterfactual final survivor IDs do not match")
    if config.cache_policy == "scrub_all":
        for i in now["cache_positions"]:
            if not torch.equal(now["cache_positions"][i], later["cache_positions"][i]):
                raise RuntimeError("scrub_all must leave identical cache key positions in both branches")
    delta_w = later["trace"]["token_work"] - now["trace"]["token_work"]
    expected = snapshot["drop_ids"].numel() * (snapshot["endpoint"] - snapshot["layer"])
    if delta_w != expected:
        raise RuntimeError("Extra continuation work differs from declared treatment")
    return dict(utility=now["nll"] - later["nll"], now_nll=now["nll"], later_nll=later["nll"],
                delta_work=snapshot["delta_work"], delta_token_work=delta_w,
                now_work=now["trace"]["token_work"], later_work=later["trace"]["token_work"],
                now_first_correct=now["first_token_correct"], later_first_correct=later["first_token_correct"],
                final_ids=now["final_ids"].cpu().tolist(),
                now_cache_lengths=now["trace"]["cache_lengths"],
                later_cache_lengths=later["trace"]["cache_lengths"])
