"""Random causal decoder fixture. No pretrained VLM, no benchmark claim.

Runs prompt prefill and multi-token cached teacher forcing. Visual IDs have
no relation to positions; earlier caches survive normal hidden-row pruning.
"""
from copy import deepcopy
import hashlib
import numpy as np
from .evidence import compact_sequence


class ToyAdapter:
    is_fixture = True
    def __init__(self, depth=8, visual=12, text=3, dim=12, vocab=9, seed=7):
        self.depth, self.vocab = depth, vocab
        r = np.random.default_rng(seed)
        self.weights = r.normal(0, .15, (depth, 3, dim, dim))
        self.output = r.normal(0, .3, (dim, vocab))
        self.embeddings = r.normal(0, .4, (vocab, dim))
        h = r.normal(0, .5, (visual+text, dim))
        ids = np.r_[np.arange(visual)*3+11, [-1]*text]
        positions = np.arange(visual+text)*2
        fingerprint = hashlib.sha256(h.tobytes()+ids.tobytes()).hexdigest()
        self.initial = dict(hidden=h, positions=positions, ids=ids, completed=0,
                            kv=[], trace=[], prefix_fingerprint=fingerprint)
        self.targets = (2, 5, 1)

    def capture(self, completed=2):
        state = self.fork(self.initial)
        self.advance(state, completed)
        state["prefix_fingerprint"] = hashlib.sha256(
            state["hidden"].tobytes()+str(completed).encode()).hexdigest()
        return state

    def inspect(self, state):
        return dict(completed=state["completed"],
                    active_ids=tuple(map(int, state["ids"][state["ids"] >= 0])),
                    protected_positions=tuple(map(int, state["positions"][state["ids"] < 0])),
                    visual_positions={int(i): int(p) for i, p in zip(state["ids"], state["positions"]) if i >= 0},
                    kv_support=tuple(tuple(map(int, c["ids"][c["ids"] >= 0])) for c in state["kv"]),
                    prefix_fingerprint=state["prefix_fingerprint"],
                    state_hash=hashlib.sha256(state["hidden"].tobytes()+b"".join(
                        c["k"].tobytes()+c["v"].tobytes() for c in state["kv"])).hexdigest())

    def fork(self, state):
        return deepcopy(state)

    def prune(self, state, keep):
        h, p, v, _ = compact_sequence(state["hidden"], state["positions"], state["ids"], keep)
        state.update(hidden=h, positions=p, ids=v)

    @staticmethod
    def attention(q, k, v, allowed):
        z = q@k.T/np.sqrt(q.shape[-1])
        z = np.where(allowed, z, -np.inf)
        p = np.exp(z-z.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        return p@v

    def advance(self, state, completed):
        if not state["completed"] <= completed <= self.depth:
            raise ValueError("invalid resume depth")
        for layer in range(state["completed"], completed):
            h, p, ids = state["hidden"], state["positions"], state["ids"]
            q, k, v = (h@w for w in self.weights[layer])
            state["kv"].append(dict(k=k.copy(), v=v.copy(), ids=ids.copy(), positions=p.copy()))
            out = self.attention(q, k, v, p[:, None] >= p[None, :])
            state["hidden"] = h+np.tanh(out)
            state["trace"].append(int((ids >= 0).sum()))
            state["completed"] = layer+1

    def align_cache(self, state, keep):
        for cache in state["kv"]:
            rows = (cache["ids"] < 0) | np.isin(cache["ids"], keep)
            for key in ("k", "v", "ids", "positions"):
                cache[key] = cache[key][rows].copy()

    def _decode(self, state, teacher):
        h = state["hidden"][-1:]
        position = int(state["positions"].max())+1
        losses, generated = [], []
        for i, target in enumerate(self.targets):
            logits = (h@self.output)[0]
            generated.append(int(np.argmax(logits)))
            z = logits-logits.max()
            losses.append(float(np.log(np.exp(z).sum())-z[target]))
            if i == len(self.targets)-1:
                break
            previous = target if teacher else generated[-1]
            h = self.embeddings[previous:previous+1].copy()
            for layer, cache in enumerate(state["kv"]):
                q, k, v = (h@w for w in self.weights[layer])
                cache["k"] = np.concatenate((cache["k"], k))
                cache["v"] = np.concatenate((cache["v"], v))
                cache["ids"] = np.r_[cache["ids"], -1]
                cache["positions"] = np.r_[cache["positions"], position]
                out = self.attention(q, cache["k"], cache["v"],
                                     (cache["positions"] <= position)[None])
                h = h+np.tanh(out)
            position += 1
        return float(np.mean(losses)), generated

    def score_suffix(self, state):
        self.advance(state, self.depth)
        nll, _ = self._decode(self.fork(state), teacher=True)
        _, generated = self._decode(self.fork(state), teacher=False)
        return dict(nll=nll, answer_tokens=len(self.targets),
                    trace=tuple(state["trace"]), backend="numpy_causal_fixture",
                    generated=generated)
