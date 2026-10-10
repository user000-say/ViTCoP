#!/usr/bin/env python3
"""E013 GPU smoke: one arm, GPU4, two real panel samples, >=3 new tokens, KV audit."""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import traceback
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
ROOT = REPO / "logs/e013_p1"
sys.path.insert(0, str(REPO))

import shutil

import e010_common
from e010_common import sha256
from e013_arms import ARMS, apply_arm_env, visual_count

e010_common.ROOT = ROOT
e010_common.RUNTIME = ROOT / "code_runtime"
e010_common.bind()
_support = "LLaVA/llava/model/multimodal_encoder/e013_stage1_support.py"
_dst = e010_common.RUNTIME / _support
_dst.parent.mkdir(parents=True, exist_ok=True)
if not _dst.exists() or sha256(_dst) != sha256(REPO / _support):
    shutil.copy2(REPO / _support, _dst)

from e009_common import apply_profile, hard, load_model, panel_lookup
from e010_common import set_e010


class _Tee(io.TextIOBase):
    def __init__(self, sink, bucket):
        self.sink = sink
        self.bucket = bucket

    def write(self, text):
        self.sink.write(text)
        self.bucket.append(text)
        return len(text)

    def flush(self):
        self.sink.flush()


def force(arm: str, *, audit: bool) -> dict:
    meta = apply_arm_env(arm)
    apply_profile("ANCHOR_A", audit=False, strict=True)
    enable, variant, pick, b, s, d = ARMS[arm]
    set_e010(enabled=True, layer=s)
    # apply_profile resets selectors; restore E013/E012 freeze.
    meta = apply_arm_env(arm)
    os.environ["E012_KV_AUDIT"] = "True" if audit else "False"
    os.environ["E013_AUDIT"] = "True" if audit else "False"
    os.environ["E013_STRICT"] = "True" if audit else "False"
    import e006_span

    e006_span.STAGE1_VISUAL = meta["visual"]
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=tuple(ARMS))
    ap.add_argument("--gpu", type=int, default=4)
    args = ap.parse_args()
    meta = force(args.arm, audit=True)
    out = ROOT / "smoke" / args.arm
    out.mkdir(parents=True, exist_ok=True)
    bucket: list[str] = []
    sys.stdout = _Tee(sys.__stdout__, bucket)
    uuid, eng, lm, raw, core, _torch = load_model(args.gpu)
    import e006_lmms_patch as patch
    from e006_lmms_patch import install

    patch._audit = lambda raw_model: None
    install()
    lookup, keys = panel_lookup(eng, lm)
    original = raw.generate

    def wrapped(*ga, **gk):
        kw = dict(gk)
        kw.update(
            do_sample=False, num_beams=1, use_cache=True,
            min_new_tokens=3, max_new_tokens=3, temperature=0.0,
        )
        answer = original(*ga, **kw)
        wrapped.seq_len = int(answer.shape[-1]) if hasattr(answer, "shape") else -1
        wrapped.ids = answer[0].detach().cpu().tolist() if hasattr(answer, "detach") else None
        return answer

    raw.generate = wrapped
    samples = []
    for key in keys[:2]:
        pred = lm.generate_until([lookup[key]])
        last = getattr(core, "e006_last", {}) or {}
        selected = last.get("selected_local")
        if hasattr(selected, "detach"):
            selected = selected.detach().cpu().tolist()
        stage1 = getattr(getattr(raw.get_model(), "get_vision_tower", lambda: None)(), "e013_last", None)
        # vision tower is on the model; try both paths
        vt = raw.get_model().get_vision_tower()
        e013_last = getattr(vt, "e013_last", None) or {}
        if hasattr(e013_last.get("persistent_ids", None), "detach"):
            persist = e013_last["persistent_ids"].detach().cpu().tolist()
            transient = e013_last.get("transient_ids")
            transient = [] if transient is None else (
                transient.detach().cpu().tolist() if hasattr(transient, "detach") else list(transient)
            )
            original_core = e013_last["original_core_ids"].detach().cpu().tolist()
        else:
            persist = transient = original_core = None
        generated = wrapped.seq_len - 1 if wrapped.seq_len == 4 else wrapped.seq_len
        row = {
            "dataset": key[0],
            "sample_id": key[1],
            "sample_index": key[2],
            "text": "" if not pred else str(pred[0]),
            "generated_tokens": generated,
            "token_ids": wrapped.ids,
            "selected_local": selected,
            "visual_in": int(last.get("visual_in") or -1),
            "visual_out": int(last.get("visual_out") or -1),
            "prefill_count": int(last.get("prefill_count", -1)),
            "decode_count": int(last.get("decode_count", -1)),
            "qed_layer": int(last.get("e010_qed_layer", -1)),
            "final_visual": int(getattr(core, "e001_last_final_visual_tokens", -1)),
            "stage1_visual": int(getattr(core, "e001_last_stage1_visual_tokens", -1)),
            "persistent_ids": persist,
            "transient_ids": transient,
            "original_core_ids": original_core,
        }
        samples.append(row)
        expect_vis = visual_count(args.arm)
        if generated != 3 or row["prefill_count"] != 1 or row["decode_count"] != 0:
            hard(f"{args.arm} selector/token gate {row}")
        if row["stage1_visual"] != expect_vis or row["final_visual"] != 96:
            hard(f"{args.arm} visual gate expect_stage1={expect_vis} {row}")
        if row["visual_in"] != expect_vis:
            hard(f"{args.arm} visual_in gate {row}")
        if not row["text"].strip():
            hard(f"{args.arm} empty text")
        if meta["enable"] and meta["variant"] == "EXTRA" and meta["b"] > 0:
            if selected is None or any(int(i) >= 172 for i in selected):
                hard(f"{args.arm} transient leaked into final K96: {selected}")
            if persist is not None and original_core is not None and persist != original_core:
                hard(f"{args.arm} EXTRA must keep original Core172 IDs")
        if meta["enable"] and meta["variant"] == "REPLACE" and meta["b"] > 0:
            if persist is None or len(persist) != 172:
                hard(f"{args.arm} REPLACE persist {persist}")
            if original_core is not None:
                kept = set(persist) & set(original_core)
                if len(kept) != 156:
                    hard(f"{args.arm} REPLACE kept {len(kept)} expected 156")

    text = "".join(bucket)
    kv = []
    for line in text.splitlines():
        if "[E012_KV_JSON] " in line:
            kv.append(json.loads(line.split("[E012_KV_JSON] ", 1)[1]))
    phases = {row["phase"] for row in kv}
    if not {"prefill", "decode1", "decode2"} <= phases:
        hard(f"{args.arm} missing KV phases {sorted(phases)}")
    if any(row["cache_tokens"] != row["expected_tokens"] for row in kv):
        hard(f"{args.arm} KV mismatch")
    # topology: layers <=D full, later compact by (172+b-96)
    delete = meta["D"]
    req = kv[0]["request_index"]
    pre = {r["layer"]: r["cache_tokens"] for r in kv if r["request_index"] == req and r["phase"] == "prefill"}
    if set(pre) != set(range(32)):
        hard(f"{args.arm} incomplete prefill layers")
    full, short = pre[0], pre[31]
    removed = visual_count(args.arm) - 96
    if full - short != removed:
        hard(f"{args.arm} removed {full-short} expected {removed}")
    if any(pre[i] != full for i in range(delete + 1)):
        hard(f"{args.arm} layers 0-{delete} not full")
    if any(pre[i] != short for i in range(delete + 1, 32)):
        hard(f"{args.arm} layers >{delete} not compact")

    record = {
        "arm": args.arm,
        "meta": meta,
        "gpu": args.gpu,
        "gpu_uuid": uuid,
        "status": "PASS",
        "samples": samples,
        "kv_rows": len(kv),
        "code_sha": sha256(REPO / "LLaVA/llava/model/language_model/vitcop_modeling_llama.py"),
        "clip_sha": sha256(REPO / "LLaVA/llava/model/multimodal_encoder/clip_encoder.py"),
    }
    (out / "smoke.json").write_text(json.dumps(record))
    print(f"[e013 smoke] {out / 'smoke.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        traceback.print_exc()
        hard(f"{type(exc).__name__}: {exc}")
