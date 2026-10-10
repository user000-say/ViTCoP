#!/usr/bin/env python3
"""E014 GPU smoke: one arm, real panel samples, Stage1/KV invariants."""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
ROOT = REPO / "logs/e014_aee"
sys.path.insert(0, str(REPO))

import e010_common
from e010_common import sha256
from e014_arms import ARMS, apply_arm_env

e010_common.ROOT = ROOT
e010_common.RUNTIME = ROOT / "code_runtime"
e010_common.bind()

for rel in (
    "LLaVA/llava/model/multimodal_encoder/e013_stage1_support.py",
    "LLaVA/llava/model/multimodal_encoder/e014_config.py",
    "LLaVA/llava/model/multimodal_encoder/e014_stage1_support.py",
    "LLaVA/llava/model/multimodal_encoder/clip_encoder.py",
    "LLaVA/llava/model/language_model/vitcop_modeling_llama.py",
):
    dst = e010_common.RUNTIME / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or sha256(dst) != sha256(REPO / rel):
        shutil.copy2(REPO / rel, dst)

from e009_common import apply_profile, load_model, panel_lookup
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
    meta = apply_arm_env(arm)
    set_e010(enabled=True, layer=meta["s"])
    meta = apply_arm_env(arm)
    os.environ["E012_KV_AUDIT"] = "True" if audit else "False"
    os.environ["E014_AUDIT"] = "True" if audit else "False"
    import e006_span

    e006_span.STAGE1_VISUAL = meta["visual"]
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=tuple(ARMS))
    ap.add_argument("--gpu", type=int, default=4)
    args = ap.parse_args()
    meta = force(args.arm, audit=True)
    out = ROOT / "smoke_gpu" / args.arm
    out.mkdir(parents=True, exist_ok=True)
    bucket: list[str] = []
    sys.stdout = _Tee(sys.__stdout__, bucket)
    report = {
        "arm": args.arm,
        "meta": meta,
        "ok": False,
        "errors": [],
        "samples": [],
    }
    try:
        uuid, eng, lm, raw, core, _torch = load_model(args.gpu)
        import e006_lmms_patch as patch
        from e006_lmms_patch import install

        patch._audit = lambda raw_model: None
        install()
        lookup, keys = panel_lookup(eng, lm)
        original = raw.generate
        env_k = int(os.environ["E006_TARGET_K"])

        def wrapped(*ga, **gk):
            kw = dict(gk)
            kw.update(
                do_sample=False, num_beams=1, use_cache=True,
                min_new_tokens=3, max_new_tokens=3, temperature=0.0,
            )
            return original(*ga, **kw)

        raw.generate = wrapped
        for key in keys[:2]:
            pred = lm.generate_until([lookup[key]])
            last = getattr(core, "e006_last", {}) or {}
            vt = raw.get_model().get_vision_tower()
            e014_last = getattr(vt, "e014_last", None)
            stage1 = int(getattr(core, "e001_last_stage1_visual_tokens", -1) or -1)
            final_v = int(getattr(core, "e001_last_final_visual_tokens", -1) or -1)
            visual_in = int(last.get("visual_in") or -1)
            visual_out = int(last.get("visual_out") or -1)
            sample = {
                "key": key,
                "pred": pred,
                "stage1_visual": stage1,
                "final_visual": final_v,
                "visual_in": visual_in,
                "visual_out": visual_out,
                "prefill_count": int(last.get("prefill_count", -1)),
                "decode_count": int(last.get("decode_count", -1)),
                "e014_budget": None,
            }
            if stage1 != 172:
                report["errors"].append(f"{key}: stage1={stage1} !=172")
            if visual_in not in (-1, 172) and visual_in != 172:
                report["errors"].append(f"{key}: visual_in={visual_in}")
            if final_v not in (-1, env_k) and final_v != env_k:
                report["errors"].append(f"{key}: final_visual={final_v} != {env_k}")
            if visual_out not in (-1, env_k) and visual_out != env_k:
                report["errors"].append(f"{key}: visual_out={visual_out} != {env_k}")
            if isinstance(e014_last, dict):
                selected = e014_last.get("selected_ids")
                added = e014_last.get("added_ids")
                removed = e014_last.get("removed_ids")
                stats = e014_last.get("stats") or {}
                b_raw = stats.get("budget", 0)
                b = int(b_raw.item()) if hasattr(b_raw, "item") else int(b_raw)
                sample["e014_budget"] = b
                if selected is not None:
                    sel = selected.detach().cpu().tolist()
                    sample["selected_n"] = len(sel)
                    if len(sel) != 172:
                        report["errors"].append(f"{key}: selected_len={len(sel)}")
                    if len(set(sel)) != len(sel):
                        report["errors"].append(f"{key}: duplicate selected ids")
                    if any(i < 0 or i > 575 for i in sel):
                        report["errors"].append(f"{key}: selected id out of range")
                if meta["variant"] not in ("AEE", "ADAPTIVE_B_ONLY"):
                    if added is not None and int(added.numel()) != b:
                        report["errors"].append(f"{key}: added={int(added.numel())} !=b={b}")
                    if removed is not None and int(removed.numel()) != b:
                        report["errors"].append(f"{key}: removed={int(removed.numel())} !=b={b}")
                elif b not in (0, 8, 16, 24, 32):
                    report["errors"].append(f"{key}: adaptive b={b} not in budget set")
            else:
                report["errors"].append(f"{key}: missing e014_last")
            report["samples"].append(sample)

        text = "".join(bucket)
        kv = []
        for line in text.splitlines():
            if "[E012_KV_JSON] " in line:
                kv.append(json.loads(line.split("[E012_KV_JSON] ", 1)[1]))
        phases = {row["phase"] for row in kv}
        if not {"prefill", "decode1", "decode2"} <= phases:
            report["errors"].append(f"missing KV phases {sorted(phases)}")
        if any(row["cache_tokens"] != row["expected_tokens"] for row in kv):
            report["errors"].append("KV cache_tokens mismatch")
        if kv:
            pre = {
                r["layer"]: r["cache_tokens"]
                for r in kv if r["request_index"] == kv[0]["request_index"] and r["phase"] == "prefill"
            }
            if set(pre) == set(range(32)):
                removed = 172 - env_k
                if pre[0] - pre[31] != removed:
                    report["errors"].append(
                        f"prefill shrink {pre[0]-pre[31]} expected {removed}"
                    )
            else:
                report["errors"].append("incomplete prefill layer KV audit")
        report["ok"] = not report["errors"]
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc()
    finally:
        sys.stdout = sys.__stdout__
        (out / "smoke_log.txt").write_text("".join(bucket), encoding="utf-8")
        (out / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"arm": args.arm, "ok": report["ok"], "errors": report["errors"]}, indent=2))
        if not report["ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
