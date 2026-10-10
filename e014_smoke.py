#!/usr/bin/env python3
"""E014 GPU smoke: one arm, real panel samples, KV/E014 audit invariants."""
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
            vt = getattr(raw.get_model(), "get_vision_tower", lambda: None)()
            e014_last = getattr(vt, "e014_last", None) if vt is not None else None
            stage1 = int(getattr(core, "e001_last_stage1_visual_tokens", -1) or -1)
            sample = {
                "key": key,
                "pred": pred,
                "stage1_visual": stage1,
                "target_k": last.get("target_k"),
                "selected_k": last.get("selected_k") or last.get("qed_selected_k"),
                "e014_last_keys": sorted(list(e014_last.keys())) if isinstance(e014_last, dict) else None,
                "e014_budget": (e014_last or {}).get("budget") if isinstance(e014_last, dict) else None,
                "e014_variant": (e014_last or {}).get("variant") if isinstance(e014_last, dict) else None,
            }
            if stage1 != 172:
                report["errors"].append(f"{key}: stage1={stage1} !=172")
            if int(last.get("target_k") or -1) != int(meta["target_k"]):
                report["errors"].append(
                    f"{key}: target_k={last.get('target_k')} != {meta['target_k']}"
                )
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
            report["samples"].append(sample)
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
    hard()
    main()
