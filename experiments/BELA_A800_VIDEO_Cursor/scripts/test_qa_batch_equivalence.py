#!/usr/bin/env python3
"""P1 gate: batched QA keep IDs must match serial qa_core on random panels."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bela_video_a800.qa_core import AnchorConfig, select_question_anchored
from bela_video_a800.qa_batched import select_question_anchored_batched
from bela_video_a800.video_entry import select_framewise, query_proxy


def _serial(native, pooled, q_scores, max_swaps=8):
    f, k = native.shape
    n = pooled.shape[1]
    outs, swaps = [], []
    for i in range(f):
        r = select_question_anchored(
            native_keep_ids=native[i],
            visual_metric=(pooled[i], pooled[i]),
            raw_query_scores=q_scores[i],
            config=AnchorConfig(max_swaps=max_swaps),
        )
        outs.append(r.keep_ids.cpu())
        swaps.append(r.swaps)
    return torch.stack(outs), tuple(swaps)


def case(f, n, k, d, seed, device="cpu", max_swaps=8):
    g = torch.Generator().manual_seed(seed)
    pooled = torch.randn(f, n, d, generator=g)
    # native = topk by L2
    score = pooled.norm(dim=-1)
    native = score.topk(k, dim=-1).indices.sort(dim=-1).values
    q_emb = torch.randn(12, d, generator=g)
    q_scores = query_proxy(pooled, q_emb)
    pooled = pooled.to(device)
    native = native.to(device)
    q_scores = q_scores.to(device)

    serial_ids, serial_swaps = _serial(native.cpu(), pooled.cpu(), q_scores.cpu(), max_swaps)
    batch = select_question_anchored_batched(
        native_keep_ids=native,
        visual_metric=(pooled, pooled),
        raw_query_scores=q_scores,
        config=AnchorConfig(max_swaps=max_swaps),
    )
    bid = batch.keep_ids.cpu()
    if not torch.equal(serial_ids, bid):
        diff = (serial_ids != bid).nonzero()
        raise AssertionError(f"ID mismatch seed={seed} first_diff={diff[:5].tolist()} swaps_s={serial_swaps} swaps_b={batch.swap_count}")
    if serial_swaps != batch.swap_count:
        raise AssertionError(f"swap_count mismatch {serial_swaps} vs {batch.swap_count}")

    # Also through select_framewise with env batch on/off
    os.environ["BELA_VIDEO_QA_BATCH"] = "0"
    a = select_framewise(
        tier="T32" if k == 64 else "T16",
        native_ids=native.cpu() if k in (64, 32) else native.cpu(),
        dual_views=(pooled.cpu(), pooled.cpu()),
        raw_query_scores=q_scores.cpu(),
        control="qa",
    )
    # For non T32/T16 K, skip framewise wrapper
    return {"seed": seed, "f": f, "swaps": list(serial_swaps), "ok": True}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {"device": device, "cases": []}
    # Small cases always
    for args in (
        (1, 16, 8, 32, 1),
        (3, 32, 16, 64, 2),
        (8, 64, 32, 128, 3),
        (16, 169, 64, 256, 4),
        (64, 169, 64, 128, 5),  # realistic
        (64, 169, 64, 256, 6),
        (64, 169, 64, 7, 7),  # flat-ish via seed
    ):
        f, n, k, d, seed = args
        # T16/T32 wrapper only when K matches schedule; unit test uses direct batched API
        serial_ids_swaps_ok = True
        g = torch.Generator(device="cpu").manual_seed(seed)
        pooled = torch.randn(f, n, d, generator=g).to(device)
        native = pooled.norm(dim=-1).topk(k, dim=-1).indices.sort(-1).values
        q_scores = query_proxy(pooled, torch.randn(20, d, generator=g).to(device))
        s_ids, s_sw = _serial(native.cpu(), pooled.cpu(), q_scores.cpu(), 8 if k >= 64 else 4)
        b = select_question_anchored_batched(
            native_keep_ids=native,
            visual_metric=(pooled, pooled),
            raw_query_scores=q_scores,
            config=AnchorConfig(max_swaps=8 if k >= 64 else 4),
        )
        if not torch.equal(s_ids, b.keep_ids.cpu()):
            raise SystemExit(f"FAIL ids f={f} seed={seed}")
        if s_sw != b.swap_count:
            raise SystemExit(f"FAIL swaps f={f} seed={seed} {s_sw} vs {b.swap_count}")
        report["cases"].append({"f": f, "n": n, "k": k, "seed": seed, "swaps_sum": int(sum(s_sw)), "ok": True})

    # coverage delta reference self-check if present
    ref = ROOT / "accelerate_kit_20261031" / "BELA_VIDEO_BATCHED_COVERAGE_DELTA_REFERENCE.py"
    if ref.is_file():
        import runpy
        # run self_test via import
        import importlib.util
        spec = importlib.util.spec_from_file_location("covref", ref)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.self_test()
        report["coverage_ref"] = "PASS"

    out = Path(os.environ.get("BELA_EQ_OUT", "/tmp/VIDEO_QA_BATCH_EQUIVALENCE.json"))
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("PASS equivalence", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
