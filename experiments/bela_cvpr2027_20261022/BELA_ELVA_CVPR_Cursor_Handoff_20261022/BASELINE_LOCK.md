# BASELINE_LOCK — 2026-10-22 CVPR2027 handoff preflight

**Git HEAD:** `14779c69055148434e9b1dac5c1eea1af4d659d2`  
**Verdict:** package is **NOT server-integrated**. Offline CPU tests PASS. Production dual-COVER + RDRIFT remain host-owned. Dynamic Stage1/Stage2 (`BELA_NEXT_*`) **NOT wired**. ACC scope this round (user override): **6 tasks** only.

## Production import paths (SHA256)

| Path | SHA256 |
|---|---|
| `LLaVA/llava/model/multimodal_encoder/clip_encoder.py` | `4d4b074fa11a778b956e3e4dd5fa937a14c8091188ff4b76a8b63ac1c64c15a7` |
| `LLaVA/llava/model/language_model/vitcop_modeling_llama.py` | `0d2546a852819ae6badcea1d6efcbfb99806c08204a9c6a42442b494d7824e8e` |
| `LLaVA/llava/model/e018_stage2_support.py` | `2389fc8b2f671623c09d895bbc8877326e2800db539c06d237c2676975b3e774` |
| `LLaVA/llava/model/e018_contract.py` | `f76ac9a378fefb49ad61a002ad4e252a2a6c905759753ce3c09790f76495158a` |
| `experiments/elva_r3/apply_llava.py` | `5cd3c004fa10a89cd74c33b59081e7bc6423789d50ad0e43400926a1f7b27976` |
| `e014_boot.py` | `4c12f11dce92657bb20ab752bc9d1d7ec58e8f11ebf3c24e6f81b5c5c45f0421` |
| `elva_scores.py` / `elva_acc_tasks.py` | frozen scorer + task ids |

## Frozen protocol

- Checkpoint: LLaVA-1.5-7B via existing e014/e010 eval path (`ANCHOR_A`).
- Stage1: dual Layer9+penultimate **legacy_native COVER** (`ELVA_R3_ENTRY_K`).
- Stage2: E018 STATIC progressive prune; RDRIFT = E020 `rank_drift` on first event only when configured.
- Anchor schedule W2048: counts `172/90/32`, lengths `4/8/20` → events `3:90,11:32`.
- Scorer: `elva_scores.py` Dense refs unchanged. TextVQA arm = `textvqa_val_ocr`.
- GPUs: physical **0, 2, 4, 7**.

## ACC task override (this round)

`gqa`, `pope`, `mme`, `mmbench_en_dev`, `scienceqa_img`, `textvqa_val_ocr`  
(NOT full R9; mmmu/ai2d/vizwiz deferred.)

## Algorithm identity

| Arm | Operator | Source |
|---|---|---|
| F0 `W2048_COVER_QATTN` | dual COVER K172 + Q_ATTN | host R3/Gate1 (reuse metrics) |
| F1 `W2048_COVER_RDRIFT` | dual COVER K172 + RDRIFT first | host CVPR H2/Gate1 (reuse metrics) |
| F2 `ENTRY160_RDRIFT` | COVER K160 + RDRIFT, `160/96/32` @4/8/20 | **new config**; original COVER/RDRIFT via host |
| F3 `ENTRY128_RDRIFT` | COVER K128 + RDRIFT, `128/64/32` @8/8/16 | **new config**; original COVER/RDRIFT via host |
| A0–A4 adaptive | bela_next policy | **NOT_RUN** — no production bind |

## Host patch for G1 entry K (OFF identity for K172)

- `e013_choose(..., core_k=None)` — default `172` unchanged; allows `{128,160,172}` with same COVER gain / REPLACE slots=32.
- `elva_r3/native_cover.py` + `cover_binding.py` + `llava_stage1.py` declare supported_k `{128,160,172}`.
- Smoke F2 pope limit=4: `stage1_tokens=160`, `final_visual_tokens=32`, RC=0.

## Gaps (blocked for adaptive)

- No read-only `greedy_order` / residual-coverage GPU trace yet (`SMOKE_AUDIT.jsonl` NOT_RUN).
- `BELA_NEXT_*` dynamic policy **not** patched into clip_encoder / E018.
- Shuffled budget control requires image-id budgets from live COVER residuals — NOT_RUN.
- A0–A4 / G2–G3 / TTFT Fixed32 this round — NOT_RUN (user scoped ACC to 6 tasks only).

## Offline package

`pytest` 25 passed / 2 skipped; `plan_offline` / `make_matched_controls` / `analyze_pope` / `audit_trace` OK on synthetic examples only.
