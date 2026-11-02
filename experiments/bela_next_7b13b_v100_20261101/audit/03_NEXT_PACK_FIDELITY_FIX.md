# NeXT engineering fidelity fix (2026-11-02)

## Distortion removed

Previous ENABLE=1 Stage1 used `simple_crop_concat`: selected per-crop patches
were concatenated with **no** `spatial_unpad` and **no** `image_newline`.
That is not Llava-NeXT’s visual sequence.

## Fix

1. `bela_next_forward/next_pack.py` — mirror HF 5.20 `pack_image_features` /
   `unpad_image`; always emit stock newlines; only drop unselected patches.
2. Dual COVER selects inside post-unpad valid pools per tile crop.
3. Stage2 keeps newlines/text; RoPE `position_ids = keep_abs` (no arange remap).
4. Custom generate continues RoPE from `max(pos)+1`; **no** forced `\n` stop.

## Gates

- `scripts/test_next_pack_parity.py`: keep-all vs stock max_abs=0 on
  672² / 480×640 / 1024×768; subset 320 patches + 48 newlines on square.
- `scripts/smoke_next7_a0_a4.py`: ALL_PASS with `pack_mode=spatial_unpad_newline`.

## Invalid prior ACC

`logs/acc_a0_a4_94pct_v2/` used `simple_crop_concat` — **do not cite**.
Rerun under `logs/acc_a0_a4_94pct_v3/`.

## Still not wired

QA-COVER B8 (algorithm, needs question context in Stage1). Config declares
`qa_swaps=8` but code does Dual COVER only. Separate from NeXT pack fidelity.
