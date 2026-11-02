# Preflight: BELA_NEXT_NATIVE_FIX_KIT_20261102 (2026-11-02)

## Inventory

| Kit provides | Repo already owns |
| --- | --- |
| `native_layout` provenance pack map | TF5.20 `pack_image_features` / our next_pack |
| `legacy_bridge` Dual+QA fail-closed | e013 Dual COVER + `bela_query_anchor` |
| `progressive_hooks` (TF4.48 DynamicCache) | Our TF5.20 Stage2 Q_ATTN + cache prune |
| `runner` eager reference | lmms_eval + hooked generate |

## Gaps closed in this round

1. **ViT mid layer 9** (was wrongly 12) — matches 1.5-7B Dual-depth.
2. **Global QA B8** via `bela_query_anchor` (was missing).
3. **Native pack + structural newline keep** via kit `map_native_packing` / `structural_keep_mask`.
4. **Stop forcing `[[672,672]]`** — Dense pinpoints restored.
5. Stage1 runs in prefill with `input_ids` so QA sees the question (not in `get_image_features`).

## Substitutes labeled

- Stage2 still uses our TF5.20 QK-based Q_ATTN (not kit’s `output_attentions` eager hooks) — same math intent; cache API differs on 5.20.
- Dual COVER is **crop-grouped e013** on packed patch space (not a new TopK).

## Prior ACC

- `acc_a0_a4_94pct_v2` / partial `v3`: mark `LEGACY_*` — not native Dual+QA L9.
- v3 MME early: 1639 abs / **89.74%** vs NeXT Dense → below 92% stop; superseded by this kit wire.

## Gate before full ACC

1. Kit CPU unit tests PASS.
2. Smoke A0–A4: `mid_layer=9`, `legacy_dual_called`, `legacy_qa_called`, schedules, Dense ENABLE=0 parity.
3. Then A0 R4 only; stop if R4 mean rel < 92%.
