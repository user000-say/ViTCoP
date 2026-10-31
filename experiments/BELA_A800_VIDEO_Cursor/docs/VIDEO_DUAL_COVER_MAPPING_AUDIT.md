# VIDEO_DUAL_COVER_MAPPING_AUDIT — 2026-10-31

## Verdict: **exact image Dual COVER cannot port** → labeled **SigLIP adaptation** only.

| Image (V100 LLaVA-1.5) | Video (A800 LLaVA-Video-7B) |
|---|---|
| CLIP-ViT-L/14, **576** patches (24×24) + CLS | SigLIP-SO400M, **729** patches (27×27), **no CLS** |
| Dual = Layer**9** + penultimate; CLS→patch attn scores | Tower truncates last encoder layer; payload = `hidden_states[-1]` |
| COVER `e013_choose` hardcodes `GRID=24` | After `avg_pool stride=2` → **13×13=169** decoder-visible IDs |
| Projector on penultimate CLIP rows | Projector on SigLIP last; mid-layer **not** projector-aligned |

## Mapping chosen for adaptation (explicit, not claimed = image Dual)

1. Capture SigLIP `hidden_states[9]` and `hidden_states[-2]` at **729** (pre-pool).
2. Apply the **same** `get_2dPool` avg stride=2 → **169** IDs; identity = pooled cell index `0..168` (same gather IDs as Stage1).
3. Saliency scores = **L2 of penultimate-pooled** rows (not CLS attention — SigLIP has no CLS).
4. COVER REPLACE on N=169, K=64, slots=32 with a **GRID=13** port of E013 relative-coverage (not the 24×24 binary).
5. Gather **projected** pooled tokens `[F,169,3584]` by those IDs.

## Labels
- Arm names: `V_NATIVE_L2`, `V_DUAL_COVER`, `V_DUAL_QA_B{2,4,8}` — never reuse baseline C/D/STAR labels.
- Papers/tables must say **SigLIP Dual-COVER adaptation**, not “production Dual COVER”.
