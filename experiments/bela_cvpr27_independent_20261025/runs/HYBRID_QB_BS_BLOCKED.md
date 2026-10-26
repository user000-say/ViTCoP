# Hybrid 2×2 arms F*_QB / F*_BS — BLOCKED

Per `docs/SOURCE_AUDIT_AND_INTEGRATION.md`: STAR QR indices into BELA Stage2 (or BELA IDs into STAR Stage2) require shared ViT→projector tokens, original Patch ID nesting, and compatible attention backends.

**No production transplant adapter exists in this repo.** Inventing a partial hook would invalidate factorial attribution.

**Action while blocked:**
1. Run / reuse official **QS** full-system baselines (T32 / T64).
2. Reuse **BB** (L2 / E5).
3. Native Stage1 ablations on fixed E3/L2 schedules (penultimate / layer9).
4. Do **not** fill FACT_2x2 hybrid cells with approximate scores.
