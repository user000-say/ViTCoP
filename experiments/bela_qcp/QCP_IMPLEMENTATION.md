# QCP Implementation

Date: 2026-10-27. Handoff: `bela_cvpr2027_handoff.md`.

## Algorithm identity

| Piece | Source | Notes |
|--|--|--|
| Stage1 Dual COVER K64 | `experiments/elva_r3/llava_stage1.py` + `native_cover.py` → `e013_choose` REPLACE/COVER | Unchanged |
| Stage2 first cut | E020 RDRIFT then QCP | QCP does **not** replace RDRIFT |
| Stage2 later cut | Native Q_ATTN | Unchanged; QCP disabled |
| Schedule L2 | `64×12 → 17×8 → 10×12`, W=1024 | Frozen |
| QCP operator | `experiments/bela_qcp/qcp.py:select_qcp` | New; not ICER rename |

## Decision rule

After RDRIFT keep \(A\) (\(|A|=17\)) on first cut:

\[
C(A)=\sum_{i\in S_0} q_i \max_{j\in A} s_{ij},\quad
\Delta C > \lambda\,(r(u)-r(v))+\tau
\]

- \(S_0\): Stage1 keep IDs (`LAST_ENTRY_KEEP`)
- \(s_{ij}\): nonnegative cosine on Stage1 Dual metric (`LAST_ENTRY_METRIC`, full 576)
- \(q_i\): same-cut `q_attn` on \(S_0\cap S\) (before QCP)
- \(r\): minmax native ranking score (RDRIFT/Q_ATTN), unmodified
- Defaults: \(\lambda=0.5\), \(\tau=10^{-4}\), max_swaps=2, donor_pool=16, drop_pool=12

## Env

- `BELA_QCP_MODE=qcp|random_swap|off`
- `BELA_ICER_MODE=off` when QCP is on (mutually exclusive)

## Arms

- **C0** L2 native: reuse `logs/.../L2_LATE_64_17_10` (no re-run)
- **C1** `L2_QCP`: QCP after RDRIFT
- **C2** `L2_QCP_RANDOM`: matched-count random boundary swaps
