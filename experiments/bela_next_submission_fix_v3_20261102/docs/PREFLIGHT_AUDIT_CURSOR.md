# V3 Preflight Audit (Cursor) — 2026-11-02

## Verdict
**CPU READY / GPU BLOCKED** — independent package installed and CPU-verified; do not claim ACC win; do not merge into live s2qk tree until Dense/AnyRes/CLS GPU oracles PASS.

## Inventory
- Path: `experiments/bela_next_submission_fix_v3_20261102` (independent; not overlaid on `bela_next_7b13b_v100_20261101`).
- Algorithms: Dual COVER (e013) + QA B8 + Stage2 input-QK + `_remap_text_start` + global slots=32 + CLS saliency default.
- Host owned by live repo: lmms-eval, HF LlavaNext **5.20.2** (runtime hash matched).

## Environment
| Item | Value |
|---|---|
| Python | vitcop-v100-4model |
| transformers | 5.20.2 |
| torch | 2.6.0+cu124 |
| modeling_llava_next sha256 | `48aaa44b127c97692880f24b2aa5d16e9799b7ea18d7dbe63a5cacbece640ddc` MATCH |
| modeling_llama sha256 | `affc6a4ab36b562a6b4c90b84c4e7939c5417a69ad7bd37e858be5583f189830` MATCH |
| Holders GPU4–7 | ALIVE (do not kill per HANDOFF) |
| Free GPU for smoke | NONE (0–3 foreign; 4–7 ours) |

## CPU gates
| Gate | Result |
|---|---|
| `compileall` bela_* | PASS |
| `bash -n next_scripts/run_one_task.sh` | PASS |
| `pytest -q tests` (isolated `bela-v3-pytest` venv) | **33 passed** (`logs/pytest.log`) |
| `tools/preflight.py --runtime --arm NEXT7_T160_A0_PAPER` | PASS (`logs/preflight_runtime.log`) |

## Gap check (GPU still required)
- Dense parity (`BELA_NEXT_V2_ENABLE=0` vs stock): **UNKNOWN**
- AnyRes UID oracle (`BELA_NATIVE_DENSE_ORACLE=1`): **UNKNOWN**
- CLS eager oracle (`BELA_VIT_CLS_ORACLE=1`): **UNKNOWN**
- Prefill→decode KV/RoPE trace: **UNKNOWN**
- R4 / mean6 ACC: **NOT STARTED** (blocked; HANDOFF forbids releasing holders)

## Algorithm identity
- Dual COVER: e013 REPLACE/COVER for N=576; unpad uses original local IDs + 4×4 coverage (tests match historical e013).
- QA: packaged `bela_query_anchor` + tokenizer must be bound by lmms entry (no auto-reload).
- Stage2: V3 rewritten input-QK + padding/causal + text_start remap — engineering repair; **not** bit-identical proof vs frozen LLaVA-1.5 e018.

## Re-audit of P0 fixes in this tree (source review)
All P0-01…P0-16 items from `AUDIT_REPORT_CN.md` are present in V3 sources. No additional code patch applied this round beyond package install + CPU verification. Live production path unchanged.

## Next (when a free GPU exists without killing holders)
1. `BELA_LMMS_LIMIT=4` smoke A0 MME with oracles on.
2. Only if oracles PASS → gated R4 under new out-root.
