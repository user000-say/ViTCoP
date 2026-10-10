# E013 handoff + addendum checklist (live)

## User locks
- ACC = full 4-dataset only (never panel/limit as accuracy).
- Latency acceptability = Fixed32 |rel Δ| ≤ 15%; **headline HOLD** until ABBA explains E012 ~90ms (addendum).

## Addendum vs original
See `ADDENDUM_VS_HANDOFF.md`.

| Step | Status | Notes |
|---|---|---|
| B/C Install + CPU | done | live llama KVFIX-equivalent `6a4f4c31…` |
| E Smoke | PASS / VALID | EXTRA `visual_before=188` under KVFIX-eq code (not AUDIT_INVALID) |
| F screen1 | done | diagnostic only |
| G screen2 | done | diagnostic only |
| H Full ACC compare | RUNNING | do not interrupt; sole ACC evidence |
| I-1 ABBA GPU4 | RUNNING | then gate @15% |
| I-2 Stage1 latency | after gate | measure; headline HOLD |
| Bridge8/12 | **mandatory** after full | reuse S8 panel for Δ8; run S12 panel for Δ12 |
| CROSSIMAGE | skipped | no verified patch |
| Offline Δcorrect / features | after full | no fake logprob |
| K E013_RETURN | auto | watchdog |

Detached: master + full queue + watchdog. No hot-patch of running jobs.
