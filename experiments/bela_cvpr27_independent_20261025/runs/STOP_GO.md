# STOP_GO — Independent handoff (2026-10-26)

## Completed / reused (do not regen)
- F1024_BB (L2), F1024_QS (STAR T32), F2048_BB (E5), F2048_QS (STAR T64), E3/A2/E1/W1024_EARLY ACC.
- CODE_CALL_MAP, SOURCE_HASHES, TIMING for E3/E1/E5/L2/A2/STAR_T64.

## Blocked
- F*_QB / F*_BS hybrids: no clean Patch-ID + attention transplant → **no invalid scores**.

## In flight / next
- E3 Stage1 penultimate-only vs layer9-only (native, same E3 schedule+Stage2).
- After that: POPE confusion / paired BB↔QS errors if sample jsonl align; STAR T32 TTFT if needed.
- **Do NOT** implement provenance retirement until Stage1/S2 causal proof from above.

## Interim scientific read (not a method claim)
- Same schedule: STAR QS wins **GQA/POPE**; BELA BB/E3 wins **mean6** via SciQA/TextVQA/MME.
- True Stage1-vs-Stage2 attribution **incomplete** without QB/BS.
- E1 rejected as headline (Fixed32 +10%); E3 remains W2048 practical baseline.
