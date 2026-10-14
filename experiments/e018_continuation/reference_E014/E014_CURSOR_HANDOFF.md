# E014-AEE — Cursor / Experiment Agent Final Handoff
**Issued:** 2026-10-10
**Owner:** ViTCoP QED Code Agent
**State:** code integrated against the two fully uploaded E013 sources; CPU static/unit tested; server/GPU/four-full-N pending.

## 0. Scientific decision

Implement **image-adaptive evidence exchange under a fixed Stage1 visual budget of 172**, NOT
an E013 EXTRA transient-context mechanism. Historical E013 suggested COVER16 replacement improves
performance against Top172, but does not prove the optimal exchange count varies per image.
E014 is explicitly a falsifiable hypothesis about *how many and which original Core tokens to
exchange*. Preserve these propositions separately.

Prior draft bug **corrected**: returning 172-b permanent tokens + b transient tokens violates the
intended experiment: only the first 172-b would be eligible final evidence. E014 returns **172
eligible selected patches**, always. Nothing is appended as context and cluster labels remain
all nonnegative in the normal QED branch.

The gain/risk equation is a **proxy**, not a derived exact marginal set utility. Keep its limitations
in the paper notes. Do not claim it measures causal information loss or is automatically calibrated.

## 1. Immutable reference / installation

The uploaded E013 snapshots are complete, and modifications were made against their exact text.

| E013 file (uploaded) | Expected server tree path | SHA256 |
|---|---|---|
| clip_encoder_e013.py | `llava/model/multimodal_encoder/clip_encoder.py` | `b89208803528019460a47100fb5984e178c85cf12ca5f8ddb2981fda420ea73c` |
| vitcop_modeling_llama_e013.py | `llava/model/language_model/vitcop_modeling_llama.py` | `6a4f4c31334fa73f9e99395c93a6bff64e60c8b7b93f951f8e54c8c804651896` |

**STOP if current files are not those revisions**; produce a mapping/reconciliation diff rather than
force-applying. Do not overwrite a diverged server working tree. Before changing anything:
- `git status --short`, `git rev-parse HEAD`, `sha256sum` the actual two code paths.
- Back up or make a named commit for E013.
- `git apply --check E013_to_E014.patch`, then `git apply E013_to_E014.patch`.
- `git diff --check` and inspect the complete patch.
- Verify the working server really imports these exact modules; inspect sys.modules path and VLM
  initialization, not only file names. Wrong import path invalidates all results.

**External runtime dependencies not uploaded in the current message:**
`e013_stage1_support.py` (exact E013 COVER baseline), `e010_runtime_support.py`,
other original ViTCoP/LLaVA support modules, weights, and benchmark eval scripts.
They remain the server repository's responsibility. Do not reconstruct them from memory.
This bundle is not a self-contained LLaVA repository.

## 2. Source changes in this deliverable

A. **`llava/model/multimodal_encoder/e014_config.py`** (new)
- Parse/validate OFF-by-default `E014_ENABLE`.
- E014_VARIANT = BASE / FIXED_COVER / FIXED_COVER_PROXY / FIXED_RISK /
  ADAPTIVE_B_ONLY / AEE.
- Fixed E014_B in {0,8,16,24,32}, E014_LAMBDA >= 0, E014_PENALTY >= 0,
  E014_AUDIT, E014_SEED. No trainable controller.

B. **`llava/model/multimodal_encoder/e014_stage1_support.py`** (new)
- Top172 raw CLIP CLS-attention Core, complement residual.
- COVER candidate proxy `G_i=a_i(1-max(0,cos(x_i,mu_all)))`.
- Eviction proxy `R_j=a_j(1-max(0,max_{k!=j}cos(x_j,x_k)))`. Self diagonal
  strictly masked, not simply subtracted.
- Choose b from {0,8,16,24,32} via prefix-sum utility.
- Exchange exactly b IDs; sorted spatial order; no duplicated final tokens.
- `FIXED_COVER` calls **the original e013_choose(...,REPLACE,COVER,...)**
  and is therefore the benchmark fidelity anchor. This original helper must exist on server.
- `FIXED_COVER_PROXY` is a provisional approximation, never silently substituted for E013.
- AEE b selection uses `.item()` once (GPU->CPU sync) to materialize the dynamic length;
  report that overhead instead of claiming free dynamic selection.

C. **`llava/model/multimodal_encoder/clip_encoder.py`** (patched complete 1,989 lines)
- E014 branch before E013 in the budget, selection, and Stage1 output construction.
- Exactly 172 Stage1 visual embeddings. `final_idx` represents newly exchanged patch IDs.
- E014 provenance in `self.e014_last`: original, selected, added, removed, stats.
- `last_dynamic_info` includes E014 budget/variant/visual count.
- `E014_AUDIT=True` emits `[E014_STAGE1_JSON]` with actual IDs and score sums.
  OFF in latency tests. Existing E013 audit/branches left untouched.
- Requires E007 ORIGINAL_172 and VIC bypass, E010/E012, QED, no legacy Stage1 mods,
  excludes E008/E009/E013 simultaneous use.

D. **`llava/model/language_model/vitcop_modeling_llama.py`** (patched complete 4,614 lines)
- E014-scoped relaxation: K96/K64/K32 permitted only when E014_ENABLE=True.
  Original E007/E012 experiment acceptance rules otherwise unchanged.
- E014 guard freezes QED, S=D=L8 or L12, no E013/E009/E008 context.
- E014 static provenance appears in `e006_last`; **dynamic b / original patch IDs
  reside in the vision tower and `[E014_STAGE1_JSON]`**, not in LlamaModel.
  Do not invent cross-module fields that the current cluster-label transport cannot carry.
- Main SDPA, QED score path, physical pruning, decode attention masks, DynamicCache
  and S/D topology **not deliberately modified**.

E. `scripts/run_e014_k96.sh`, `run_e014_k64.sh`, `run_e014_k32.sh`,
   `e014_env.sh`, `run_e014_suite.py`, `analyze_e014.py`, `tests/test_e014_selector.py`.
- Wrappers set frozen flags but invoke YOUR REAL evaluator. No fabricated benchmark paths.
- Suite supports 8 GPUs, JSON command templates, per-run JSONL manifest and strict full-N counts.
- Analyzer reports B distribution and normalized same-K macro metrics.

## 3. Required P0 tests — execute before any full accuracy

**P0a: local/static (already PASS with uploaded source)**
- `python -m compileall llava/model/multimodal_encoder/clip_encoder.py llava/model/multimodal_encoder/e014_config.py llava/model/multimodal_encoder/e014_stage1_support.py llava/model/language_model/vitcop_modeling_llama.py`
- `python -m unittest discover -s tests -p 'test_e014_selector.py' -v`
- Twelve synthetic CPU tests currently PASS: cardinality, exchange sets, controls,
  adaptive choice, self-diagonal redundancy, config validation.

**P0b: legacy-off identity (server must verify)**
- E014_ENABLE=False: run one E007/E010/E012 K96 S=D=L8 image with frozen baseline env.
- Compare Stage1 IDs, QED selected IDs, final logits (within expected backend tolerance),
  decode1/decode2 KV and output against E013 source without this patch.
- Confirm E013 REPLACE_COVER16 can still run when E013_ENABLE=True, E014_ENABLE=False.
- This is stronger than a compile check: confirm project import path really uses patched code.

**P0c: exact E013 comparator (critical)**
- Run old E013:
  `E013_ENABLE=True,E013_VARIANT=REPLACE,E013_PICK=COVER,E013_B=16`.
- Run new E014:
  `E014_ENABLE=True,E014_VARIANT=FIXED_COVER,E014_B=16`
  and identical model, image, question, CLIP geometry and S=D settings.
- Compare all original/selected/added/removed IDs, Stage1 feature tensor, final QED
  survivor IDs, logits and 4 metrics on same sample. If nonidentical, STOP and investigate.
- Do not substitute `FIXED_COVER_PROXY`: it may follow a related formula but the exact
  uploaded E013 helper is absent from this package.

**P0d: GPU smoke on real model**
- K96, L8: run BASE / FIXED_COVER16 / FIXED_RISK16 / ADAPTIVE_B_ONLY / AEE.
- E014_AUDIT=True and E012_KV_AUDIT=True ONLY here; COMPUTE_EFFICIENCY=False.
- Check Stage1 in 576, out 172, 172 eligibility, IDs within 0..575, unique,
  exactly b added and b removed, no overlaps/duplicates.
- Check QED selector calls =1 on prefill, 0 on decode; physical length K96
  post-layer L8; post-prune attention mask/cache consistent for decode1/decode2.
- K64 and K32: separate GPU smoke (newly relaxed gates, not previously validated)
  with the SAME invariants, no silent fallback to K96.
- Audit: `E014_AUDIT=True` prints JSON **only for offline inference**.
  AEE b histogram should be measured, not presumed non-degenerate.
- Run with variable image aspect/content cases and benchmark real samples.

If a smoke gate fails, diagnose and patch only the isolated E014 branch, rerun OFF identity,
and log the failure. Do not change historical E013/E012 reference semantics.
Small prototype failures should trigger diagnosis, not terminate the half-day/day experiment
plan without investigating actionable alternatives.

## 4. P1 experiment matrix — four FULL benchmarks, not a small panel

Model/protocol MUST match E013:
- LLaVA-1.5-7B, FP16, SDPA,
- frozen image preprocessing, generation options, question span,
- Stage1=172, QED final-K=96 initially, S=D=L8 primary (L12 bridge if useful),
- E013/E008/E009 off during E014 comparisons,
- no training, no added decoder layers, no altered QED formula.

**K96 primary:** four full tasks *every arm*:
- BASE
- FIXED_COVER B=0/8/16/24/32 (original E013 helper, exact identity gate)
- FIXED_RISK B=16
- ADAPTIVE_B_ONLY
- AEE

This is 9 method-budget points × 4 full benchmarks = 36 evaluation jobs,
parallelizable across 8 GPUs. Budget/variant consistency and full N are part of
acceptance, not optional later paperwork. Do not cut to tiny panel after smoke.

**K64/K32 stress-tests (only after K96 GPU proof):**
- BASE
- FIXED_COVER B=16 or K96 best-fixed-b pre-registered ahead of time
- FIXED_RISK same B
- ADAPTIVE_B_ONLY
- AEE
= 5 × 4 × 2 = 40 full jobs. The provided scheduler implements these job plans.

**Offline diagnosis (separate from timed/accuracy matrix):**
- Collect Stage1 JSON audit on representative samples across all four benchmarks.
- Histogram b; compare b across image/task characteristics and correctness changes.
- Paired per-example comparison AEE vs best fixed; confidence intervals/bootstrap.
- Evaluate `lambda` / `penalty` sensitivity on DEV/PILOT only; freeze before FULL test.
- Optional oracle-best-b is **offline diagnostic** requiring the same example under
  all fixed-b arms; never present in-sample oracle as a deployable method.
- Preserve negative runs. Do not optimize only headline macro and discard MME drops.

Use previously reported E013 full-split counts only if your benchmark commands
are the same: GQA=12578, POPE=9000, MME=2374, MMBench=4329.
See `configs/expected_n_e013_protocol.json`; reject silently filtered evaluations.

Benchmark metrics:
- GQA full accuracy
- POPE full F1 (check split/aggregation protocol)
- MME official score
- MMBench full official score

Also measure:
- Stage1 selector overhead, CLIP & prefill, TTFT, Fixed32/E2E,
- GPU peak memory, token trajectory, b distribution, and compute config.
- Compare *paired, identical protocol*, no audit hooks in timing.
- E014 changes Stage1 composition, **not** the Stage1 token count;
  therefore improved speed is NOT guaranteed.

## 5. 8-GPU full-launch contract

Prepare `commands.json` with one **actual verified full-eval command** for each
GQA/POPE/MME/MMBench and `expected_n.json` recording the true test N for those
specific split/filter settings. Never guess exact N. Each command receives environment
from the per-K wrapper and writes to `{out_dir}`. The existing evaluator/postprocessor
must write `{out_dir}/eval_summary.json` with `n` and `score`.

```bash
python scripts/run_e014_suite.py --plan k96 --dry-run
python scripts/run_e014_suite.py --plan k96 \
  --cwd /real/vitcop/repo --gpus 0,1,2,3,4,5,6,7 \
  --commands-json /abs/commands.json \
  --expected-n-json /abs/expected_n.json \
  --out /abs/logs/e014/k96
```

Manifest statuses:
- `VERIFIED_FULL_N`: command success and N exact
- `FAILED_RUN`: nonzero process
- `FAILED_FULL_N`: missing/mismatched n
- `FAILED_DRIVER`: scheduler/command parse exception
- `UNVERIFIED_SMOKE`: --allow-unverified; NOT a full result

After K96 gates, launch K64 and K32 in separate plans. Max parallel jobs=number
of unique GPU IDs; tasks/log paths never share destination folders. Preserve full
stdout/stderr (in `run.log`) and evaluator raw outputs.

Any `n` mismatch INVALIDATES that arm's accuracy row; do not silently rescale
or truncate to fake completeness.

## 6. Must-return artifacts / analysis

Return a single structured handoff `E014_RETURN.md` or JSON with:
- git SHA, source file SHA, actual import path, model SHA/dtype/backend, GPU type,
  benchmark versions, exact eval entrypoints/configs/seeds.
- CPU tests result and OFF identity.
- E013 REPLACE_COVER16 ↔ E014 FIXED_COVER16 token-ID + output parity.
- Per-method/per-b/full-benchmark N and primary metric.
- Per-method/per-b wall time, Stage1 latency, TTFT, Fixed32, E2E, GPU peak.
- K96 then K64/K32 KV/cache/selector smoke traces.
- AEE per-image b hist + utility summaries, lambda fixed values.
- Pairwise comparison vs **best fixed** (not only Top172), uncertainty and failure modes.
- Complete failed jobs/logs, not only the winners.
- A judgment: AGREE / MODIFY / REJECT E014 adoption with measured Pareto evidence.

Acceptance:
1. Correct 172 eligible Stage1 and K96/64/32 physical final counts.
2. E013 baselines fully reproducible; no history regression.
3. Four full N for every named metric row.
4. AEE > strongest same-K fixed COVER comparator in reproducible, meaningful ACC
   **without violating the agreed latency/TTFT gate**. Treat <15% slowdown as an
   exploratory engineering tolerance from historical discussions, NOT automatic
   evidence of speedup; prioritize a Pareto improvement if possible.
5. Dynamic b has credible non-degenerate per-image behavior; otherwise label it
   a tuned fixed heuristic, not a successful adaptive controller.

## 7. Known limitations and failure watchlist

1. **Exact E013 COVER scoring unavailable here.** FIXED_COVER executes the server's
   original helper; the proxy score used by AEE must be audited for ID/ranking differences.
2. `G` and `R` may live on different distributions. Default lambda=1 is not
   academically validated. b can collapse to 0 or 32; report it.
3. Approximate independent gain/risk is NOT the exact swap utility
   `F(S-j+i)-F(S)`. Avoid falsely presenting it as such in paper.
4. Computing 172×172 cosine matrix and extracting adaptive integer b may
   add measurable latency. Add timer instrumentation; do not hide extra operations
   under the Stage1 call.
5. The two given sources reference other runtime files; GPU execution and module
   import cannot be verified offline in this environment.
6. K32/K64 paths relax **only** E014 experiment guards. E012 topology and masks
   require real server smoke: compile success does not prove correct prefill/decode.
7. E014 AEE and exact E013 COVER might differ for reasons OTHER than eviction:
   candidate score approximation must be explicitly contrasted by FIXED_COVER_PROXY.
8. No new adaptation algorithm is publishable solely from the code; only validated
   results/ablation/mechanism comparisons justify a CVPR claim.

## 8. Rollback

Use a clean git commit of E013 for rollback. `git diff` and `git status` after each
operation; revert only the E014 commit or run `git apply -R` on the original patch,
not a destructive reset in a dirty repository. Verify OFF baseline identity again.

## 9. Regeneration 2026-10-10 addendum

- The two uploaded E013 files were read in full and verified by SHA256.
- Every CLIP/Llama edit is marked `# E014-AEE-BEGIN/END`.
- Six additional synthetic CPU tests check zero-b fast path, exact helper order,
  helper signature, incorrect comparator cardinality, geometry and utility extremes.
- `E014_VARIANT=FIXED_COVER` still requires the server-original `e013_stage1_support.py`: do NOT substitute a proxy without renaming the arm.
- AEE's `b` dynamic decision has a `.item()` sync; measure it honestly.
- Hold full GQA/POPE/MME/MMBench accuracy to historical N or verified equivalent.
- Give Cursor enough autonomy to spend a meaningful half-day to one-day phase on
  smoke, paired ablations, four-full-N evaluations and timing without arbitrarily
  aborting exploration after the first nonpositive pilot result; report all runs.
- K64/K32 are newly allowed by E014-scoped config gates, not validated GPU runs.
