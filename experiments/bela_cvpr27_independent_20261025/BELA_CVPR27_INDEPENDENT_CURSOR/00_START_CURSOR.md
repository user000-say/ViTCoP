# BELA/ELVA: CVPR 2027 Evidence-Driven Final-Sprint Handoff
**Version 2026-10-25 / replaces all prior exploratory Handoffs.**
Read this file before running anything. Today is 2026-10-25. By October 31, lock all crucial experiments, a coherent main-method candidate (or a defensible negative decision), and a full paper draft. CVPR 2027 registration Nov 10 AoE; full text Nov 16 AoE.

## NON-NEGOTIABLE INDEPENDENT RULES
1. Ignore pressure to implement a user/agent-suggested idea without causal evidence. A new module is *not* progress unless it beats clean same-cost baselines on GQA/POPE/MME/MMB plus TextVQA and wall-clock time.
2. Correct metrics: **GQA raw ACC, POPE F1, MME Total and Perception SEPARATELY, MMBench, ScienceQA-IMG, TextVQA**. Secondary AI2D/MMMU. W = total visual token × active decoder layers is NOT full FLOPs; final K32 is not mean K32.
3. Physical token removal, correct KV/mask/position IDs, same exact checkpoint, raw image IDs, official prompt, answer parser and evaluator must be confirmed. `RDRIFT at both cuts` remains UNVERIFIED.
4. Comparisons on the final held-out benchmark must NEVER be used to retune hyperparameters. Use fixed sampled development partitions and paired, image-cluster diagnostics, then freeze and confirm.
5. W1024 and W2048 prioritized; W4096 after mechanism and paper; never inflate W to claim a quality improvement.
6. One single algorithmic idea at most after the audit. No copying STAR-Pro QR + CoRePrune rescue + HAP head weight into a paper "new" method. Those are controlled references, not candidate contribution.

## LATEST GPU EVIDENCE (user reports, not independently reproduced here)
| Arm | W | GQA | POPE F1 | MME Total | MMB | mean6 | TTFT | Fixed32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A2 (172×4+120×8+20×20) |2048|57.62|85.44|1774.9|63.14|96.68|122.5|1218|
| E1 (172×4+120×8+32×8+12×12)|2048|57.95|85.46|1797.9|62.71|96.86|127.0|1340|
| E3 (172×4+120×8+24×10+16×10)|2048|57.76|85.46|1779.1|63.32|96.85|123.0|1204|
| E5 (128×12+37×8+18×12)|2048|57.88|84.96|1735.7|61.86|96.08|121.4|1199|
| L2 (64×12+17×8+10×12)|1024|55.57|82.21|1651.8|61.34|93.56|114.5|not supplied|
| W1024 early (96/60/8, real cut intervals must be checked)|1024|54.06|81.95|1736.7|60.31|93.24|not supplied|not supplied|
STAR-Pro paper, same LLaVA model family: at W2048 (128/37/18) GQA59.41 POPE87.52; at W1024 (64/17/10) GQA57.20 POPE86.60. **Not a fair matched implementation comparison until same prompts/evaluator/checkpoint**. Its published MME is perception-scale and not directly comparable with our reported MME Total.
GQA/POPE gaps persist even using EXACT STAR-Pro schedule shape. More ratio sweeps do not answer which selector is defective.

## INDEPENDENT RESEARCH DECISION
**Reject E1 as main method:** mean6 +0.18 pp vs A2 but Fixed32 +122 ms (10.0% slower) and TTFT +4.5ms; the accuracy gain does not explain cost. Keep E1 as survival-timing mechanistic evidence.
**Keep E3 as current practical W2048 baseline** (not new contribution): mean6 96.85, GQA57.76, POPE85.46, MMB63.32, TTFT123.0, Fixed32 1204. It has THREE physical decoder cut events after layers 4,12,22 — audit ALL three.
**L2 W1024 is the current late-cut reference:** despite copying rival budget trajectory, POPE still lags. Stage1/signal/implementation factors need component-level causal isolation.
**Do NOT claim "late tokens harmful" or "visual info fully absorbed"**. B2 prior results: retaining more K at W2288 recovered GQA/MME; mixed results indicate survival utility is task-dependent.
**Stage1 dominance is only a hypothesis.** Stage1 single-vs-dual does not settle the interface: must quantify whether entrance candidate quality or downstream scorer accounts for the STAR-Pro gap.

## PRIMARY EXPERIMENT — CROSSOVER AT IDENTICAL PRUNING SCHEDULE
This is the highest-information experiment of the week, not a paper method.
For W1024 `64×12+17×8+10×12` and W2048 `128×12+37×8+18×12`, perform 2×2:
- B/B: BELA exact CURRENT Stage1 COVER + BELA actual Stage2. Previous L2/E5 is REPORTED baseline.
- Q/B: competitor STAR-Pro official QR Stage1 candidate *indices* + BELA Stage2 on same model/tensors.
- B/S: BELA Stage1 selected original indices + STAR-Pro official Stage2 attention selection on same model/tensors.
- Q/S: official STAR-Pro end-to-end, same checkpoint and official scorer; run on OUR evaluator too.
Keep K0, cut layers, all intermediate K, beam/temperature/prompt, image preprocess and image IDs identical.
If transplantation cannot preserve attention/KV/patch mapping, **BLOCK** cross arm; do not publish invalid scores. Reproduce competing method separately and continue native ablations.
For each benchmark, compute factorial Stage1 effect, Stage2 effect and interaction. Pair same sample IDs. `tools/factorial.py` calculates from official-scored JSONL.
If QR entrance wins for GQA/POPE while Stage2 unchanged, prioritise independent Stage1 design.
If STAR attention wins holding BELA candidates fixed, prioritise **attention signal quality**, not another ViT layer.
If both help and interaction significant, real joint design is needed; BUT do not simply publish Q/S as new BELA.

## SECOND EXPERIMENT — IDENTIFY BOTTLENECK WITHOUT COPYING RIVALS
At E3 (fixed W2048, K0=172 and 3 true cuts), compare Stage1:
S1. existing dual COVER (exact current code).
S2. best simple single layer: penultimate (L23 if layer indexing confirmed), original score/rule held matched.
S3. Layer9-only same K and same baseline selector.
S4. depth-fusion exactly current with a more robust within-image score calibration; run only if raw layer/head score analysis reveals scale mismatch.
All above fixed SAME Stage2 E3. Do not replace COVER and layer source simultaneously without a crossed control.
A separate two-stage scorer check holds Stage1 fixed and compares actual S2 score vs Q-Attn, RDRIFT (if valid signal) and random seed 11/22/33 **at the actual cut**. If scorer already equals one control, do NOT duplicate.
Examine POPE FP/FN and TextVQA lost/correct pairs. An entry-only high-W diagnostic may reveal an irreversible Stage1 error floor, but it is NOT an efficient or equal-W competitor.
Do not sweep 60 ViT/head variants this week; no real ViT encoder progressive pruning before proving Stage1 is limiting. CoRePrune already established progressive ViT and requires full CLIP re-execution, not fake gather.

## ONLY CONDITIONAL METHOD CONTRIBUTION
Research name only, NOT a promised CVPR novelty: **provenance-aware evidence retirement**.
If and only if Stage1 evidence is shown to predict corrected/failure cases and Stage2 candidate-ranking has systematic coverage failures:
- Stage1 provides original Patch IDs, calibrated cross-depth evidence provenance (e.g. stable vs depth-exclusive), and a cheap per-candidate representation of coverage; DOES NOT use target answer labels or future decoder attention.
- Stage2 uses exactly available current question-to-visual scores; starts from its existing TopK and makes at most b=4/8 fixed-cardinality bounded corrections for candidates identified as uncovered AND problem-relevant; no additional full forward, no more cut events, no restoration of truly pruned tokens.
- Same W, same current E3 physical cut positions, all supported original identities, selector overhead separately measured.
- Necessary ablations: current A2/E3, score-only, provenance-only, whole policy, shuffled provenance, random-b exchanges, best static TopK, STAR-Pro/CoRePrune/MiCo method comparison. A mechanism is novel ONLY if behavior and outcomes are genuinely different from prior query-weighted coverage / set deletion.
If these causal preconditions are not established, do NOT implement it. Freeze simpler winner; manuscript must not claim new latent evidence theory.

## GPU DISPATCH (four GPUs originally 0,2,4,7; rebalance if hardware differs)
Day 1 / P0: source/evaluator audit + patch ID/KV traces + reproduce official STAR-Pro Q/S W1024 and W2048. Run 3-image smoke under both source checkouts.
Day 2 / P1: 2x2 crossover dev GQA/POPE + TextVQA; simultaneously single-vs-dual Stage1 native E3, POPE subtype diagnostics.
Day 3 / P2: diagnose errors and one gated minimal correction; 3 real input smoke, no more than two candidate hyperparameter values. Run W1024 matched validation FIRST.
Day 4 / P3: freeze selection from dev, 6–8 tasks full official confirmation and real latency; same model/checkpoint/time scope. Cross Stage1 × Stage2 independent control if method candidate survives.
Day 5 / P4: one additional backbone if feasible, otherwise full negative controls/failure cases, finish tables/figures and paper.
By Oct29 FREEZE new algorithm. By Oct31 full eight-page draft (not guarantee publication). Paper drafting begins immediately; no fake achieved claims.

## GO/NO-GO (precommitted)
- Improvement objective: meaningful joint GQA+POPE (toward ~59/87 W2048, ~57/85 W1024) plus no systematic TextVQA/SciQA harm, with <=~3% measured full TTFT/Fixed32 relative best same-W simple baseline; this is a **target**, not promised outcome.
- One-task GQA improvement with major POPE/MEE drop is not a winner.
- Small mean6 <~0.15 pp without consistent paired or out-of-subset gains is insufficient novelty.
- Any mechanism equal to shuffled tags or random exchanges -> kill module.
- If baseline/full STAR repro diverges drastically from paper, audit environment before interpreting candidate gaps.
- If Stage1 wins and Stage2 stays weak, admit separate contribution and abandon grand unified story. If neither wins, retain a validated strong fixed baseline and write honest limitations. **Submitting is required; inventing novelty/results is not.**

## CURSOR REQUIRED HAND-IN (nightly)
`CODE_CALL_MAP.md` (three cuts), `PATCH_ID_TRACE.jsonl`, `SOURCE_HASHES.json`, `FACT_2x2.csv`, `PAIR_ERRORS.json`, `POPE_CONFUSION.json`, `RESULTS_MASTER.csv`, `TIMING_RAW.csv` (full image-to-first token AND Fixed32), `SOURCE_DIFF.patch`, `STOP_GO.md`, `PAPER_MAIN.tex`.
Raw stdout/stderr, per-sample predictions, exact eval commands and dataset manifest hashes. Every 'reported', 'verified-gpu', 'not-run', 'blocked' MUST be distinct.
