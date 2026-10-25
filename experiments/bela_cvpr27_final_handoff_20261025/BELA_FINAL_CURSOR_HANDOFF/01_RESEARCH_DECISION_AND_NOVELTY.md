# Research decision memo — 2026-10-25

## A. Four observations with status
1. A0→A2: same nominal W2048, POPE +2.13, MMBench +0.94, GQA −0.22, TextVQA −1.08. [REPORTED]
2. A2→B2: first 12 decoder layers unchanged in nominal K, deep K20→32 and W2048→W2288: GQA +0.82, MME +29.3, TextVQA +0.24, POPE −0.08. [REPORTED]
3. W1024 accuracy degrades considerably for L4/L12 cut schedules; no evidence that W1024 **intrinsically** cannot perform better. [REPORTED partial]
4. Proposed "vision has been absorbed by text, and deep tokens are harmful" is NOT a proven mechanism. [HYPOTHESIS]

## B. The exact novelty difficulty
STAR-Pro (2026-09-05) already: pre-fusion subspace-cover candidate + progressive decoder attention refinement; the deployed LLaVA-1.5-7B T64 schedule is 128×12+37×8+18×12; T32 is 64×12+17×8+10×12. Its matched-compute 100 schedule sweep matters.
CoRePrune (2026-09-22) already: depth-dependent and set-conditioned removability, ViT-side progressive deletion + post-cross-modal set rescue.
QCPruner (2026-09-17) already: query-conditioned coverage on target and representative (and very different early dense LLM work from W1024).
MiCo (2026-09-28) already: unified coverage proxy before and inside the decoder.
SIEVE (2026-08-11) already: cross-modal absorption/residual and residual diversity.
ATP-LLaVA/PyramidDrop already: dynamic budgets and depth-progressive pruning.

Therefore **"COVER + RDRIFT + dynamic K" is not sufficient by itself**.
We must either:
(A) establish a concrete low-cost, causal, stage-conditioned retirement mechanism that demonstrably improves Pareto and crosses Stage1/Stage2;
OR
(B) conclude that evidence supports only a strong budget allocation/configuration; submit honestly with scope limits without fake novelty.

## C. Primary hypothesis, alternatives, and falsification
H1 — Task-specific evidence remains useful in a bounded middle-to-deep interval. A2's K20 from L13 may be too early; E1 (32 through L20, then 12) may recover GQA while preserving POPE at identical W. Test E1/E2/E3 with exact same first12 and selector path.
- H1 is weakened if E1/E2 do not improve paired GQA/OCR or materially worsen POPE/latency.
- E1's extra pruning event is a confound. Report selector overhead and a stage count control; do not call the result a pure timing effect.
H2 — W1024 fails because the first irreversible cross-modal cut is too early and the entry set is insufficiently mature. L2 (64 to L12, 17 to L20, 10 afterwards) vs L1 (96 to L4) tests schedule systems at exactly W1024.
- H2 is weakened if even legitimate late-first-cut with BELA COVER can't improve, or entry K64 loses key evidence.
H3 — RDRIFT/top-K produces an incomplete joint answer-evidence set at the final cut.
- Only act if fixed schedule/candidate, trace + paired-error analysis support wrong identity. Test QAttn/real RDRIFT/random and cheap query relevance before any extra guard.
H4 — Stage1 is the dominant error floor.
- If a small "Stage1-only high-W diagnostic" already loses most errors, no Stage2 fix can reliably rescue missing patches. Prioritize COVER entry signal/quality diagnosis, no costly new Stage2.

## D. Only one method candidate may be promoted this week
Temporary name: **Evidence Retirement Scheduling (ERS)**, not an established contribution.
Primitive: preserve a sufficiently broad entrance, use already-available text-visual signals at legal cut boundaries, keep still-needed evidence for additional blocks and release it once selected residual support is low. To be a real contribution, ERS must outperform:
- A2 at exactly W2048 on GQA/POPE/|MME/MMB| and preserve TextVQA/SciQA, with real time reported;
- same schedule with ordinary QATTN/actual RDRIFT and random controls;
- simpler fixed late schedule and Stage1-only quality control.
Do NOT add trainable controller, expensive perturbation replay, reactivation of hard-deleted tokens, or a second full forward pass this week.

## E. Efficiency caveat
W = sum K_l; Q=sum K_l^2 (visual-only square proxy).
A0 Q203616, A2 Q241536, E1 Q243456, B2 Q254016; same W does not mean same Q or TTFT.
Vision encoder may stay full-resolution for BELA's multi-layer COVER while CoRePrune removes vision tokens early: provide separate vision time, projector time, selector and LLM prefill, Fixed32, memory. Profiler instrumentation MUST not affect accuracy runs when disabled.

## F. Reviewer's strongest questions
1. Why isn't ERS merely a STAR-Pro variant with another schedule? => Need task-conditional mechanism and fixed-schedule-control gains; otherwise don't claim distinction.
2. Why isn't it CoRePrune's conditional removal? => Avoid repeat perturbation and rescue; show lightweight observable future-need signal and measured selector cost.
3. Why Stage1 dual features? => K-matched Layer9, penultimate, dual COVER + (selector × Stage1) cross table.
4. Why does POPE rise but GQA drop? => Paired category and error evidence, not blanket "hallu reduced"; show FP/FN.
5. Why only LLaVA-1.5-7B? => Try secondary backbone if possible; if not, explicitly limit scope.
6. Is it fair W? => Provide stagewise counts, encoder work, actual latency, raw numbers, protocol.
7. Is there a real improvement or a tuning artifact? => Unseen confirmation split, uncertainty intervals, one frozen hyperparameter.
