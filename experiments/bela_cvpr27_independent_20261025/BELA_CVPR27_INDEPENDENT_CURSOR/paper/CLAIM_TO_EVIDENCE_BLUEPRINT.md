# CVPR 2027 paper — scientific claim to evidence map
No result beyond history reports may be prewritten as achieved. Same full scoring protocol needed for every claim.

## Current candidate research question
"When is visual evidence admitted too poorly to be recovered by later reasoning, and when is evidence of sufficient quality wrongly retired during progressive inference?"
This is distinct from simply "attention + coverage + dynamic K" as a description, but *not automatically a novel method*. STAR-Pro/MiCo/CoRePrune/HiPrune/HAP show close prior art.

## Three possible outcomes — choose ONE after fixed 2×2
**Outcome A: Candidate-pool-limited.**
QR/QB substantially recovers W1024 GQA/POPE while BELA Stage2 held fixed; high-W Stage1-only proves missing patch evidence. Paper must introduce a novel effective Stage1 criterion (not generic QR/three-layer attention/head heuristics) and independently show transfer to later Stage2. If no real new criterion surpasses strong baselines, contributions remain limited and story must be lowered.

**Outcome B: Cross-modal-selector-limited.**
B/S substantially recovers scores with BELA original candidates held fixed; raw average attention already defective. Investigate calibrated head/question receiver score using available LLM attention at true cut layers, but PriorTR/HAP/QCPruner already study related cues. Only claim novelty for an empirically new causal mechanism, not renaming attention head weighting.

**Outcome C: Genuine interaction.**
Individually neither QR nor STAR Stage2 explains much, but interaction substantial. Then a real admission-to-retirement interface could matter: Stage1 emits **evidence provenance metadata**, Stage2 chooses survivors using evidence-specific conditional need across depth without new expensive passes. A legitimate contribution requires (i) joint error recovery, (ii) shuffled-provenance negative control, (iii) full independent 2×2 Stage1×Stage2 ablation, (iv) same W and time, (v) not merely MiCo coverage or CoRePrune set-greedy under a new name.

**Outcome D: Insufficient evidence.**
Use strongest validated simple baseline; do NOT invent novelty or SOTA; write method/analysis at supported strength. Submission is a goal, acceptance not guaranteed.

## 8-page CVPR 2027 storyboard
1. Introduction 0.9 pages — gap to STAR-Pro under *exact same layer-budget schedule*, task-dependent variance and real time (no cherry-pick).
2. Related work 0.7 pages — 2024 FastV, 2025 PyramidDrop/VisionZip, 2026 STAR-Pro/CoRePrune/QCPruner/MiCo/HAP/HiPrune and prior dynamic budget methods, note all overlaps.
3. Controlled empirical study 1.2 pages — E3 vs E1 (efficiency caution), 2×2 source/selector crossover at W1024/W2048, error and CI.
4. Final actual method 1.7 pages — only if working low-overhead new mechanism validated. Legal runtime metadata, Stage1/Stage2 interface, complexity; else rewrite as rigorous mechanistic analysis.
5. Main results 1.5 pages — GQA/POPE F1/MME Total+Perception/MMBench/ScienceQA-IMG/TextVQA/AI2D/MMMU; W1024/2048 and optional W4096; p50/p90 TTFT+Fixed32.
6. Independent ablations 1.2 pages — single vs dual ViT; real selector vs QAttn/STAR; random/shuffled controls; source ablations and timing.
7. Conclusion/limitations 0.8 pages — gaps, task-specific failures, one backbone if applicable; no assumed "all visual info absorbed".
Reference and **mandatory** LLM usage statement follow official 2027 limits.

## Mandatory figures
F1: same-W visual survival curves (token count per decoder layer) E3/E1/L2/STAR; avoid misleading final-K nominal ratios.
F2: actual model architecture only AFTER live source truth; arrows Stage1 token provenance → Stage2 scorer only when implemented.
F3: 2×2 factor effects with error bars GQA/POPE F1, discuss possible protocol mismatch.
F4: matched W accuracy versus TRUE end-to-end TTFT and Fixed32; mark E1 as latency-costly.
F5: input examples with object/word evidence IDs retained/dropped, both positives and failures; paired correctness.
Table 1: all absolute scores; Table 2: full system efficiency; Table 3: independent module and negative controls.
All figures/paper claims link to a raw scorer result and a source git hash.

## Conservative English abstract skeleton
"Under fixed layer-average compute budgets, token selection before fusion and token retirement during decoder interaction jointly determine the visual evidence available for answering questions. We systematically separate these two effects through matched-schedule component swapping and trace which kinds of evidence are lost. [INSERT VERIFIED METHOD IDEA ONLY IF IT BEATS PROPER CONTROLS]. Across [ACTUAL CHECKPOINTS] on [ACTUAL BENCHMARKS], the resulting approach [INSERT INDEPENDENTLY MEASURED RAW SCORES AND WALL-CLOCK TIMES]. We discuss when the benefit comes from evidence admission versus retirement, and expose failure modes on [VALIDATED TASK TYPES]."
Do not call component swapping itself a contribution unless it truly advances understanding beyond prior analyses.

## Paper operations
Official guidelines https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines
CVPR paper registration/abstract deadline Nov10 AoE, full Nov16 AoE; author list/OpenReview profile issues must be handled before enrollment deadline. Draft and experimental evidence should be complete by Oct31 to leave revision time.
