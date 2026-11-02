# 2025–2026 Image Token Pruning Literature & Protocol Matrix
## Date: 2026-11-01; status: externally verified paper facts vs BELA hypotheses

### 1. STAR-Pro (2026-09), paper and public NeXT implementation
Paper: https://arxiv.org/html/2609.05916
Code: https://github.com/EasonAI-5589/starpro
NeXT evaluation: https://github.com/EasonAI-5589/starpro/blob/main/docs/evaluation.md
Schedule code: https://github.com/EasonAI-5589/starpro/blob/main/llava/model/language_model/modelling_llama_star.py

- Backbone: Vicuna LLaVA-NeXT-7B/13B (not Mistral; fixed checkpoints need actual SHA).
- Paper NeXT input: controlled 672×672 → 5 visual crop groups ×576 patch tokens =2880.
  The standard AnyRes `image_grid_pinpoints` and `mm_patch_merge_type`
  must be explicitly audited; special `image_newline` does not count as 2880 patches.
- NeXT-7B: 32 decoder layers, after L12/L20:
  T160 (94.4% of 2880 compressed) **320→91→46**;
  T320 (88.9%) **640→182→92**.
- NeXT-13B: 40 decoder layers, after L15/L30:
  T160 **320→80→40**; T320 **640→160→80**.
- These achieve **exact patch-token-layer budgets**:
  7B 5120 / 10240; 13B 6400 / 12800, respectively.
  T here refers **actual per-image 5-crop layer average**.
  STAR's tables sometimes denote T=32 and T=64 per-base-576-crop unit;
  for the five-crop image that maps to actual T=160 and T=320.
- STAR also published matched-W token-variation arms:
  7B 320→100→40 and 640→200→80;
  13B 320→84→34 and 640→168→68.
  These are in the present arm matrix as **reference ablations**,
  not invented BELA hyperparameters.
- STAR Stage1 residual pivoted-QR; Stage2 same-layer text→visual Q attention.
  Do NOT replace BELA's Dual COVER+QA with QR and call it BELA.
- STAR 10-task relative aggregate NeXT7B at aggressive ~94.4: 97.0%
  and at ~88.9: 99.0%; NeXT13B 97.5%, 98.9%.
  These are cross-paper targets not already reached by BELA.
- STAR actual code for NeXT image is available; prior video source release
  constraints are irrelevant to this image experiment.

### 2. MMTok (2025 preprint / ICLR 2026 version)
Paper: https://arxiv.org/html/2508.18264
Code: https://github.com/Ironieser/MMTok
- Joint text→visual and visual→visual coverage:
  text-aligned/projection-layer visual features for text-visual,
  raw pre-projector visual features for vision-vision.
- Default hyperparameters from the paper:
  `tau_t=0.02`, `tau_v=0.2`, `alpha=0.5`.
  Their precise softmax normalization is across visual candidates;
  these *are MMTok objective parameters* and should NOT be treated as
  direct drop-in BELA QA/donor weights.
- MMTok supports dynamic sample ratio for NeXT. "Up to 160/320"
  is an **input retained-token cap**, NOT progressive layer-average T160/T320.
  It reports NeXT7B up160 ~95.1% / up320 ~97.3%;
  NeXT13B up160 ~95.1% / up320 ~96.4%
  under its own comparative protocol, not our current budget metric.
- Ablation: combining image-image and text-image coverage is valuable.
  For BELA, validate **true mid+pen Dual COVER mapped through 5 crops**
  and use bounded original-candidate QA; do not directly import whole MMTok.
- In MMTok LLaVA-NeXT13B, later decoder refinement can help at extreme caps,
  but this is not a transferable claim about our precise L15/L30 cut choices.

### 3. TOPS (2026-10 arXiv v3)
Paper: https://arxiv.org/html/2606.27161
Code: https://github.com/MAX-7237/TOPS
- Objective: task relevance + information coverage + semantic diversity.
- Their hyperparameter table NeXT:
  7B at 94.4% alpha=.5 lambda=.5; 88.9% alpha=.5 lambda=.4;
  13B at both alpha=.5 lambda=.5.
  Stage1 shared alpha1=.5, lambda1=.5.
- These are **TOPS-specific objectives, not BELA weights**. They suggest
  general coverage preservation and small QA correction are sensible.
- Critical publication arithmetic check (Table 9):
  NeXT7B T160: stage320→160→40, cut after 12/24
     W=12*320+12*160+8*40=6080 → ACTUAL T=190, not160.
  NeXT13B T160: stage320→160→40 after15/30
     W=15*320+15*160+10*40=7600 → ACTUAL T=190.
  NeXT7B T320: 640→320→80 after12/24
     W=12160 /32=T380.
  NeXT13B T320: 640→320→80 after15/30
     W=15200 /40=T380.
  Thus **the written Table 9's nominal target does not match
  our arithmetic under the strict W layer-average formula**.
  This is a protocol ambiguity, NOT a claim of an implementation bug
  (source instrumentation on their implementation not performed).
  Do not enter these rows as strict matched-W in BELA table.
- TOPS reports NeXT7B rel 96.4 at94.4, 99.1 at88.9;
  NeXT13B rel around96.6/99.1 in source analysis.
  Different accounting/harness matters.

### 4. CoIn (CVPR 2026) and STD (EMNLP 2026)
CoIn: https://openaccess.thecvf.com/content/CVPR2026/papers/Du_CoIn_Coverage_and_Informativeness-Guided_Token_Reduction_for_Efficient_Large_Multimodal_CVPR_2026_paper.pdf
STD: https://aclanthology.org/2026.emnlp-main.1226/
STD code: https://github.com/Twilight03/STD
- CoIn aligns coverage and informativeness for aggressive high-res pruning.
- STD studies different stage reduction criteria/semantic drift;
  its public repo was only "Coming Soon" at time of inspection.
- We do not invent unsupported NeXT-13B model-specific settings for these
  methods, and will not claim official code was run on V100.

### 5. High-resolution query-aware crops (October 2026)
Paper: https://arxiv.org/html/2610.24102
- Query-aware, hierarchical crop budget; global vs local crop reallocation
  is a possible NeXT-specific improvement.
- It requires testing thumbnail/global preservation against OCR/local crop
  fragmentation. Our proposed `thumbnail_boost` is a **separate ablation**,
  not a faithful reimplementation of that paper.

## 6. Evidence-driven BELA response

Good LLaVA-1.5-7B performance proves that Dual COVER+bounded QA
can be competitive under prior configuration, but **does not prove** that
pooling 576 to 2880 and changing Vicuna7B→13B will retain performance.
The 5-crop feature arrangement is a genuine architecture change.
Prior BELA W4096 R6 Q_ATTN 98.77 vs hybrid RDRIFT 98.49;
W2048 hybrid slightly better. Therefore **Q_ATTN_ALL is the simpler
cross-backbone main control**, and RDRIFT remains an optional later test.
Prior video A800 60.07 on S56 is *not* an image NeXT result.

## 7. Comparison table requirements

Always show for each method:
model checkpoint/conv template/actual AnyRes patch N,
Stage1 input K0, decoder cuts, later survivors,
**patch-only W and actual mean**, newline/text token counts,
full prefill FLOPs + total TTFT, task metrics and matching Dense SHA.
MMTok's input-only cap, STAR's layer average and TOPS's published nominal
T are three different budget descriptions.
STAR-paper rel and BELA historical R6/R9 are not numerically the same
unless tasks, aggregation, metric scaling and matched Dense are identical.
STAR average normalizes MME total as MME/20; inspect exact paper eval script.
