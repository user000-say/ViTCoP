# BELA / CVPR 2027 — Stage1 QA-COVER FINAL Cursor Handoff
**Date:** 2026-10-27  
**Status:** CPU reference tested; **not merged to production**, **no new GPU accuracy has been observed**.  
**Single algorithm candidate:** Question-Anchored Dual COVER (QA-COVER), a *diagnostic* query-conditioned entry replacement on native BELA.  
**Must preserve:** native Dual COVER + native RDRIFT, K64 L2 `64×12→17×8→10×12` (W1024) and native E3 W2048.

## 0. Research decision; do not keep changing the objective
Past user-reported L2: GQA55.57, POPE F1 82.21, MMB61.34, TextVQA-OCR53.41. Native E3: GQA57.76, POPE85.46, MME1779.1, MMB63.32, TextVQA55.13.
B1 COVER32 FLAT53.09/74.53; B2 ENTRY_ORDER52.16/79.09; B3 S16 55.15/80.96; admission b=2 55.64/82.15; QCP C1 55.69/82.28, 99.6% samples swapped. **Stop all those branches; await already running QCP C2 without resubmitting.**
CD32 GQA57.00/POPE87.67 and STAR-Pro T32 GQA57.19/POPE86.67 are strong competitors, **not BELA components or proposed novelty**.

L2 POPE 9k balanced: FN1188, FP245, TP3312, TN4255, yes-rate39.52%, F1~82.21.
CD32: FN621, FP470, TP3879, TN4030, yes-rate48.32%, F1~87.67. CD reduces FN *and increases FP*. H1 missing entry evidence, H2 Stage2 removal, H3 cross-stage mismatch, and H4 yes/no readout bias remain competing explanations.

## 1. Why this one last Stage1 arm?
Change the **identities** of a bounded number of native K64 patch tokens **before first LLM prefill**; do not change K, original patch feature values, decoder schedule or RDRIFT.
Different from failed Stage2 QCP: donors can come from **all 576 original vision patches**, not just the already-alive K64. Different from admission b=2: donors are selected based on *current question* evidence rather than only ViT source-layer identity.

The existing Dual COVER remains the native strong base. Query relevance must be a real signal: current proposed adapter computes cosine between **already-existing full-grid mm_projector outputs** and **question-only input-token embeddings**. Such cosine similarity is only a **hypothesis**, not an established joint embedding space. Do NOT claim aligned semantics. Use mismatched-query control and matched random substitutions before retaining the module. The extra full-grid projector cost must enter TTFT.

AgilePruner demonstrates saliency vs diversity depends on scene complexity, and hallucination need not decrease with diversity. Therefore `saliency_guard_slots` defaults to **0** and is only a preregistered secondary ablation if saliency maps are already available without changing the attention backend. No dynamic K/router in this sprint.

## 2. Source-code binding in the LIVE BELA repository
Locate using `rg`:
- `experiments/elva_r3/llava_stage1.py`: original Dual COVER stage, `selected["metric"]`, production K selection, final physical patch gather.
- `LLaVA/llava/model/e018_stage2_support.py`: original RDRIFT and L2 cut timing.
- `prepare_inputs_labels_for_multimodal`, `encode_images`, `get_model().mm_projector`, `get_input_embeddings`.
- Existing `LAST_ENTRY_KEEP`/`LAST_ENTRY_METRIC`: inspect request scope. Global mutable state is **not** safe for concurrent requests; first run batch=1 and refuse overlap.

Copy `experiments/bela_query_anchor/*.py` and `configs/*.json` into a fresh isolated branch. The `llava_adapter.py` entry point is **real importable Python**, but the next snippet is **pseudocode** because the production repo and exact call site were NOT supplied:
```python
# Immediately AFTER native Dual COVER returns ORIGINAL IDs and BEFORE image gather.
result = apply_llava_query_anchor(
    mode=os.getenv("BELA_QA_ENTRY_MODE", "off"),
    native_keep_ids=native_keep,                 # K64 original CLIP patch IDs
    dual_metric=selected["metric"],             # [576,D] native descriptor
    full_vision_features_for_projector=full_hidden,  # [576,C] input expected by mm_projector
    question_only_ids=question_content_ids,      # real question, NO -200/labels/templates
    mm_projector=model.get_model().mm_projector,
    llm_input_embeddings=model.get_input_embeddings(),
    projected_full_vision=already_projected_grid_or_None,
    config=AnchorConfig(max_swaps=8),            # other defaults in config JSON
    diagnostics=trace,
)
keep = result.keep_ids
# Gather the ORIGINAL visual features for the decoder by these ORIGINAL IDs.
# Overwrite request-local entry IDs AFTER selection; keep full-grid metric as descriptor.
# NEVER feed the Layer9 descriptor as the LLM input if native LLaVA uses a later layer.
```
If using `question_only_ids` with language templates `Is there a ...?`, prefer a **frozen, label-free content-token mask** so generic "is there" tokens do not drown out object nouns. Pass `question_content_mask` to the adapter; the adapter forbids negative image placeholder IDs even if the mask would remove them. Freeze this extraction in the experiment spec; never use ground-truth object labels.

The provided input helper is an **experimental prompt-to-patch proxy**; unverified alignment is a scientific limitation. It must not be passed to production by default. `off` must be exactly equivalent to L2, including patch ordering and predictions. Single-image LLaVA-1.5 576-patch only; non-576 or multi-crop requires explicit mapping tests.

## 3. Required controls: run on SAME images, prompts, generator and model
| Arm | Stage1 | Stage2 | W | Question source |
|---|---|---|---|---|
| `L2_NATIVE` | exact current Dual COVER K64 | native RDRIFT | 1024 | actual |
| `S1_QANCHOR_L2` | QA-COVER K64 (`max_swaps=8`, other hyperparameters fixed) | native RDRIFT | 1024 | actual |
| `S1_QUERY_SHUFFLE` | identical QA-COVER K64 | native RDRIFT | 1024 | **other image** question, frozen permutation |
| `S1_MATCHED_RANDOM` | **same number + same drops** as QA-COVER, donors random | native RDRIFT | 1024 | actual |

**Not another open-ended search:** no sweeping 2,4,8,16 swaps, no CLIP-text-tower substitution, no schedule scan. The saliency guard remains OFF; if the query hypothesis fails, do not automatically "rescue" it by making the attention guard the main algorithm.

Off-line diagnostics (no new GPU if logits/logs exist):
1. Finish existing C2 and document QCP's `delta_coverage <=1e-6` / rank-penalty signs if diagnostic logs exist; do not reopen QCP.
2. Match POPE native, CD, C1 and new S1 exact sample IDs and scoring parser; pooled vs macro splits explicit. Confirm tokenizer yes/No variants, any invalid outputs, and whether native readout uses greedy generated text or restricted-token logits.
3. POPE pairs: track baseline FN→TP and TN→FP **on identical images/questions**; report yes-rate and each random/popular/adversarial split.
4. GQA: record exact-match pairwise gained/lost by question type; crop/patch overlay on a small diagnosis set if COCO object boxes are available; visual evidence claimed only with paired causal replacement not proxy heatmaps.

## 4. Production gates before full evaluation
**G0 / correctness:**
- `BELA_QA_ENTRY_MODE=off` outputs identical K/patch IDs and model predictions to existing L2.
- Stage1 native and final K=64; RDRIFT remains 64×12→17×8→10×12 = W1024.
- Unique original patch ID; sorted spatial order; same actual LLM visual embedding for a given ID.
- Negative/non-question IDs rejected, content text truly from current image's current prompt.
- No model answer or future states are used; no extra full LLM forward.
- No global request leaks; run single-image, batch1; track projector and selector cost separately.
- Smoke cases: POPE positive/negative and GQA, *same image different question*; query-shuffle must actually change embeddings.
- If raw query scores are flat, module must fall back to native.

**G1 / priority pilot (GQA & POPE):**
Preregistered initial thresholds (NOT expected outcomes):
- GQA >= **56.37** (+0.80) **OR** POPE F1 >= **84.21** (+2.00) vs L2.
- Other target must not decline materially; and must outperform matched random and shuffled question on the same ID list.
- Lower POPE FN must not be bought just by uncontrolled FP inflation / yes-rate drift.
- Use paired bootstrap clustered by image ID; report uncertainty.

**G2 / protect & efficiency:** only after G1, run MMB>=60.84 and TextVQA-OCR>=52.91; then remaining MME & SciQA.
- Report TTFT P50/P90, Fixed32, selector/projector milliseconds, peak VRAM under unified backend/precision/GPU.
- Favor overhead <=5% TTFT as engineering gate, not guaranteed physical law.
- If passed, apply **unchanged algorithm/hyperparameters** to W2048 E3: K172 before native E3 RDRIFT/schedule; ensure no core metric regression.

**Decision:** If S1 does not beat query-shuffle and matched random, **query relevance is not established**, even if mean F1 slightly moves. If no G1 pass or too much overhead, STOP this module and freeze native BELA; do not add QCP, dynamic budgets or a fresh Stage2 in the remaining time.

## 5. Four GPU execution (0 / 2 / 4 / 7)
Day 1: freeze production commit/checkpoint/ids/scorer and run CPU + one-GPU smoke. Off/no-op regression first; do not start 4 GPUs with an interface failure. Parallelize the same 4-arm pilot once valid, one GPU per dataset/arm as existing exclusive scheduler supports. Do **not** kill the ongoing C2. Verify exclusive `CUDA_VISIBLE_DEVICES` against actual physical numbering.
Day 2: merge GQA/POPE, POPE confusion and matched controls; decision by 2026-10-29. If gate passed, 4 remaining benchmarks and E3 within Oct 30–31. Freeze algorithm by Nov 1. Draft Introduction, Related Work, protocol and E3 baseline table immediately, independently of new candidate.

## 6. Deliverables Cursor must return
- `LIVE_CALLMAP.md`: actual entry point and real tensor shapes (not guessed)
- `QA_COVER_CODE_DIFF.patch`: production change only; off mode exact
- `QA_COVER_OFF_REGRESSION.json`, `QA_COVER_SMOKE.json`
- `run_meta.json` each arm with git/checkpoint/scorer/prompt hashes and actual per-layer K
- `predictions.jsonl` question/image IDs and original decoded text, official scorer output
- `entry_trace.jsonl`: native/final entry IDs, q std, swaps, match-shuffle mapping, post-L12/L20 survival
- `pope_confusion.json`: pooled/split F1, precision, recall, TP/FP/TN/FN, yes-rate, invalid counts
- `latency.json`: vision, full-projector, selector, LLM prefill, decode, TTFT, Fixed32, p50/p90, peak memory
- `FINAL_DECISION.md`: success/failure, negative controls, causal limits, explicit freeze decision.

## 7. Prior art and claims boundary
- AgilePruner ICLR 2026 code: erank + CLS-attention-dependent diversity; saliency and diversity are not uniformly beneficial. https://github.com/cvsp-lab/AgilePruner
- STAR-Pro (2026): coverage by pivoted QR before LLM, progressive text-to-vision refinement inside Decoder. https://github.com/EasonAI-5589/starpro
- CDPruner NeurIPS 2025: CLIP-conditioned DPP, text encoder and DPP cost. https://github.com/Theia-4869/CDPruner
- QCPruner (2026): question-conditioned bilateral facility coverage. https://arxiv.org/abs/2609.19990
- CoverPruner EMNLP 2026: query-weighted coverage over projected tokens plus lightweight first-layer attention probe. https://aclanthology.org/2026.emnlp-main.349/
- QuietPrune CVPR 2026: query-conditioned pruning *inside ViT*; this is NOT the same place as BELA Stage1 after full ViT. https://openaccess.thecvf.com/content/CVPR2026/html/Gao_QuietPrune_Query-Guided_Early_Token_Pruning_for_Vision-Language_Models_CVPR_2026_paper.html
- POPE readout audit Oct 2026: candidate token IDs and whitespace variants can shift reported F1 without changing model. https://arxiv.org/abs/2610.25964
- CVPR 2027: enrollment Nov 10, full manuscript Nov 16 AoE. https://cvpr.thecvf.com/Conferences/2027/CallForPapers

**Do NOT claim** "first query-conditioned pruning", "first coverage preservation", "Stage1 definitively caused POPE FN", "existing projector cosine necessarily grounded", or "CPU unit tests prove accuracy." The new method is only an evidence-based candidate; BELA's actual contribution requires matched-budget result, native ablations, honest latency, and clear differentiation from prior art.
