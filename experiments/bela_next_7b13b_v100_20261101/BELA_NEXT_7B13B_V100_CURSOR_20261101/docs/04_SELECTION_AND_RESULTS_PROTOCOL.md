# Correct experimental comparison and decisions

## The paper headline and our scores can be misleading
- STAR-Pro paper NeXT7B 94.4% retains ~97.0% of
  its 10-task dense aggregate (not necessarily equivalent BELA 7B R6).
- MMTok up-to-160/320 refer to **final tokens before Decoder**;
  comparison at same W not established.
- TOPS 2026 appendix's published Stage I/II token counts and nominal target
  are arithmetically inconsistent under our strict `W=sum_layers K`.
  Do not allege a code bug. Instead run a numeric recheck or cite
  its table as different definition/unclear paper.
- The BELA 7B image 98.77 R6 at 77.8% (W4096) is a different checkpoint,
  task suite and token regime than NeXT7B T160/T320.
- All final metrics must include the matched in-run Dense using **same**
  processor image crop policy and tokenizer prompt template.
- Format each data row:
  model_sha,budget_T,arm,source_code_sha,processor_sha,actual_patch_N,
  avg_patch_W,avg_special_W,final_patch_K,task_name,split,n,
  raw_metric,dense_same_run,retention_pct,TTFT_mean,p50,p90,peak_memory

## DEV screening: pre-register independent panel, don't tune official tests
Suggested fixed panel: GQA,POPE,TextVQA,MME from non-final or allowed
calibration splits, 128–256 images/questions each where legally available.
Make a stratified OCR/position/diagram sample subgroup for each to identify
true AnyRes crop issues. Reuse exact IDs across all arms of same model cell.
Dev only. Freeze random seed, dataset versions and SHA of sample_id file.

Arms initial schedule competition:
```
7B / 160: A0, A2, A3, A4   [A1 optional]
7B / 320: A0, A2, A4       [A1 optional]
13B/ 160: A0, A2, A4       [A1 optional]
13B/ 320: A0, A2, A4       [A1 optional]
```
Default QA B8 and Q_ATTN_ALL fixed across these,
same Stage1 group quota=uniform.
Top 2 on dev + baseline retained for true paired error matrix,
not chosen by "relative R6" alone.

`tools/score_dev_matrix.py` accepts metric and same-Dense task values.
It averages per-task ratios, **not raw MME/acc values**, and rejects
test/final split names and mismatched Dense/task panels.
Use paired correctness from task's native scorer:
`tools/paired_dev_accuracy.py` with `dense_correct`, `baseline_correct`,
`candidate_correct`; don't make exact answer-string equality proxy
for question answers/normalized evaluation.

Don't declare winner on ~0.05pp; require consistent trends in OCR/POPE/GQA,
paired error counts and real time/memory. Report exact McNemar p
and, where possible, video/image-cluster bootstrap CI for task metrics.

## QA B after schedule and crop choice:
At T160 K0=320, try `B0/B8/B16`:
- B8=2.5% nominal candidate swapping, B16=5%.
At T320 K0=640, try `B0/B8/B24/B32`:
- B8=1.25%, B24=3.75%, B32=5%.
The actual donor count is bounded by available reject pool, QA cap
and real swaps. If B16/B32 cost > accuracy improvement,
favor B8 or B0 at that budget.
B values only meaningful after enforcing total vs per-crop semantics.

## Crop quota ablation:
- First all 5 groups same quota (320→64 each, 640→128 each);
- If middle/late OCR high-res evidence missing, separately test
  global-thumbnail group weight1.5 with same exact K0 and same W.
- Always retain ≥1/crop where feasible; count per crop.
- "thumbnail boost" may hurt fine detail; reject if TextVQA/AI2D fall
  despite gains in more holistic benchmarks.
- Only after verifying actual 5crop-index mapping, not from text suffix.

## Final actual benchmark
Prioritize matching STAR's *published tasks and scoring*;
at least GQA,ScienceQA-IMG,TextVQA,POPE,MME,MMBench EN/CN,
MMVet, plus user's earlier AI2D/MMMU/VizWiz for broader cross-model story.
Measure speed under 4×V100 separately from reported A800/H100 paper speed.
Performance retention is matched dense on EXACT model and dataset.
If available also report performance under more aggressive crop/AnyRes
native mode as *separate generalization*, not mixed official baseline.

## Stop rules
- fail P0 anyres / logits / qk / cache => no accuracy arms;
- any wrong W => no claim of 94.4/88.9;
- 13B OOM => return config & memory traces, don't switch to 8bit silently;
- no correct QA dual mapping => B0 + baseline first then repair;
- all arms roughly same => freeze simpler A0, don't brute-force 100 weights.
