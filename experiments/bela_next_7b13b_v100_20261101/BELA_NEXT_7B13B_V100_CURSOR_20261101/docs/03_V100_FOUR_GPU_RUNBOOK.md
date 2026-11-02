# V100 4-GPU Practical Runbook — READ BEFORE STARTING

## A. First 30 minutes: host state and correctness, not benchmark
V100 host and repo absolute path must be read from actual local Cursor terminal,
NOT copied from historical A800 `/data1/xiangyue/...`.
Old conversations mentioned physical IDs 0/2/4/7 idle at an earlier time.
They are NOT guaranteed idle now.

```bash
set -euo pipefail
# Change this to where Cursor actually unzipped this package:
export PKG=/ABSOLUTE/PATH/TO/BELA_NEXT_7B13B_V100_CURSOR_20261101
export PYTHONPATH="$PKG/src:$PKG:${PYTHONPATH:-}"
python -m unittest discover -s "$PKG/tests" -v
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
# Replace with the four AVAILABLE physical GPU IDs (never assume):
python "$PKG/tools/v100_preflight.py" \
  --gpu-ids 0,2,4,7 \
  --config-7b /ACTUAL/NE_XT_7B/config.json \
  --config-13b /ACTUAL/NE_XT_13B/config.json \
  --output v100_inventory.json
python "$PKG/tools/plan_priority_queue.py"
python "$PKG/tools/print_arm_env.py" --arm NEXT7_T160_A0_PAPER
```

The final 2 Python commands **print configs only**. `BELA_NEXT_V2_*`
are NOT recognized by upstream LLaVA until integrated into the real code.
Do not call them "started experiments".

Gate checks:
- `torch.cuda.get_device_capability()` should equal `(7,0)` for V100.
- `torch_dtype=torch.float16`, no bf16, no flash-attn2.
- CUDA 11.x driver compatibility and `transformers`/`accelerate`/LLaVA versions
  as installed; avoid upgrade just to force modern FlashAttention2.
- Check `fp16` residual overflow/NaN and P0 no-prune output parity.
- Per-GPU memory must fit real model+CLIP+KV+selector; sample batch=1.
- Use `nvidia-smi --query-compute-apps` to confirm shared GPUs really available;
  don't kill unrelated users' processes.

## B. Choose actual model-parallel grouping according to memory

Approx. fp16 model weights alone:
- NeXT7B ~14GB (decimal) before CLIP/projector/KV/buffers,
- NeXT13B ~26GB before vision encoder and KV.

4 × V100 16 GB: safest initial probe
- 7B split across two GPUs e.g. first group two cards (IDs from preflight),
  another independent 7B job across the remaining two only if proven safe.
- 13B use four GPUs for first smoke if two card allocations OOM.
4 × V100 32 GB:
- 7B potentially one GPU but actual prefill memory must be profiled;
- 13B potentially two GPUs, but actual image contexts and cache determine fit.

**Important**: 4× distributed data-parallel replicas ≠ 4× parameter memory.
Use existing project `accelerate` model sharding / `device_map` with vetted
cross-GPU device transfers or real tensor parallel; `torchrun` alone does NOT
shard a 13B model. Cross-device model attention hooks must support pruning.
When CUDA out-of-memory:
- Save OOM stats, batch stays1, reduce unrelated processes via scheduling;
- Avoid hidden input image downsampling; changing resolution invalidates
  fixed five-crop protocol.
- Never silently select 4bit/8bit quantization to make a paper baseline fit;
  if absolutely needed report separately with fresh Dense baselines.

## C. Example safe config work (NOT an upstream LLaVA command)

```bash
export PKG=/ABSOLUTE/PATH/TO/BELA_NEXT_7B13B_V100_CURSOR_20261101
export PYTHONPATH="$PKG/src:$PKG:${PYTHONPATH:-}"
python "$PKG/tools/print_arm_env.py" --arm NEXT7_T160_A2_LATE
# Human Cursor must port these exact flags into active BELA
# source and use the EXISTING evaluation entrypoint with correct model.
```

For real inference in the local repo Cursor must:
1. Identify and print original working evaluation command (not invented).
2. Implement feature gate `BELA_NEXT_V2_ENABLE=1`, preserve `=0` original.
3. Adapt metadata / crop selects/physical gather; run smoke.
4. Feed `sample-id`, patch count, per-layer visual patch counts,
   position+cache actual equality assertions to manifest JSONL.
5. Validate with `verify_run_manifest.py --mode paper`; lock SHA.
6. Run dev panel with paired item-level results before any official full pass.

## D. Concurrency schedule on four GPUs

Rather than 20×10 full tasks, queue:
7B T160 →7B T320 →13B T160 →13B T320.
Separate model weights load once per consistent GPU group where possible.
Within each cell, run A0, A2, A3/A4 (with A1 as optional), matching same
dev panel IDs and same processor pinpoints.
For 7B group2 GPU parallel independently within 4 cards *if* actual free
VRAM supports. For 13B 4GPU model-parallel sequential evaluation may be necessary.
No need to run full official 10-task for all 20 arms.

After each cell chooses schedule, one-factor QA B-sweep then optional
global-thumbnail crop quota. If no improvements over A0 under dev,
keep A0 and retain accuracy before chasing speed.

## E. Exact efficiency measurement
- 10 warmups + >=50 measured fixed sample prompts, cuda.synchronize.
- p50/p90 per first-token latency (prefill+first decode), end-to-end
  wall including preprocessing/decode separately; high-res CLIP time not omitted.
- Need GPU peak allocated/reserved, CLIP/projector/QA costs,
  patch_W vs newline/text_W, actually profiler FLOPs (fused attention caveat),
  per-prompt actual image resolution and token counts.
- The optimum is not a numerical R6 alone. If two arms near within
  paired variance and latency/complexity gap, prefer simpler/cheaper one.
- Don't use answer gold/test labels in any Stage1 query score.
