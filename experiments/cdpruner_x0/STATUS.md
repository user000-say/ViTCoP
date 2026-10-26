# CDPruner X0 — official source on our evaluator

- Source: https://github.com/Theia-4869/CDPruner → `third_party/CDPruner`
- Env: `/home/xiangyue/envs/starpro` (torch 2.1.2 / transformers 4.37.2 matches CDPruner `pyproject.toml`)
- Adapter: `llava_cdpruner_lmms.py` → lmms-eval (same 6 tasks / Dense refs as BELA)
- Retain K: **64** and **32** (`visual_token_num`; fixed for whole decoder — not BELA W=ΣK·L)
- Checkpoint: LLaVA-1.5-7B snapshot (same as STAR/BELA)

## Note vs BELA W1024/W2048
CDPruner K=32 keeps ~32 tokens every layer (W≈32×32=1024 only if counted that way, but pruning is **single early keep**, not progressive mid/late). Compare as **final-K baseline**, not iso-schedule to L2/E3.
