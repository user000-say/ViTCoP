# Cursor 执行 Handoff：BELA NeXT-7B P0 修复部署与实证验收

## 本轮唯一目标

把附件中的 `BELA_NEXT7B_SUBMISSION_FIX_V3_20261102` **作为完整独立新包**放到服务器的一个**新**目录，严格按本文件执行离线审查、CPU 单测、模型源码锁定、原生视觉对齐、QA/Decoder 数值 oracle、真实 GPU 小样本、R4/mean6 评分。**不要用现有运行目录上的临时增量改写替代。不要覆盖正在跑的 s2qk A0，尤其不要 kill/hold GPU4–7 当前实验。** 没有可用 GPU 时只交付离线结果、标记 GPU BLOCKED。

### P0 — 代码部署（不自动启动任何实验）

1. 将整个包解压到 `.../bela_next_submission_fix_v3/`。包中 **`bela_next_forward` 等已更名为独立模块**，不要把它们强行并到旧 `next_forward`。原始 `next_forward/` 仅在参考目录。禁止修改 site-packages 中的 HF 文件。
2. 检查 `python --version`、`python -c 'import transformers,torch;print(transformers.__version__,torch.__version__)'`、`nvidia-smi`、checkpoint SHA/路径、`lmms_eval` git commit、模型 processor 配置。确认目标 transformers **5.20.2**，以及 `reference/original_runtime_tf520/IMPORT_PATHS.json` 中的 `modeling_llama.py` 和 `modeling_llava_next.py` 哈希完全匹配。若不匹配，停止，收集真实源码 diff，不能设置 `BELA_REQUIRE_RUNTIME_HASH=0` 来硬跑。
3. 同源 tokenizer 由 `next_scripts/lmms_bela_next_entry.py` 绑定至 `self._model.model._tokenizer`。若 lmms-eval 版本没有提供 `_tokenizer`，先找真实 tokenizer 持有者，在新分支适配并写测试，禁止自动下载不同 tokenizer。
4. 在新包根目录执行：`python -m pytest -q tests`、`python -m compileall -q bela_next_forward bela_next_budget bela_next_fix bela_query_anchor`、`bash -n next_scripts/run_one_task.sh`、`python tools/preflight.py --runtime --arm NEXT7_T160_A0_PAPER`。保存全部日志和 exit code。

### P1 — 原生 Dense 对齐先于剪枝

5. **Dense parity**：全新 checkpoint 同环境，比较未安装 patch 的原生 HF 与安装 patch 但 `BELA_NEXT_V2_ENABLE=0` 的输出；至少单图多比例、同文本、同 dtype、same processor/crops/chat template/seed、首 token logits max abs/rel diff 与贪心序列；数值不一致一律 FAIL。
6. **AnyRes oracle**：在新 GPU smoke 上打开 `BELA_NATIVE_DENSE_ORACLE=1`，逐个比较 ViT→mm_projector→原生 `pack_image_features` 的完整 packed 序列与我们的未剪枝 packed；核对 `image_sizes` H/W、grid_pinpoints、required_crop_count、pad crop trimming、base/tile 的 original UID、本地 24×24 index、unpad 输出的 `(row,col)`、image_newline、processor image placeholders。覆盖横向、纵向、方形、窄图、需要 padding 的图像；任何 UID 对不上即 FAIL（包括无越界却指错 patch）。
7. **CLS oracle**：设置 `BELA_STAGE1_SALIENCY=cls_attention`（默认），对 1–2 样本设置 `BELA_VIT_CLS_ORACLE=1`；若 SDPA 不回传 attentions，在单独的 oracle 进程临时启用 eager 视觉后端（不得把 oracle 修改带入真实 ACC），与真实 ViT penultimate CLS→patch softmax 对比。并核对 `mid_layer=9` 对应历史 1.5 实际 capture_outputs 语义，而非凭层编号猜。

### P2 — Stage1 COVER、QA、真实索引审计

8. 打开 `BELA_TRACE_FULL_UID=1`（仅小样本）；逐 Crop 提供有效容量、K0 quota、全局 32 slots 的实际分配及每 Crop 原始 local patch ID。`sum(slots_per_crop)` 必须不超过 32 且 K0=320；不准给每 Crop 32 slots。运行对比 `e013_choose(REPLACE,COVER)` 的单 Crop 576 Patch 严格选点单测。诊断 `norm_proxy` 只允许显式消融，默认必须真实 CLS。
9. 每样本交出：`question_n_tokens`、原始 USER/ASSISTANT 边界、`qa_swaps_cfg`、`qa_swaps_reported`、`qa_reason`、`dual_ids`、`qa_ids`、donor/drop 记录、qa 前后 Crop counts。`qa_swaps_cfg=8` 是上限而不是保证每样本 8 次，QA 若长期为 0，标记 FAIL 并排查模板及 cosine 信号。
10. Stage1 真正保留的是 320 枚 Patch **加**合法 image_newline；必须把 patch 数与 packed 总长度分别统计。验证原始 token_layer W 的计数口径只含视觉 Patch，不要把 newline 混入预算。

### P3 — Stage2 Decoder 严格验收

11. Layer12/20 使用 Layer-**input** norm/Q/K/RoPE/真实 GQA，并从**有效文本 Query**对有效视觉 Keys 做 causal+padding softmax；检查真实 attention_mask、Query 边界、无文本或有 padding 必须拒绝非法状态。用 eager 注意力/可控输入做 Q_ATTN 分数 oracle，不能只看 `torch.isfinite`。
12. 每次物理剪枝前后记录完整 `keep_abs`、原始 Patch UID 集合、`position_ids`、`text_start_before/after`、`cache` 每层 K/V 维度和顺序；assert 严格嵌套、无 resurrect、无重复、无越界、每个保留 patch 的 UID 不变。特别用 **尾随 image_newline** 的输入验证后续 Q_ATTN 不把 newline 当文本 Query。
13. 对包含 CLS/AnyRes 裁剪的真实输入检查 Layer12: 320→91，Layer20: 91→46；验证 K/V cache 的所有已填充层同样按照 keep_abs gather；decode 1/2/8/32 个 token 的 logits finite，KV 长度、RoPE 绝对位置连续增长，比较 Dense 和 unpruned no-cut 在相同调用栈下的首 token 行为。
14. **生成协议**：`generate` 只承诺 batch1、贪心、单束；如 lmms-eval 任务使用 beams/sampling/额外 processors/未知 stop-string 规则而被拒绝，不得私自吞参数；记录不兼容项并用 test + code diff 修复，确保 stop criteria 被执行、EOS/max_new_tokens 统一、OOM 直接失败、不生成伪答案。

### P4 — 公平精度验收和投稿标准

15. 在上面全部 PASS 后，才单独挑一张**空闲** GPU：

```bash
export BELA_PYTHON=/path/to/python
export LMMS_EVAL_ROOT=/path/to/lmms-eval
export BELA_LMMS_LIMIT=4
export BELA_NATIVE_DENSE_ORACLE=1
export BELA_VIT_CLS_ORACLE=0
export BELA_TRACE_FULL_UID=1
CUDA_VISIBLE_DEVICES=<verified-free-gpu> bash next_scripts/run_one_task.sh NEXT7_T160_A0_PAPER mme /new/path/SMOKE_V3
```

16. 再跑 4 个 R4 任务（GQA, POPE, MME, MMBench），**每任务使用独立新目录、禁混重试**，从相同 checkpoint/dtype/processor/lmms-eval scorer 得到完整 absolute/relative；运行 gate：`python next_scripts/gate_next7_arm_r4.py --arm NEXT7_T160_A0_PAPER --out-root /new/path/R4 --single-floor 88 --r4-min 92`。门槛 88/92 是用户历史早停策略，非投稿充分条件。若跳闸停掉**自己新启动**的任务，不触碰原队列。
17. 若 R4 合格，再跑 ScienceQA-IMG 和 TextVQA，报告 NeXT Dense normalized **mean6**，每任务 absolute、relative、95% bootstrap CI，输出 3 次独立 seed/rep 的均值和标准差（若方法确定性强仍记录环境变动）。对齐今年 STAR-Pro、ST-Pruner、QCPruner 等同 NeXT checkpoint、5 Crop、相同名义/实测视觉预算、图像预处理、任务 split 和 scorer，不得横向硬比不同协议的百分比。
18. 投稿实验必须有：Stage1 CLS-only vs Dual COVER vs Dual COVER+QA，Stage2 no-cut vs Q_ATTN，global slots 0/8/16/32/64、Crop equal/thumbnail boost、QA B0/B4/B8、两次剪枝形状消融；每组严格 W5120，不得混 W1024 或 W2048。另报真实 prefill 延迟、decode tokens/s、端到端延迟、峰值显存、额外选择开销和 token-layer 口径。

### 回传（缺一不可，不能口头“已修复”）

- 新分支 commit、全体修改源码、`git diff --binary`、模型和 runtime 哈希、`preflight.json`、环境版本。
- `pytest`/compile/bash 测试日志和报告；GPU oracle 的输入类型（不需要原图隐私数据）、原生 pack 数量/UID digest、CLS 数值误差。
- 10 个不同高宽比样本的完整 `summarize_trace()` JSON（含 `full_keep_abs`、`full_patch_uids`、每次剪枝 Cache 长度），所有失败样本栈追踪。
- Dense parity 对照、生成 stop 规则、当前真实任务日志和 scorer JSON。
- 发现错误必须写 `FAIL + file:line + patch + new test`，不可写未附证据的 PASS。

**停机条款**：模型源码 hash 不符、原生 pack 不对、QA 签名跑不通、任何 index/UID 错、score/缓存数学 oracle 不过、Dense parity 不过：停止 ACC 调参，先修工程。此 Handoff 不构成用户允许占用、清空或终止现有 GPU 作业的授权。
