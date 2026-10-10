# Cursor E013-P1 — 半天至一天的高信息量实验执行卡

**发起日期:** 2026-10-09。**角色:** Cursor 只做核查、备份/复制已写好的完整源码、运行已有 runner、留存日志、汇总结果。**禁止** Cursor 自行开发新 selector、乱改缓存、修改 QED 公式或测时脚本。新算法的数学定义、实现、调度脚本、CPU 单测均在包内。真实 GPU 验证和实验结果尚未产生。

## A. 本次科学目的，不可改动

Stage2 冻结：**LLaVA-1.5-7B / FP16 / SDPA / batch1 / E012 S=D=8 / QED96**。Stage1 仅比较两类预注册实验：

* **Candidate Repair (REPLACE):** 总视觉 token 始终 172；Core Top156 加上原 Top172 之外的16个 patch，比较 RANDOM / ATTN / COVER。新补的16个允许成为最终 QED96。
* **Transient (EXTRA):** 冻结原 Top172，不改变最终 K96；临时加入 16 个随机/ATTN/COVER/NULL/DUPLICATE token，它们在真实 decoder L0..8 参与计算，必须禁止进入最终96。另测 b=0/8/32；L12 仅在需要复核 E008/E009 时使用桥接实验。

本次不做 QED-RS、训练回归器、动态预算模型或 full 96-job sweep。不要把临时 COVER heuristic 写成已验证的原创算法。它用于判断余下值得做什么创新。

## B. Step 0：必须在原服务器逐项核实（约 15–30 min）

1. `cd <真实ViTCoP服务器代码仓库根目录>`。
2. `pwd; git rev-parse HEAD; git status --short; nvidia-smi -L; python -c "import torch;print(torch.__version__,torch.cuda.is_available())"`。
3. 上传并解压本包到仓库以外的目录，如 `logs/e013_p1/E013_P1_Package`（不要在 Python 包路径下解压一堆临时脚本）。
4. `bash logs/e013_p1/E013_P1_Package/e013_install.sh --dry-run`。
5. EXPECTED source SHA256 before installation:
   * current Stage1 `clip_encoder.py`: `4674dfac5e3fb824f3ecd3151f7864487b83b96c4891501b0e0378b24286608e`
   * current E012 decoder `vitcop_modeling_llama.py`: `614b86c9f7ca7564e12669fec0776730c019fa7422c2c06f21ac12e0c869a3b1`
6. 如果 SHA 不一致，**立即停止覆盖**，打包实际服务器源码、git diff 和 E012 代码路径返给 Code Agent。当前包以此前给你的 E012 代码包为基线，不能猜测最新服务器代码。
7. 确认服务器 E012 成功运行所需 `llava.model.e010_runtime_support`、模型权重、lmms-eval、图片/标注和实际 runner 均存在。**不要重写它们。**

## C. Step 1：装载完整代码（约 10–20 min）

```bash
PKG="$(realpath logs/e013_p1/E013_P1_Package)"
bash "$PKG/e013_install.sh" --apply
python -m py_compile \
 LLaVA/llava/model/multimodal_encoder/clip_encoder.py \
 LLaVA/llava/model/multimodal_encoder/e013_stage1_support.py \
 LLaVA/llava/model/language_model/vitcop_modeling_llama.py
python "$PKG/test_e013_cpu.py"
python "$PKG/e013_launch.py" --phase screen1 \
 --cmd 'python YOUR_VERIFIED_EXISTING_EVAL_RUNNER --output_path {out_dir}'
```

最后一条为 **dry-run plan**，不可当作真实执行。记录 installer 输出的备份路径；所有新改动在 `clip_encoder_E013.py`、`vitcop_modeling_llama_E013.py` 与独立 helper 中以 `E013-P1 2026-10-09` 标注。不要用原 E010 decoder 覆盖已验证 E012 decoder。

## D. Step 2：填入真正跑通过的命令，只允许编辑四个字符串

打开 `$PKG/E013_COMMANDS_EDIT_ME.sh`，把四个 `__PASTE_VERIFIED_...__` 替换为服务器**真实已经在 E012 成功运行的** smoke、512样本准确率 panel、全量四数据集、128-panel Fixed32 runner 调用参数。其余行不动。

必须保留原命令的：权重路径、dataset config、图片尺寸、题干/answer tokenizer、explicit question-span adapter、batch=1、precision、max_new_tokens/Fixed32、采样、随机种子。让所有 arms 使用**完全相同的 sample_id 清单与遍历顺序**。

四条命令必须能把自己的结果写到 `{out_dir}` 对应目录（使用原 runner 支持的 `--output_path` / `--output_dir` 等真实参数；**不要照抄示例参数去发明不存在的 CLI**）。如原 runner 不支持唯一输出路径：STOP，提供真实 help/log，不可并行运行可能互相覆盖输出文件的任务。

```bash
source "$PKG/E013_COMMANDS_EDIT_ME.sh"
[[ "$E013_SMOKE_CMD" != *'__PASTE_'* ]] || { echo 'unconfigured smoke command'; exit 2; }
[[ "$E013_PANEL_CMD" != *'__PASTE_'* ]] || { echo 'unconfigured panel command'; exit 2; }
[[ "$E013_LATENCY_CMD" != *'__PASTE_'* ]] || { echo 'unconfigured latency command'; exit 2; }
```

注意：新程序 `e013_launch.py` 直接通过 `subprocess.run(argv)` 运行已有评估器，不修改模型评测实现；使用 `CUDA_VISIBLE_DEVICES=<physical gpu index>`，各进程内部设备编号可为 cuda:0。

## E. Step 3：必须 PASS 的真实 GPU smoke（GPU4，约 30–60 min）

```bash
python "$PKG/e013_launch.py" --phase smoke --gpus 4 \
 --cmd "$E013_SMOKE_CMD" --audit --execute --out logs/e013_p1
```

`smoke` 内含 7 组：BASE_S8、B0_S8、REPLACE_COVER16、EXTRA_ATTN16、EXTRA_COVER16、EXTRA_NULL16、EXTRA_DUPLICATE16。每组至少 **2 条真实 image+question，每题生成≥3新 tokens**。至少人工核查每组第1个 sample 的：

- BASE_S8 与 B0_S8：Stage1 patch IDs / prefill QED selected IDs / 最终输出严格一致。
- REPLACE：视觉输入 172，原始 Core Top172 中保留 Top156 并用外部16 patch 替换；**最终 K96 可以从新172选**。
- EXTRA16：视觉输入 188，前172个 patch IDs 必须与 BASE 相同；末16个 label=-3；QED eligible=172；实际最后保留96个且全部来源于 Core172。
- 真正的 prefill 与 decode1/decode2；QED 一次 prefill 选择、0 次 decode 选择；层 `≤8` 具有长视觉 cache，层 `≥9` 使用精简 cache（相对原 input **减少 172+b−96** 个视觉 positions；不是缓存只剩76个 token）。
- `E013_AUDIT/E013_STRICT/E012_DIAG/E012_KV_AUDIT` 在正式准确率和计时必须关闭。

任何形状、mask、SDPA fallback、缓存、选中ID或输出错误：**HARD STOP**。保留原始 traceback，不以“负结果”继续推进。回滚：`bash "$PKG/e013_rollback.sh" "$(cat logs/e013_p1/ACTIVE_BACKUP_PATH.txt)"`。

## F. Step 4：8GPU panel 第一波（各任务必须固定 512 样本，约 1–3h 依现有 runner）

```bash
python "$PKG/e013_launch.py" --phase screen1 --gpus 0,1,2,3,4,5,6,7 \
 --cmd "$E013_PANEL_CMD" --execute --out logs/e013_p1
```

所有 GPU 同时使用，但每卡任一时刻只有一个 VLM 任务。8 arms：

| GPU | arm | Stage1 | 目的 |
|---|---|---|---|
| 0 | BASE_S8 | 原 CLS Top172 | source-replaced baseline |
| 1 | B0_S8 | 开 E013 但 b=0 | identity/overhead anchor |
| 2 | REPLACE_RANDOM16 | Top156+random16 | 固定预算负对照 |
| 3 | REPLACE_ATTN16 | Top156+CLS next16 | 重要性对照 |
| 4 | REPLACE_COVER16 | Top156+Core-aware coverage16 | 固定预算互补性 |
| 5 | EXTRA_RANDOM16 | Core172+random16（final不存活） | 暂态真实上下文负对照 |
| 6 | EXTRA_ATTN16 | Core172+CLS next16（final不存活） | 重要性补充 |
| 7 | EXTRA_COVER16 | Core172+coverage16（final不存活） | 暂态互补性 |

建议 512 是固定的多任务 panel（例如每项128），样本具体 ID 在**运行前冻结**；无须对 MME/POPE 的128条 panel计算正式 F1/MME 总分，先使用个例胜负、gold log-prob（若原 runner已支持）、incorrect->correct 等诊断。

## G. Step 5：8GPU 第二波（仅 smoke 与 panel1 干净时，约 1–3h）

```bash
python "$PKG/e013_launch.py" --phase screen2 --gpus 0,1,2,3,4,5,6,7 \
 --cmd "$E013_PANEL_CMD" --execute --out logs/e013_p1
```

8 arms：EXTRA_NULL16、EXTRA_DUPLICATE16、EXTRA_ATTN8/32、EXTRA_RANDOM8/32、EXTRA_COVER8/32。比较**同 budget** 的 RANDOM/ATTN/COVER 与 NULL/DUPLICATE；记录 b=0/8/16/32 的每样本 correctness 和实际 token-layer work。

如果 E008 在 L8 无法复现，不要马上宣称机制被否定；有限追加一次 L12 桥接（不改变主研究锚点）：

```bash
python "$PKG/e013_launch.py" --phase bridge12 --gpus 0,1,2,3 \
 --cmd "$E013_PANEL_CMD" --execute --out logs/e013_p1
```

包含 BASE_S12、EXTRA_RANDOM16_S12、EXTRA_ATTN16_S12、EXTRA_COVER16_S12。明确 L12 只是检查 E008/E009 的深度依赖，不与 L8 结果直接混算。

## H. Step 6：full accuracy 只运行值得晋级的 1–2 个 arms

从 panel 证据按预注册规则选最优 REPLACE 和/或 EXTRA 候选，必须同时包含 `BASE_S8` 与 budget-matched 的强对照。**不要 16 arms 全量跑四任务**，半天至一天目标达不到时就保留 screen 原始结果如实汇报。

```bash
# 示例。只有上述 panel 真正通过后才能更改 arms 名单执行。
python "$PKG/e013_launch.py" --phase full --gpus 0,1,2,3,4,5,6,7 \
 --arms BASE_S8 REPLACE_ATTN16 REPLACE_COVER16 EXTRA_RANDOM16 EXTRA_ATTN16 EXTRA_COVER16 \
 --cmd "$E013_FULL_CMD" --execute --out logs/e013_p1
```

全量必须报告四个官方 scorer：GQA N12578、POPE F1/acc N9000、MME perception+cognition N2374、MMBench N4329；报告 normalized macro 与**分别每项分数**，绝不拿简单平均正确率代替 POPE F1/MME。共享的 baseline 应在同一代码快照复现，否则只作为历史参考。

## I. Step 7：GPU4 延迟交错对照（其他7张卡若仍满载需先结束/避开）

### I-1. 先判断 E012 表中94ms的可疑差异

```bash
python "$PKG/e013_launch.py" --phase crossover --gpus 4 \
 --cmd "$E013_LATENCY_CMD" --execute --out logs/e013_p1
```

严格顺序 LEGACY L12 → E012 S12D12 → E012 S12D12 → LEGACY L12，128 **完全相同**样本，GPU4，20 warmup不计，5 repeats，同一输入边界、固定实际生成32 tokens、原生 CUDA synchronize；debug全部OFF。每 run 必须保留 `sample_id/repeat/ttft_ms/fixed32_ms/generated_tokens/gpu_uuid/input_policy` JSONL；4个 JSONL 用下面校验器：

```bash
python "$PKG/e013_latency_crossover_gate.py" \
 --legacy-a <A.jsonl> --e012-a <B.jsonl> \
 --e012-b <C.jsonl> --legacy-b <D.jsonl>
```

如果 95% CI 不在预设 equivalence band 内，**不可把 E012 的94ms或 E013 的速度收益归因于算法**；转回原 `e012_latency.py`/`e010_latency.py` 分析计时/缓存/SDPA/生成长度，保留准确率结果与 logs。

### I-2. 仅在 I-1 通过后做 Stage1 真实 break-even

```bash
python "$PKG/e013_launch.py" --phase latency --gpus 4 \
 --arms BASE_S8 REPLACE_ATTN16 REPLACE_COVER16 EXTRA_ATTN16 EXTRA_COVER16 \
 --cmd "$E013_LATENCY_CMD" --execute --out logs/e013_p1
```

不要把 `EXTRA_RANDOM` 的 RNG 时间作为新算法 Pareto 主方法；它是准确率负对照。使用包中 `e012_latency_summary.py` 配对分析任意二臂；记录真实 selector-only、vision/projection、prefill、decode 及 Fixed32 时间（如原 profiler已支持；否则标为未测，**不许补造数字**）。

## J. 科研晋级规则 / 禁止的错误结论

**H1 成功的证据：**同为172输入 token，REPLACE_COVER16/其他原创候选稳定好于 BASE 和 REPLACE_RANDOM16/ATTN16，同时附上 selector 本身 operator cost；最终论文可研究什么 early evidence 值得被替换。

**H2 成功的证据：**同为172+b早期预算、final K96 固定，EXTRA_COVER16 稳定优于 EXTRA_RANDOM16/EXTRA_ATTN16，并且改善不会被 NULL/DUPLICATE解释。再观察样本级预算需求是否不同、是否可由廉价 CLIP 信号预测。仅此之后才进入 Counterfactual Processing Utility predictor 的 E014 实现。

如所有真实 Extra 都好于 NULL 但 COVER≈RANDOM/ATTN，则只支持“真实短暂视觉上下文有价值”，**不支持新多样性模块**。如果 REPLACE 质量不变且慢，否决它。若 b32比b16好但代价更高，记为 trade-off 而非速度突破。统计 CI、独立随机种子和未用于调参的任务非常必要；理想的正式 full macro 增益目标约 +0.25–0.30pp（非数学硬下界）。

论文拟研究 **“Which visual tokens need to be processed, versus retained?”**。Algorithm v2 可能是 cost-aware counterfactual processing utility 的预算/候选决策，但 E013-P1 只是首次证明有 headroom，不能抢先宣称已发现该算法。

## K. 最终 Return 格式（不可省略）

`E013_RETURN.md` 必须至少包含以下结构：

```text
1) Source: git SHA, original code SHA, E013 three source SHA, 8 GPU index+UUID,
   torch/transformers/CUDA versions, exact runner argv and input_policy.
2) Install & smoke: B0 vs BASE selected IDs/output; REPLACE/EXTRA patch IDs;
   prompt spans, QED=1 prefill/0 decode, KV topology, 2+ decode steps,
   FAIL/PASS with raw log file paths.
3) Panel: exact paired image/question IDs & per-arm score; all failures;
   b=0/8/16/32; each seeded random policy; table of wins/losses.
4) Full official metrics and sample counts, CI by matched examples/image clusters.
5) ABBA crossover: 4 JSONL, CI, timing-equivalence verdict.
6) Stage1 latency (only after crossover): TTFT/Fixed32 mean med P90 CI;
   GPU UUID, warmups, repeats, 32 tokens; overall Pareto.
7) Decision: H1/H2 accept/reject/inconclusive and reason;
   no claim of algorithmic novelty from a nonvalidated heuristic.
8) Rollback status, source diffs, return archive path, next minimal patch request.
```

**Hard failures:** source mismatch, missing runner, broken smoke, metadata/candidate leak, cache shape invalid, SDPA fallback, unrecoverable OOM or benchmark IDs mismatch. **Soft scientific negatives:** no accuracy gain, cover inferior to random, budget not predictable; log and continue within budget, do not secretly change formula.

核心科学目标：先找到“哪个可被删除的视觉 patch 值得花算力处理”的可预测证据，再决定如何把它变成算法模块。这是我们 E008–E012 累积结果和论文切入点之间最有价值的连接。
