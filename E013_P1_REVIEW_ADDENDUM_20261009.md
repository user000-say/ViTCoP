# E013-P1 Code Agent 独立复审补充执行卡（2026-10-09）
## 状态
**KEEP CURRENT RUNNING JOBS; PATCH ONLY VERIFIED AUDIT BUG; ADD SUPPLEMENTAL DIAGNOSTICS AFTER CURRENT PANEL.**
本文件仅补充已有 `CURSOR_E013_P1_HANDOFF.md`，不覆盖原始命令，不修改运行中的模型。
仅适用于 E013-P1 原版 + 明确记录的 KVFIX；一切分支记录 SHA、配置、起止时间。

## 紧急：KV audit 修复与不干扰运行
原版 `vitcop_modeling_llama_E013.py` 中 `removed_at_layer` 假定 K−172；对 EXTRA b>0 错误。已单独提供 `vitcop_modeling_llama_E013_KVFIX.py` 和 `.patch`。
- 已启动的准确率任务不要热替换源码或 live Python 进程。
- 原版 KV audit 的 EXTRA 结果只能标记 `AUDIT_INVALID`，不能判 SMOKE_PASS；已跑准确率保留原始结果，等待同源码匹配 KVFIX 的 GPU smoke。
- 下一次安全重启时先 SHA 对照并应用审计专用补丁，重跑真实 EXTRA0/16/32 prefill/decode1/decode2；确认 `172+b-96` 个位置剪掉。
- 严禁偷偷修 selector、Stage1 或 timing，修复前后评测分别归档。

## 必须保留并完成现有 headroom
`screen1` 8 arms、`screen2` 原控制项、REPLACE 和 EXTRA 同预算对照；每个样本 ID、图像 ID、随机 seed、源码版本保持一致。
`REPLACE_ATTN16` 并非简单 Top172：源码从原 Top172 以外取候选替换弱 16 个 Core token。明确写在论文表注中。
Random 依赖 request_index；不同 worker 样本顺序可能得到不同 patch。只有校验 sample_id→request_index 完全一致，才能声称严格配对；多随机种子需独立运行。
Panel 不能将子集 MME 原始分与全量 MME 总分混算。

## 桥接实验：改为必做，但不修改当前运行中的作业
冻结 4 arms，同一配对 panel：
- BASE_S8
- EXTRA_ATTN16_S8 / EXTRA_RANDOM16_S8（两个选择规则可择一为原实验主对照，建议都保留）
- BASE_S12
- EXTRA_ATTN16_S12 / EXTRA_RANDOM16_S12
`Δ8=metric(EXTRA_S8)-metric(BASE_S8)` 与 `Δ12` 分别在同一 S 内比较。不可用 EXTRA_S12 − BASE_S8 解释机制。
如果 P1 已经采集 BASE_S8 与 EXTRA_S8 且 sample IDs 完全匹配，可以复用，不必重跑。
桥接准确率建议先用逐题 correctness 和有可靠 scorer 的同面板聚合；必须冻结评估 panel。

## CROSSIMAGE 控制：新分支、另行审计、不得临时修改现有选择器
当前 `e013_stage1_support.py` 的 pick 集合 **不支持 CROSSIMAGE**；旧 `clip_encoder_E013.py` 的 E009 旧分支虽包含 CROSSIMAGE16，但 E013 要求 E009 关闭。不能只设环境变量冒充 E013 CrossImage。
准备独立小补丁前必须制定 deterministic donor mapping（同源集合、不同 image_id、无目标图泄漏），同 16 个投影前 CLIP patch，明确归一化/顺序/position IDs/插入位置与 REAL16 一致，配对 seed，核查 `QED eligible=172`、decode cache。
仅将通过上述验证的 CROSSIMAGE 结果用于「内容特异性」主张。当前没有 CROSSIMAGE 不妨碍 H1/H2 headroom 分析，但不能声称已隔离真实图像语义效用。

## 表示路径诊断：不阻塞正在运行的评估
已存在 E013 Stage1 原始 persistent/transient patch-ID 元数据，可通过 audit 输出。
对入围且准确率出现提升的 64–128 同图配对样本，离线保存：
- BASE vs EXTRA 的 K96 survivor IDs / Jaccard；
- 在 QED S 层，Core172 对应 hidden state 的规范化余弦差及 L2 shift（固定 sample/layer/position 映射，不同长度需先按 original patch IDs 对齐）；
- 再考虑 *单独的* fixed-survivor-ID 干预。这需要新模型分支，不允许 Cursor 自行修改并混入 E013 主 run。
未做 fixed-ID 干预时只能说 context influences predictions / selection，不能说已证明信息转移到 retained token。

## Utility 可预测性：区分当前可算和确实没记录的内容
`e013_stage1_support.py` 当前不输出 gold-answer logprob。现有生成式 accuracy runner 不一定有 teacher-forcing token log-prob；不要把正确/错误差当成 Δ log p，也不要要求 Cursor 伪造。
第一阶段从相同样本构造 `correct(BASE), correct(EXTRA), Δcorrect`，计算样本受益率、受损率和配对 bootstrap CI（多题同图按 image_id cluster bootstrap）。
源代码 E013_AUDIT 路径可输出部分 cheap Stage1 特征，请先核对真实日志中字段完整度；不足的特征用 **离线从已有 patch/score 缓存**计算，若没缓存则仅在独立 diagnostic rerun 保存；不往正式 latency path 添加 CPU/GPU 同步。
第二阶段另行安排 teacher-forcing 诊断获得 gold answer log-prob；需确保不同条件文本答案串与 tokenizer 完全一致，再定义真实同图 patch vs donor patch 的 `Δ log p`。
先冻结训练/校准与独立评估 image_id 划分；仅报告 held-out Spearman/AUROC，禁止在 full benchmark 上拟合门槛并报告同集性能。

## 延迟主张的硬门槛
E012 `LEGACY L12` vs `S12D12` Fixed32 相差约 90ms：同物理 GPU4、ABBA 交错、128 固定 panel、同输入样本/生成32 tokens、同计时边界、同 kernel/backend/cache/LLM code SHA；需先解释差异，不能混合历史 E010/E012/E013 snapshot 毫秒数宣称算法优势。
准确率 headroom 可以继续；latency headline 在此之前 HOLD。

## 汇报和门槛
交付：1) 现有 jobs 全量 raw stdout/stderr、status 和 sha; 2) 8/12 桥接配对结果; 3) 各 Stage1 arm 按预算配对胜负、对应 sample IDs、img IDs、随机 seed; 4) KVFIX 后真实 GPU cache smoke; 5) available cheap-feature vs Δcorrect 相关、缺失 logprob 的明确记录; 6) 哪一个 arm 仅 headroom 支持、哪个控制能支持更强因果说法。
未经对照验证，不准写「COVER 是新方法」「temporary evidence 被吸收」「adaptive utility 可预测」。
