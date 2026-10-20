# 直接发送给 Cursor / 实现 Agent 的执行指令（2026-10-20）

你接手 CVPR27 ELVA 项目的**代码改造与实验**。先完整阅读本包 `FINAL_HANDOFF_CN.md`。真实工作仓必须是**昨晚跑出R3 V1–V5结果的服务器版本**，本包的E014旧快照/E020.3外部算子不是当前服务器仓，禁止直接覆盖真实代码。完成研究对象是W2048 R4尽量≥97、W1024 R4尽量≥94，压缩率分别88.9%和94.4%，并以不挑数据集的九项质量与同卡TTFT为独立证据。结果不能伪造。

## 第一轮必须做的真实代码修改

1. 查询真实仓当前git SHA、模型、CLIP层号、实际Stage1 `pen/L9/dual`、`COVER/PCOV`、原RDRIFT和原Q_ATTN函数，checkpoint、scorer。保存 `run_manifest.json`。缺真实RDRIFT则明确说明，禁止公式近似代替。
2. 将 `elva/core.py` 的Schedule/TokenLedger/trace验证绑定到当前decoder真实 forward。你需要写**实际可执行**的宿主代码补丁，不仅是新独立文件；每个block计算前记录token数，完成第b个block后原host selector + 原host物理gather/KV/mask/positions。验证物理token trace与数值W。
3. Stage1保持投影的pen hidden不变，V1–V5原选择函数不变；相同配置的剪枝IDs和逐题输出可复现（GPU数值细差必须注明）。不要把 `elva.combine_rank_scores` 当成原R3 dual；它只是新的独立诊断候选。
4. W2048先跑原 `V4 pen+COVER`+RDRIFT FIRST 和 `V5 dual+COVER`+RDRIFT FIRST，FINAL仍Q_ATTN，关merge。使用已有V4/V5同条件做对照。
5. W1024查原V4实际trace与PCOV60_B12的覆盖细节。接线 `E94_K80_48_16` 和 `E94_K80_64_16`，固定pen及PCOV B12后做小panel对照；严格 W1024。第2轮才补入口128或96组。
6. 开始记录每题原patch ID热区，输出热点图和question/image ID；评分只交官方脚本。原四任务R4和ScienceQA-IMG要分别报告，不得混平均。
7. 在两档做出强固定精度点后，选少量真实同prefix反事实运行，比较共同S+Ga与共同S+Gb，记录真实生成/loss与耗时；只有预测信号有效才开发在线路由。

## 本地先验收

```bash
python -m unittest discover -s tests -v
python -m elva.cli plans plans/schedules.json
python -m elva.cli official-r4 results_examples/official_scores.json
python -m elva.plot_patch_maps results_examples/actual_ids.example.json example_patchmaps.png
```

## 服务器真实验收（关键，不可省）

- 事件层号明确为完成block后的剪枝；32个block各留多少物理视觉token必须真实记录，W与预期完全一致。
- 物理hidden长度、patchID、RoPE、mask、KV cache和decode1/2对齐；无图和多crop不支持的情况显式失败。
- 对同图多问题真实SQA-IMG/POPE/MME按image分组统计；旧四任务官方scorer commit一致。
- 旧REF复现不对先排查，而非继续开发新公式。
- 同GPU同输入ABBA/BAAB计时，包含额外多层打分和COVER、gather开销，TTFT不因为W小就自动声称变快。

## 结果决定的后续收敛

- W2048 RDRIFT有效：在胜出者只补 `E88_ENTRANCE144`、`E88_WIDE_SHORT`、`E88_NARROW_LONG`，在开发集淘汰后最多两组全四任务确认。
- W1024 中段更长有效：优先保留 `E94_K80_48_16`；宽短有效保留 `E94_K80_64_16`；均无效回退原V4并补真实K60同入口日程检查，不乱调β。
- W1024/W2048均固定后，才对胜者补一个β候选、一个merge原α备选，不能多维同时更改。
- 未来效用不可预测：取消在线router，保存等预算真实反事实负结果；不要再训练更大头直到提供新证据。
- 11/04–11/06前冻结主方法，开始九集×两预算确认，11/10前完成注册，11/16前提交正文。

### 每天向研究 Agent 回传（必须）

```
(1) git SHA / config id / GPU / scorer
(2) 实际patchID、逐层token trace、W check
(3) 原四任务四项得分 + R4; 新任务单独分数
(4) 对照是否严格配对，控制了哪些因素
(5) 同卡TTFT原始日志（不能用跨卡均值）
(6) 至少三例真实保留区域热点图和正确/错误翻转
(7) 下一步仅一到两条，列明理由和停止条件
```

交付完成标准不是包里代码被import，而是**修改后的真实R3宿主forward和官方生成结果均可验证**。失败时将可复现的patch.diff和trace发回，禁止跳过或静默改预算。
