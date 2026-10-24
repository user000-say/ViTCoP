# BELA / ELVA CVPR 2027 — Cursor Handoff V2 (2026-10-24)

## 研究目标（优先级固定）
绝对分：GQA、POPE F1、MME Total、MMBench 必须有竞争力；ScienceQA-IMG / TextVQA 退化不可掩盖；VizWiz 暂不参与筛选。
Stage1与Stage2需要能经受交叉消融；不能仅凭组合模块写故事。
效率必须报告实际 TTFT、Fixed32、selector overhead、峰值显存，而非只给 W。
任何建议，包括用户、审计 Agent 与本 Handoff，都须由源码及控制实验检验。
CVPR 2027 论文与实验并行，最终以可投稿、可复核的强架构为目标。

## 已报告结果（并非此次代码包重新跑出的GPU结果）
A0 172/90/32 W2048: GQA57.84 POPE83.31 MME1767.8 MMB62.20 SciQA68.96 TextVQA55.97 TTFT123.7 Fixed32=1234.6
A2 172/120/20 W2048: GQA57.62 POPE85.44 MME1774.9 MMB63.14 SciQA68.67 TextVQA54.89 TTFT122.5 Fixed32=1217.7
A3 172/130/16 W2048: GQA57.35 POPE85.68 MME1775.9 MMB62.80 SciQA68.67 TextVQA54.88 TTFT120.3 Fixed32=1177.1
B1 172/90/20 W1808: GQA NOT_RUN POPE83.45 MME1747.8 MMB62.37 SciQA68.86 TextVQA54.51；时延待测
B2 172/120/32 W2288: GQA58.44 POPE85.36 MME1804.2 MMB62.80 SciQA68.37 TextVQA55.13；时延待测

## 当前真正的技术判断
B1相对A0只减少deep K，POPE几乎不变；A2大幅提升POPE更可能与中层K增大及保留集合身份改变相关，而不能声称深层小K本身有益。
B2相对A2只增加deep K：GQA+0.82 / MME+29.3 / TextVQA+0.24，POPE-0.08。它支持深层视觉证据对某些任务仍有价值，但B2预算增加240，不能当同成本SOTA。SciQA相对A2继续下降。
QCPruner的最终K32不是层平均K32；论文中前8层576 tokens，名义W5376，因此不要和W2048相提并论。MME Total与Perception不能混用。

## P0（立即执行：阻断条件）
1. 补跑 B1 GQA 和 B1/B2 TTFT/Fixed32；同GPU和计时方式，保留原始样本、重复次数与p50/p90。
2. 锁定git SHA、checkpoint、数据划分、prompt、评测程序哈希、Dense参考、predictions.jsonl。
3. 读取现役源码，明确第4层和第12层剪枝分别是什么selector（RDRIFT还是Q-Attn），并列调用位置、score来源及采样时点。原Patch ID、gather、position、KV cache及mask核对。不可凭历史推断。
4. POPE random/popular/adversarial分别输出TP/FP/TN/FN；GQA按题型分组，TextVQA按OCR失败类型进行paired分析。

## P1（优先：机制确认）
完成A0/A2/B1/B2 四角实验；B1 GQA补齐后再计算两因素交互。不同W只作因果诊断，不做公平效率排序。
固定同一sample_id，比对 A2/B2 GQA恢复题、TextVQA恢复题，以及POPE新误报/漏报；导出剪枝事件原始Patch ID。
若B2改进只来自最终多12个token，则用固定最终K20、相同W2048做Token身份消融：现役选择器、合法Q-Attn、合法RDRIFT、Random多种子，必须先确认实际selector代码。

## P2（同预算主线）
目标：在W2048保留A2的POPE优势，并尽量恢复B2的GQA和TextVQA。
不批准直接加入Evidence Guard、ICER变体、CoRePrune扰动或复杂动态控制器。
一个低成本诊断候选 172/125/18 (=2048) 可作为预算分配中间点，但不得把单纯调比例当论文贡献；更优先依据真实错题与Patch trace决定是晚剪、延长部分关键token生存、还是改善最终K20的身份选择。
所有改动默认OFF，提交小diff和3个smoke再运行大评测；效率与ACC并行。

## P3（三档预算推广，不盲目套比例）
W1024：首先复现历史 `60×10+28×10+12×12`，再审计 Stage1 可否提供96入口，测试 `96×4+50×8+12×20` 作为候选同W日程。优先GQA/POPE/TextVQA。
W2048：以A2作为冻结基线；B2只用于解释后段价值。
W4096：只有Stage1入口K256可用且Patch映射正确才测试 `256×4+184×8+80×20`；验证Stage2是否有独立收益及实际延迟，不靠提高预算来宣称算法新颖。
用户前述“4028”与“2048”有歧义，本次主目标按W1024/W2048/W4096执行；若实际需要W4028，必须作为独立预算重新精确校验。

## P4 与强论文、论文成稿
STAR-Pro官方实现作为优先同机对照；QCPruner/CoRePrune必须对齐预算、模型和评测协议。W不同的表只能作为背景参考，不能声称公平赢。
必须保留错误案例与失败实验。论文不能写“深层多余Token有害”除非固定中层容量等干预确认删除它们改善答案。
交付: BASELINE_LOCK.md, SELECTOR_CALL_MAP.md, PATCH_TRACE.jsonl, POPE_DIAG.json, PAIRED_DIAG.json, LATENCY_RAW.csv, RESULTS.csv, SOURCE_DIFF.patch, STOP_GO.md。
