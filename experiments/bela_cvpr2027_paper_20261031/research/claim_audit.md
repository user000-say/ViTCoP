# 主张、数字与引用核查

核查日期：2026-10-31。正文数字来自仓库里的 `metrics.json`，不是本轮新跑的 GPU 评测。

## 代码身份

- 问题条件更新的 GQA/POPE 运行记录提交为 `6b011ea5f3dbdee54a25c8df3a5e4d0736d8040d`，配置 `experiments/bela_query_anchor/configs/S1_QANCHOR_L2.json`，`qa_mode=s1_anchor`，`max_swaps=8`。
- 该提交到当前 `0204d8e` 的差异不改变 K=64 的 COVER 公式、`query_anchor.py` 的交换式或 `e020_lib/dispatch.py` 的 rank_drift。后来增加的多轮交换上限和替代初始化默认关闭。
- 原生六任务里，GQA 与 POPE 的 `L2_NATIVE` 与 `L2_LATE_64_17_10` 分数逐位相同。MME、MMBench、ScienceQA-IMG、TextVQA-OCR 的原生分数取自 `logs/e014_aee/full/bela_final_p1_20261025/L2_LATE_64_17_10/`。该配置的 stage1/stage2 与 L2 原生一致，且没有 QA。L2_NATIVE 自己的后四项被队列以 POPE 低于 85 跳过。
- 问题条件更新的六项都在 `logs/e014_aee/full/bela_stage1_queryanchor_20261027/S1_QANCHOR_L2/`。后四项由 `launch_qacover_protect.py` 复用同一 `env_for`。

## 分数

显示到两位小数后与正文一致：

| 任务 | 原生 | 问题条件更新 | 指标 |
| --- | ---: | ---: | --- |
| GQA | 55.5653 → 55.57 | 56.5193 → 56.52 | exact_match，12578 |
| POPE | 82.2142 → 82.21 | 86.1636 → 86.16 | F1，9000 |
| MME | 1651.7823 → 1651.78 | 1715.6754 → 1715.68 | 感知+认知，2374 |
| MMBench-EN | 61.3402 → 61.34 | 61.7698 → 61.77 | gpt_eval_score，4329 |
| ScienceQA-IMG | 68.9142 → 68.91 | 69.6083 → 69.61 | exact_match，2017 |
| TextVQA-OCR | 53.4100 → 53.41 | 53.0620 → 53.06 | exact_match，5000 |

未舍入的 MME 差是 63.89。正文不写这个差值，只写两端的两位小数，避免和 1715.68−1651.78=63.90 混用。

相对参考 GQA 61.9、POPE 85.8、MME 1852、MMBench 64.1、ScienceQA-IMG 69.51、TextVQA-OCR 58.20。六项比值的平均：原生 93.5639%，更新 95.3414%，四舍五入为正文的 93.56% 与 95.34%。W=1024，1024/(576×32)=1/18。

POPE 精度/召回：原生 0.93112/0.73600，更新 0.91296/0.81578。召回乘 4500 得到整数 3312 与 3671，与平衡的 4500/4500 一致。正文写成 93.11%、73.60%、91.30%、81.58%。

匹配随机：GQA 0.559628 → 55.96，POPE F1 0.833435 → 83.34。错配：GQA 0.556766 → 55.68，POPE 文件不存在。

## 引用语境

相关工作里的判断对应交接时已定位的原文职责：CDPruner 的条件 DPP、MMTok 的联合覆盖、MoB 的双目标、SCoRe 的显著性加权 k-center、CoIn 的体积覆盖、QCPruner 的问题加权设施选址、STAR-Pro 的入口覆盖加嵌套精炼、SparseVLM 所读版本含高效注意力路径。本轮没有重新下载这 15 篇 PDF。引用键沿用 `references.bib`，编译后没有未定义引用。未使用的书目条目留在 bib 中，没有为凑数量写进正文。

## 模板

`cvpr.sty`、`preamble.tex`、`ieeenat_fullname.bst` 的 SHA256 与 `inputs/evidence_ledger.json` 中的记录一致。文档类仍是 `10pt,twocolumn,letterpaper`，`\usepackage[review]{cvpr}`，作者为 Anonymous。
