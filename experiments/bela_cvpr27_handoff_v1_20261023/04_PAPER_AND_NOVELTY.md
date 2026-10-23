# 近期论文真正给 BELA 的约束（截至 2026-10-23）

| 方法与来源 | 真实关注的研究问题 | 对 BELA 的结论 |
|---|---|---|
| PyramidDrop / VisionZip / STAR（2024–2025） | 显著性、多样性、渐进式压缩 | “覆盖 + 渐进”本身不是创新 |
| [STAR-Pro](https://arxiv.org/html/2609.05916)，[GitHub](https://github.com/EasonAI-5589/starpro) | Pivoted-QR 候选池 + 随层变化的文本视觉注意力 | 最强的直接同 W 竞争者，优先同机测试 |
| [QCPruner](https://arxiv.org/html/2609.19990) | 通过 query anchor/效用同时加权覆盖目标和代表 | 完整系统高分不等于最后的 selector 公式带来全部收益 |
| [CoRePrune](https://arxiv.org/html/2609.26484) | 在深度与共同删除集合中估计可删除性 | “联合删除风险”已有先行；不能换名字就算创新 |
| [MiCo](https://arxiv.org/html/2609.34330)，[GitHub](https://github.com/MAX-7237/MiCo) | 语义擦除下的信息覆盖目标，两个阶段统一优化 | 将 Stage1 信息传给 Stage2 不能独立形成贡献 |

STAR-Pro 原文的 T64 日程为 128×12 + 37×8 + 18×12 = W2048；它完整前 12 层累积 1536，BELA A2 对应 1648，不能泛泛说 STAR-Pro 只是“更多早期视觉计算”。但候选身份、连续的交互机会、剪枝时机、query signal 都变了，必须因果拆解。

QCPruner 附录的受控 Table 16 固定查询效用和剪枝设置后，在 LLaVA K64 下 QCPruner 和 CDPruner-DPP 的七任务相对值均为 98.6%；完整系统强不能简单归功于子集选择器。

**候选论文研究问题，而不是已成立贡献：** “视觉证据在不同 decoder 深度具有不同的存活必要性；在计算预算约束下，能否从早期的互补覆盖关系与实时图文交互中，经济地预测后续删除风险？” 若最后只得到固定比例最优，不能写成输入自适应创新；必须有可执行机制、Stage1×Stage2 交叉消融、同 W 同评测、真实延迟和失败样本才能主张 CVPR 创新。
