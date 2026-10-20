# 本轮核查的原始来源

核查日期：2026-10-18。收录用来改变研究判断的关键近邻，不宣称穷尽检索。论文自报数字不是本项目复现。

| 来源 | 与本项目直接相关的内容 | 原始链接 |
|---|---|---|
| FastV (2024) | 早层后视觉token剪枝；必须比较已有交互后剪枝 | https://arxiv.org/abs/2403.06764 |
| PyramidDrop (2024/2025) | 渐进视觉压缩，不可将分阶段本身视为新颖性 | https://arxiv.org/abs/2410.17247 |
| SparseVLM (2024/ICML 2025) | 文本raters、动态比例、token recycling，附录有FlashAttention兼容讨论 | https://arxiv.org/html/2410.04417v4 |
| LLaVA-PruMerge (2024) | 选择与合并的已有路线 | https://arxiv.org/abs/2403.15388 |
| Libra-Merging (CVPR 2025) | importance/redundancy和prune/merge取舍，不相似token不合并 | https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Libra-Merging_Importance-redundancy_and_Pruning-merging_Trade-off_for_Acceleration_Plug-in_in_Large_CVPR_2025_paper.html |
| FOLDER (ICCV 2025) | 高效匹配/聚合；主要位置与我们交互层不同 | https://openaccess.thecvf.com/content/ICCV2025/html/Wang_FOLDER_Accelerating_Multi-Modal_Large_Language_Models_with_Enhanced_Performance_ICCV_2025_paper.html |
| VisionTrim (2026) | 视觉选择与文本引导补充合并 | https://arxiv.org/abs/2601.22674 |
| ADSC (公开2026) | 无打分均匀下采样、FlashAttention兼容的强低成本参照 | https://arxiv.org/abs/2602.12618 |
| When Token Pruning is Worse than Random (CVPR 2026) | 信息深度依赖任务与模型；深层随机参照 | https://openaccess.thecvf.com/content/CVPR2026/html/Wang_When_Token_Pruning_is_Worse_than_Random_Understanding_Visual_Token_CVPR_2026_paper.html |
| CVPR 2027 CFP | 原创研究要求、投稿日期、同期工作规则 | https://cvpr.thecvf.com/Conferences/2027/CallForPapers |

Libra网页本轮搜索能返回官方摘要，但直接再次打开被403拒绝；不声称完整重读该PDF。上轮提到的OccamToken/ET-Prune不作为本轮新实现成立的依据；在正式写相关工作和主表前应绑定它们的真实版本与代码，不能仅凭上轮概述认定优劣。
