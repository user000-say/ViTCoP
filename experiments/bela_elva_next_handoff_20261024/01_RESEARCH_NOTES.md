# 研究决策依据

- A2的POPE比A0提高，但GQA/TextVQA下降，说明最佳同W日程是任务相关的权衡。
- B1仅削减深层并没有带来同等POPE收益，因此“删掉深层冗余会直接提点”不成立。
- B2保留中层K120，同时把后段K20恢复到K32，GQA和MME明显上升；这支持深层仍有必要证据的假设，但Token身份是否变化需trace。
- B2的W2288大于A2的2048，若最后有较高TTFT不能成为主方法，只能成为有价值的机制干预。
- 将动态分配作为论文创新的条件：可用决策前信号预测每个样本的预算需求，且优于最强固定日程和随机预算分配；不能用未来层的信息决定过去的K。
- 停止条件：GQA无可靠改善、TextVQA进一步下降、POPE下降明显、或者实际Accuracy-Latency Pareto恶化。
- 相关原论文：https://arxiv.org/abs/2609.19990 (QCPruner); https://arxiv.org/html/2609.05916 (STAR-Pro); https://arxiv.org/html/2609.26484 (CoRePrune)
