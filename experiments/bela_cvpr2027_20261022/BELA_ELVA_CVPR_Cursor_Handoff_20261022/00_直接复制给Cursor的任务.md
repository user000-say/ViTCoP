# 直接复制给 Cursor 的执行任务（2026-10-22）

你是接手 LLaVA-1.5-7B 上 BELA/ELVA 的研发工程师。请阅读本目录：
- `01_研究负责人独立判断.md`：真实科研目的、历史结果、为何这些实验；
- `02_CURSOR_HANDOFF_CN.md`：必须遵守的接线步骤、先验检查与停止条件；
- `03_历史实验与近邻核查索引.md`：E001—E014 与后续 47 轮实验历史；
- `configs/experiment_matrix.csv`：事先固定实验臂；
- `bela_next/` 和 `scripts/`：可通过 CPU 测试的参考实现。

**必须先在项目工作目录内审计现役代码。不要假装此包已经包含当前生产的 COVER/RDRIFT 源码。E014 是旧快照，不能直接覆盖现役源文件。禁止把用户最新想法直接当成实验结论。**

第一阶段只做：
1. 冻结代码 Git SHA、所有实际 import 路径、checkpoint、数据集、Prompt、官方 scorer、GPU 性能配置，产出 `BASELINE_LOCK.md`。
2. 测试本交付包：`python -m unittest discover -s tests -v`（进入本包目录）。
3. 在现役原始模型上实现**只读 Trace**：Stage1 选点 ID、真实 COVER 对不同 K 的实际集合/加入顺序、双层视觉特征；RDRIFT 各次原 Patch ID、post-block 剪枝、每层实际 Token 数、位置/KV/mask。不可改变旧选择规则。
4. 核对 W2048 的真实 Q-Attn/RDRIFT 基线；POPE random/popular/adversarial confusion、GQA 类别；官方 STAR-Pro 的同机参照与实际 TTFT。
5. 审核 Gate G1 的两组静态同预算配置 (K0=160 和128)，**先只提交实验建议和需要修改的代码 diff**，不要无确认启动海量 GPU 搜索。

只有上述通过后才开启新分支开发：
- Stage1 COVER 未覆盖特征信号（默认 OFF，既有算法不变）。
- Frozen DEV quantile 策略下 1536/2048/2560 的入口+交互预算分配。
- 匹配相同预算档数量的 shuffled budget 对照。
- Stage2 cut12/cut20 的同样本 W 相等的二选一日程；若不能使用低成本的同 ID 历史注意力则不可开启。

**代码禁令：**不可伪造 `greedy_order`（空间排序 ID 不是贪心加入顺序）；不能调 test 集阈值；不能把 token-layer 减少率当 FLOPs/TTFT；不能把新模块降低 FFN/Attention 的收益声明成真实测速；不能直接改 checkpoint/评分器；不能在动态预算实验里省略 shuffled 控制；不能在 prefill 后错误复活已删 Token 或错位 KV/position。

请最终提交真实 `BASELINE_LOCK.md`、生产源代码 diff、单元测试和 GPU prefill+decode1/2 核验、POPE 逐类诊断、逐样本实际 W、完整基准和 TTFT/Fixed32、`STOP_GO_DECISION.md`。没有跑完的项目标记 `NOT_RUN`，不要补数据。

**目标是形成有条件动态 Stage1 + 渐进 Stage2 的可证伪研究证据，不是为了保持 BELA 名称而不惜代价增加模块。CVPR 注册 2026-11-10 AoE，投稿 2026-11-16 AoE，代码及论文同步推进。**
