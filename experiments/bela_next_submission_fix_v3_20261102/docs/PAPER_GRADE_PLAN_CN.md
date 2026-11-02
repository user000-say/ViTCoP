# BELA NeXT-7B 的投稿级实验计划（工程验收后才执行）

1. **固定预算**：NeXT 名义 2880×32 Dense；W5120 = 94.444% layer-token compression。Stage1 `320`, L12 保留 `91`, L20 保留 `46`，即 `320×12+91×8+46×12=5120`。旧 1.5 W1024 最强对照是 `64×12+17×8+10×12=1024`，不能把 W4096 最好臂挪作极限压缩对照。
2. **算法保真 gate**：One Crop 的 `e013_choose` COVER 选点严格等价；真实 CLS 权重、同源 question tokens、global QA-B8、Stage2 Q_ATTN 数值 oracle、原生 AnyRes UID 原封不动、KV/RoPE 正确、Dense 输出不变，全部有独立证据。
3. **统计指标**：6 benchmarks 给 absolute and normalized Dense mean6、平均差/置信区间、样本数与 split；不能只报 R4 或最佳值。对任何中止/超时/解析异常给失败计数，绝不降级为 Dense 或空答案。
4. **主效应和交互消融**：单独剥离 COVER、QA、Stage2；T=160 和 T=320 两档（**分别 W5120 和 W10240，绝不混用**）；等 W 形状 `[320,91,46]`, `[320,100,40]`, `[288,136,48]`, `[256,160,64]`。这些是预注册候选，均不是已经证明的最优参数。
5. **跨 Crop 对照**：等配额 vs 基图 thumbnail boost；global slots 8/16/32/64；QA B0/B4/B8；每次变化只改一个因素，同时记录 base/tile token 存活、QA crop drift 和关键区域证据召回。算术上确保每组真正满足 W 固定。
6. **真实效率**：视觉塔、mm projector、Stage1 选择、L12/L20 attention-QK 探针、KV gather、prefill、decode、端到端 wall time、峰值显存与额外 memory。至少一个严格固定平台 benchmark：warmup、run 数、batch=1、dtype、CUDA/torch/attention backend、计时范围。不能只用名义 Token 率代替节省。
7. **学术比较**：STAR-Pro 官方开源包含 NeXT 5 Crop 与 T160/320，主要用于对齐输入协议和 matched budget；不要直接移植它的 QR 选择来冒充 BELA。ST-Pruner 等新方法的 published claims 必须原协议与用户本地 scorer 重复；论文表格中的高百分比不必然是同一计算口径。
8. **创新论证**：定量证实 Dual depth + spatial novelty + query anchor + decoder adaptive refinement 的互补性；展示修复前工程消融 vs 正确实现；展示 token survival heatmap、跨裁剪 coverage / question evidence retrieval、失败案例归因。任何“修复后至少达到某指标”都只能作为假设，不写为结论。
