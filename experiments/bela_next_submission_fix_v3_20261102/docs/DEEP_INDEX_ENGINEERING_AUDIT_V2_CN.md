# BELA NeXT-7B：52 项工程失真深度审计门槛 D01–D52

**标准**：必须提供真实实际 import 文件:函数:行号、run/source SHA、真实 tensor shape/index/UID 对齐结果。CPU 合成单测 PASS 不等于 GPU 的 PASS。所有阈值、特殊处理、补丁与 fallback 都要从运行时配置/源代码核对。复现失败有定位证据；无证据 UNKNOWN。每项按 `AUDIT_RESULTS_V2_TEMPLATE.md` 回报。

## A. 模型、评测、环境（D01–D07）

- D01 — checkpoint、processor 与 tokenizer 版本固定，VisionTower / LLM / projector 在相同配置下加载，不混 Vicuna/Mistral 的 prompt 模板。
- D02 — `inspect.getfile`/`__file__`/进程起始配置指纹正确，在线 worker 与仓库同名源码不混淆，记录 monkey patch 生效时间。
- D03 — adapter 剪枝完全关闭时的 Dense 全链：preprocess → CLIP → projected features → pack → merge → logits，与同环境原生 Dense 数值对照；测精确/容差及可能的非确定性。
- D04 — 视觉 feature 入口（CLIP 哪层输出、remove CLS、projector）与原生 checkpoint 配置一致；多层选点不能改写原模型送进 LLM 的视觉内容。
- D05 — 各任务样本 ID、图像预处理/格式、prompt/chat template、解码参数和 scorer 对齐；单独输出 SciQA 原始答案与官方/诊断抽取，不能偷换官方评分。
- D06 — GPU/进程占用与队列管理只读；活跃 s2qk A0 R4 不热改/中断，GPU0–3 不占用；旧 v4 监听不恢复空心跳（如任务已结束则按现实状态报告）。
- D07 — 跑分每个任务的 run_id/source SHA/config hash、逐样本输出路径归属唯一；失败与重跑不能混分母。

## B. 原图 → AnyRes Crop → CLIP patch（D08–D17）

- D08 — `image_sizes` 统一为 `(H,W)`，从 PIL/processor 到 native grid 不宽高倒置；宽图/竖图需要覆盖。
- D09 — `image_grid_pinpoints`、`select_best_resolution`、缩放与 padding 位置遵循实际 checkpoint，原版 Dense 和 adapter 输入一致。
- D10 — base thumbnail 与 high-resolution tiles 的 crop 数、顺序、crop0 是否特殊、group offset 全明确；固定五裁剪是特定协议而非全部样本永真。
- D11 — 每张图 raw patch count 与 CLIP `patch_size/image_size` 相容，CLS 与 patch 区间边界 `(index 0 vs 1)`、hidden-state 层编号不发生偏一。
- D12 — CLIP 第9层与 penultimate 特征确指预期 block，不能将 `[9]`、`[10]` 与 layer9 标签混用；归一化/融合权重保留 7B 原算法。
- D13 — 每个 Patch 的唯一原始 UID 跨 `(sample,crop,local_index)` 唯一，不把每个 crop 的 `0..575` 当全局 ID。
- D14 — `view/reshape/permute/transpose/contiguous` 与 (row,column) 网格展平的关系有数值型 UID oracle，比对 row-major 与 column-major。
- D15 — patch grid 下边界、右边界、padding 头/尾、ceil/floor/round off-by-one，以及极端长宽比均有样本和断言。
- D16 — 图像数大于1、batch>1、padding side、image placeholder 个数正确，每张图与文本片段的边界不会互相串位。
- D17 — 进入 Stage1 的 token 是否包含最终被 unpad 丢弃的 padding？如果允许先选 padding，必须证明预算不会白花或给出修复证据。

## C. Unpad → Newline → Native pack → LLM embed（D18–D26）

- D18 — 核对**实际运行版本** `pack_image_features / unpad_image / image_newline`，并用上游未改动代码作为独立 reference；识别被自定义 concat 绕过的情况。
- D19 — 每一个 unpad 后局部 Patch 的 UID 来自哪个 crop、原始二维坐标；无重复、无缺失、无坐标错位，native map 与原生 reference 一致。
- D20 — `image_newline` 是可辨认的结构向量，数量、每行位置、是否有末行 newline 都与 native Dense 协议一致；绝不以 raw CLIP patch 假装它。
- D21 — Base thumbnail 与高级 crop 在 pack 中的相对顺序、图像边界及多图分别计算；同一 UID 在 native pack 只有一个位置。
- D22 — `feature_lens` 与最终 pack 真实长度相等，不能静态假设 2880 即原生 packed 长度；记录每样本 `raw_patches/valid_patches/newline/packed_length`。
- D23 — 原生 pack token 与 `image_token_index` 展开后的 LLM 绝对位置映射可逆；有左/右 padding 时 `input_ids`、`attention_mask` 和 `position_ids` 一致。
- D24 — 中间任何 `index_select/gather/scatter` 的索引在使用的**当下坐标系**内，记录 `(space_name,upper_bound,indices,uid_before,uid_after)`，旧 Crop local index 不得直接用于 packed tensor。
- D25 — 独立参考与当前代码对齐 UID，**同样合法的整数 index 但选错原始 Patch**须判 FAIL；仅验证 `min>=0/max<size` 不算通过。
- D26 — 保护标记范围（image_newline, image/text special token, prompt text）按真实协议区别处理；不能当普通视觉 patch 计算候选或误删。

## D. Stage1 Dual COVER（D27–D33）

- D27 — 实际调用的是原版 Dual COVER 逻辑，不是偷偷改为 sum/average attention、随机 score、STAR pivoted QR 或 `topk` 替代。
- D28 — `coverage` 距离/相似度与 normalization、saliency/penalty、tie-break、seed/稳定排序、`slots` 定义同 7B W1024 源码逐函数 diff。
- D29 — K 预算严格全图守恒，各 Crop quota / base 配额 / 有效候选长度/短缺补额可逐样本重现；`slots=32` 究竟全局还是每 crop 有证据。
- D30 — 如果将 COVER 切成 per-crop subproblem，核对 slots 的缩放与不重复覆盖区域，不能把全局 32 slots 在每一个 crop 上无意识复制五份。
- D31 — Selector 的 CLIP 特征与最终被送入 Projector/LLM 的 Patch UID **一一对应**；可保留原始视觉 embedding，但禁止跨 Patch index 错取。
- D32 — 有效选点集合满足唯一性、数量、mask 与边界条件；任何预算不足或重复选点要显式标记而非悄然降 K。
- D33 — 实际 `64→17→10` 的 7B 对照与 NeXT `320→91→46` 运行配置和入口 K 在代码中而不只是配置文件中相符；Stage1 所处时机明确。

## E. QA-COVER B8（D34–D39）

- D34 — QA 的调用点真实执行，配置写 `qa_swaps:8` 不算；运行 trace 含 accepted/attempted 与正/反例。
- D35 — `donor_pool=16`、`drop_pool=12`、`B=8`、`seed=17` 是历史参考值；是否按全图最多8次或每 crop8次必须由现役实际代码确认，不能静默改变语义。
- D36 — query embedding 与 visual embedding 的空间一致且规范化正确，跨 CLIP/projector/text 模态的对齐有代码与分数尺度核对。
- D37 — donor 需来自最初 eligible 未入选 Patch（或与历史逻辑完全一致的指定集合），drop 需当时已入选；每次替换后 K 恒定且无重复。
- D38 — `gain / novelty / coverage-loss guard` 接受准则与历史实现逐函数比较；不把 B8 误解释为强制接受8次，真实 accepted 可低于 B8。
- D39 — 多 crop 的 QA 交换不能丢失 UID 映射、误踩 newline 或削弱 global coverage；记录前后每 crop 分布和改动集合。

## F. Stage2 Q_ATTN + 索引链（D40–D46）

- D40 — 事件 `11:17 / 19:10` 究竟是 layer11/19 计算后剪、还是 layer12/20 前剪，确认真正的 12/8/12 视觉 token-layer 预算。
- D41 — Q_ATTEN_ALL 精确文本 Query 范围（所有文本 vs 最近文本）、位置与 mask；不存在把视觉/structure token 当文本 Query 的错误。
- D42 — 使用真实 attention 的 Q/K（含 Q RoPE、K RoPE、scale、mask、grouped-query heads）及正确 head 聚合；与独立精度 reference 对齐。
- D43 — Flash/SDPA 后端如果不提供 attention weights，禁止静默使用伪分数或全部零；任何 fallback 都要显式输出来源和验收状态。
- D44 — 第一次/第二次物理 gather 后视觉 UID 集合严格嵌套，旧索引必须经当前 survivors map 重新映射，绝不让已经被删的 UID 重新出现。
- D45 — 每次剪枝仅删目标视觉 Patch；对照文本/role/newline/特殊标记的保留计数、顺序和 prefill token identity。
- D46 — decoder tensor 真实缩短；同长度 shortcut/仅 mask/只改 YAML 不算物理剪枝，记录逐层序列长度与实际速度/显存口径。

## G. RoPE / KV / generate（D47–D52）

- D47 — `position_ids`/`cache_position` 在裁剪和生成阶段具备明确 policy，不能无意识重编为 `0..L-1`；如 intentional renumber，需 Dense 对照与 decode 数值核验。
- D48 — attention_mask / causal mask / batch padding 与视觉 patch 删除后的 attention Q/K shape、head dim、dtype、device 一致。
- D49 — 逐层 KV cache `key/value length` 相等并可解释，允许剪枝层之间 cache 长度不一样；不能错误断言各层一样长。
- D50 — 第一及后续生成 Token 的 `past_key_values`、`cache_position`、prefill/decode 分支、generation hook 与停止条件行为正确，长短问题/不同输出长度测试。
- D51 — 自定义 `generate` 的采样、EOS/newline 停止、`max_new_tokens`、答案清洗与原生 Dense 协议一致，不制造 ScienceQA `B.` 等格式偏差。
- D52 — 真实 checkpoint 上完成：剪枝关闭 Native Dense 等价；剪枝开启 Stage1 UID/QA/两次 Stage2/2+ decode tokens 证据；六任务评分不改变；在源码/参数冻结后才放行参数搜索。

## 必须提交的硬证据

1. 每样本 `(orig_H,orig_W,grid,crops,raw_patch_UID,unpad_UID,native_pack_UID,llm_UID,selected_UID)` 详细映射；至少一个宽图/竖图/方图/少 crop/多图。
2. 每一次 `gather/index_select` 的当前 coordinate space、size、index 列表、输出 UID 与**独立原生 reference** UID；可用 `bela_trace_gate_v2.py` 初筛。
3. 两条历史 W1024 版本，原生 HF Dense，现役 NeXT 版本三方源码 diff 的关键函数/行号、QA 调用证据、Q/K 的独立数值 oracle 和第2个生成 token 后的 KV 数值/长度证据。
4. `P0 工程错误` 与 `P1 参数优化` 分开记录。**新 slots/B/crop/日程一定在 P0 实现保真验收后再做**。跑分目标不是本轮的 PASS 条件。
