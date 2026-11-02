# 审计报告：已定位的缺陷、修复和剩余风险

依据：用户上传的 NeXT-7B 生产源码包（git `fc4c08c463f71d2df083b622842f29ae6de9d5cf`，其 docs 描述 transformers 5.20.2）。使用 `PATCHES/` 查询逐文件 diff。这里**区分代码证据、修复意图、GPU 证据**。

| ID | 原始问题/代码证据 | 修复文件 | 验证层级 |
|---|---|---|---|
| P0-01 | `next_forward/apply_llava_next.py` 通过硬编码 `BELA_NEXT_7B13B_V100_CURSOR_20261101/configs/arms.json` 找配置，当前 ZIP 单独部署会失效 | `bela_next_forward/apply_llava_next.py` 配置本地化、W 检验、版本及源码 SHA256 | CPU PASS，真实生产环境待比对 |
| P0-02 | 旧 `generate` 人工限制最多 256 token；CUDA OOM 直接 `return input_ids`，会让失败样本混入评分 | 新严格贪心解码：无 OOM 静默返回，显式拒绝 unsupported 参数，有限支持自定义 stopping criteria | CPU 参数测试 PASS，生成协议 GPU UNKNOWN |
| P0-03 | 旧 `hooked_cg_forward` 的 Dense 分支可能清理 `pixel_values`/`image_sizes`，改变 Dense 行为 | BELA 关闭时调用被捕获的原生 Forward，未清理入参 | 源码审核 PASS，数值 Dense parity GPU UNKNOWN |
| P0-04 | 原 `native_stage1.py` 直接取 `pixel_values[0]`，没有依 `image_sizes` 截掉批处理 padding 的 crops | `_trim_to_native_crops` 使用目标版本 `image_size_to_num_patches` | 源码检查，真实 AnyRes 多比例 GPU UNKNOWN |
| P0-05 | 原输入图片占位符重写能掩盖 processor 原始视图长度与 native packed 长度不一致 | `_verify_native_placeholders` 在修剪之前核对原生 packed_len、连续性和 mask | CPU PASS，真实 processor GPU UNKNOWN |
| P0-06 | 原 Dual 实现可能每 Crop 重复使用 32 个 COVER slots；原 saliency 存在 norm-proxy/CLS 混淆风险 | 全局 slots=32 分配并约束各 Crop 有效余量；通过真实 CLIP CLS Q/K 计算 top-k saliency；`e013_choose` 对比 | CPU 多组对照 PASS，CLS eager oracle GPU UNKNOWN |
| P0-07 | 固定 24×24 和 unpad 后 Crop 内 index 在不等宽网格上不能直接复用 | `origin_crop`、`origin_local_patch` 映射，带原始坐标的 4×4 覆盖计算 | CPU 乱序测试 PASS，真实多比例 UID oracle GPU UNKNOWN |
| P0-08 | 旧 QA 部分依赖 `/home/xiangyue/ViTCoP/experiments` 和外部 `bela_query_anchor`；可能未真正使用对应 prompt 的 tokenizer | QA 代码本地闭环、lmms-eval 同源 tokenizer、只保留真实 USER…ASSISTANT 之间的原始 token ID；不再全 prompt 回退 | CPU PASS，真实多任务模板 GPU UNKNOWN |
| P0-09 | 旧 Stage2 的 Q/K 与 attention/缓存使用路径缺少数值与屏蔽证据；单纯 physical index in range 不够 | 输入侧 Q/K、GQA 重复、Causal+Padding mask、独立数值对照、真实 KV gather 长度校验 | CPU QATTN/PADDING/KV PASS，真实 fused backend oracle GPU UNKNOWN |
| P0-10 | Stage2 裁剪后的 `text_start=max(patch_pos)+1` 会把尾随 `image_newline` 混入 text Query | `_remap_text_start` 使用**完整物理 keep 索引**重映射真实文本起点，禁止跨越结构标记 | CPU 显式反例 PASS，真实 trace GPU UNKNOWN |
| P0-11 | 原 Stage2 cache 长度校验不充分，RoPE 位置重置的风险 | 逐层 DynamicCache gather、绝对 position_ids 保序、长度/UID/子集断言 | CPU PASS，prefill→多步 decode GPU UNKNOWN |
| P1-12 | 原实验 shell 可自动占用/保持 GPU、kill 被追踪 job、复用旧输出 | 历史队列脚本移至 `reference/original_scripts`；新 `run_one_task.sh` 不自动调度/杀进程；同名结果拒绝覆盖 | Shell 静态检查 PASS，真实调度未运行 |
| P1-13 | 旧 R4 gate 绝对路径硬编码 Dense、按排序随意取第一个 `_results.json` | 固定包内用户提供的原始 Dense 结果 JSON；遇到重复结果立即报错，禁止混分 | 静态代码 PASS，需实验域评分复核 |
| P1-14 | 没有模型启动源码指纹及每次剪枝真实 UID 证据 | preflight SHA256、BELA_TRACE_FULL_UID 轨迹、模型导出 `summarize_trace()` | CPU/synthetic PASS，GPU 抽样待回传 |

| P0-15 | 原 `sampling_policy=locked_paper_five_crop` 只在 JSON 中声明，运行时未核验；样本变成 1/3 Crop 仍能被误当 2880 基准的 94.4% 压缩 | `_require_paper_crop_protocol` 强制论文固定五 Crop，偏离直接拒绝 | CPU PASS，模型 processor GPU UNKNOWN |
| P0-16 | 原 QA tokenizer 可以按 model 路径重新加载，存在与实际 processor tokenizer 身份/added tokens 不一致的隐患 | `_tokenizer_for` 只接受评测入口绑定的**同一个 tokenizer 实例**，拒绝自动加载 | CPU PASS，端到端 GPU UNKNOWN |

## 必须公开的主要不确定性

1. 本包含 `e013_stage1_support.py` 和 QA Anchor 逻辑，但**没有 LLaVA-1.5 W1024 64/17/10 那次完整的 Decoder Q_ATTN 生产源码/日志**。必须拿到原始版本，完成 Q_ATTN 得分定义、层号和推理协议的一对一数学对照，不能凭相似命名宣称算法完全复现。
2. 即使注入 CLS Q/K 与 global slots 后精度提高，仍可能受 NeXT 多 Crop 重复证据、高分辨率噪声、任务长度、生成与评分差异影响；准确率提高不具确定性。
3. 目前未在服务器执行 NeXT checkpoint：无 Dense logits parity、原生 crop packed 全序比对、CLS eager oracle、真实 2 次 KV 剪枝和逐 token decode 回归，也没有六任务 mean6 或端到端时延。
4. 当前原始成绩包：旧 v4 A0 R4=90.2525%（相对 NeXT Dense）；新 s2qk 只给出了 MME=81.6077%、MMBench=84.3590%，且已被早停。**这些不能解释成 V3 修复后的成绩。**

## 如何识别假修复

- `qa_swaps_cfg=8` 不等于实际 B8 交换；需要 `swaps>0` / `qa_reason` 和 donor/drop UID 集合，单样本 0 也可能合法。
- `patch.max()<packed_len` 不等于采集正确；必须 UID 与真实原生 pack 一致。
- K0/K1/K2 正确不等于 Layer12/20 的 Q/K 信号正确；必须源码、算术和数值 oracle 同时通过。
- 层均 W5120 不含 image_newline、文本和生成开销；论文加速要另报真实 prefill FLOPs、latency、峰值显存、端到端时间及 token-layer 口径。
