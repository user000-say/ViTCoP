# BELA NeXT 7B/13B × 94.4%/88.9% · V100 四卡主 Handoff
日期：2026-11-01  | 目标：训练自由图像 Token 剪枝跨模型竞争力与精度–效率
状态：独立代码/配置已通过本地 CPU 单测；**尚未接入远程 V100**。
Cursor 在真实 V100 repo 中完成逐个真实调用接入后才能开始 GPU 实验。

## 0. 本次模型和方法边界

- 完全不涉及 A800 的视频 ACC 60.07、LongVideoBench、帧/时序预算。
  这次是 LLaVA-NeXT Vicuna image **7B +13B**。
- 以用户 LLaVA-1.5-7B 已经表现不错的 BELA
  **Dual-depth COVER → QA bounded donor-swap → Stage2 progressive attention**
  为主要可迁移模块。不换成 STAR-Pro QR，也不把 MMTok完整算法拿来当BELA。
- 以论文 Table 5 的 NeXT token budget / cut 作为真实外部预算参照。
  **STAR T32** 的表面记号是单基础patch单位；5crop时本实验实际
  T160 patches/decoder-layer/image，T64↔T320。
  94.4%=1−160/2880，88.9%=1−320/2880。
- 四个单独计算预算：
  NeXT-7B 32层：T160 W5120；T320 W10240。
  NeXT-13B 40层：T160 W6400；T320 W12800。
  都是**patch-only visual token-layer**，nonpatch/newlines/text额外计。
- 不得再将 image 7B 的 576→K 直接硬编码到 NeXT。
  Star五裁剪(672×672)是 official comparability arm;
  native NeXT AnyRes（真实每题 patch N可能变化）单列一个协议，
  必须重新测同采样/同grid的 Dense，不与五裁剪成绩混表。

## 1. 图像版已有的直接依据，不要和 Video 结果混

历史 LLaVA-1.5-7B 数据：
- W4096，32层，固定 Stage1 `K256`, Dual COVER slots32 + **QA B8**
  (max_swaps8, donor16, drop12, seed17)，cuts afterL12/L20；
  后续 K74→36（512? 不，这里 256→74→36 严格 W4096）。
- 同日程 A，P3 两阶段纯 Q_ATT 的 GQA60.42、POPE87.00、
  MME1775.17、MMB64.52、SciQA68.72、TextVQA57.18，
  R6约98.77。Hybrid first RDRIFT laterQATT的R6约98.49。
- W2048 六任务曾见 hybrid略高 R6：不能宣称 RDRIFT 全部无效；
  但到 NeXT 一开始选纯 Q_ATTN_ALL 作为更稳健、可比的固定主线，
  在同架构/同日程/相同 dev panel 验证后才开启弱历史项。
- QA B8/K256≈3.1%，到 NeXT K320时B8只有2.5%，K640时B8只有1.25%，
  所以 QA_B 不能只测 B8 而宣布“BELA QA在13B上不行”。
- 五裁剪下 QA 应明确**每图总次数**还是**每crop次数**。
  初轮建议整图最多B8，按单独crop选择器以可靠元数据安排；
  次轮 B16/B24/B32测试，真实交换次数、耗时、候选ID必须记录。

## 2. 完整同 W日程：每格五臂（paper+源paper token-var+3个BELA提案）

`configs/arms.tsv` 包含完整20组；`budget.py`对每组验证准确W。

NeXT7B T160：afterL12/20，W5120
- A0_PAPER `320→91→46`（论文）
- A1_PAPER_VAR `320→100→40`（论文token-var）
- A2_LATE `320→82→52`（新增：末层+6）
- A3_LESS_S1 `288→136→48`（新增：S1-K减10%）
- A4_STRONG_LATE `256→160→64`（新增：给末层更多）

NeXT7B T320：afterL12/20，W10240
- A0 `640→182→92`
- A1 `640→200→80`
- A2 `640→170→100`
- A3 `576→248→112`
- A4 `512→320→128`

NeXT13B T160：afterL15/30，W6400
- A0 `320→80→40`
- A1 `320→84→34`
- A2 `320→72→52`
- A3 `288→112→40`
- A4 `256→128→64`

NeXT13B T320：afterL15/30，W12800
- A0 `640→160→80`
- A1 `640→168→68`
- A2 `640→144→104`
- A3 `576→224→80`
- A4 `512→256→128`

**A0 is NOT STAR-Pro method reproduction**:
A0 uses STAR-Pro's *schedule*, but Stage1 selector always BELA Dual COVER+QA.
To report true STAR-Pro scores, cite its original paper or separately
run official STARPro code under same checkpoint+benchmark.
Do not label BELA A0 "STAR-Pro" in experiment tables.

## 3. 为什么这组比盲搜更多K合理

- A0：与对手相同层均计算预算且相同 cut, 测 BELA selector本身。
- A1：公开的paper matched-W token-var控组，允许比较预算敏感性。
- A2：K0不变，弱化中期过度精简，强保留后期视觉证据；
  7B/13B stage depths不同，注意后期 K 的正负增幅不同。
- A3：适度减轻 prefill 前段高成本，换给中期空间；
  只重算 Dual COVER K，不允许从 K320/640事后截取。
- A4：更偏向中后层；在视频 S56 有收益的启发下作为探索，
  但 image NeXT 不保证方向一致；尤其 OCR/图表对初始覆盖敏感。
- 本轮 cut 固定以隔离候选预算。若前4组均不行，
  **下一轮**才改 cut（13B L15/L30, 7B L12/L20 不是同一深度）。
- 任一组真正并非同 W、图片被额外裁剪或模型层数不对，
  STOP，不报告为“94.4%/88.9%”。

## 4. 分阶段实验，而非四卡直接狂跑20个全10任务

Gate 0（只读/小样本）：
- 4块 V100当前确实可使用，不是过去某轮“0/2/4/7空闲”就现在空闲；
- 7B/13B模型config / tokenizer / processor / source SHA和模型实际加载；
  7B32层、13B40层，Vicuna不是任意NeXT-family；
- 五裁剪 576×5 real patch index、image_newline mask、
  Dense no-prune logits parity、Q/K reference、position/KV proof；
- `BELA_NEXT_V2_ENABLE=0`完全原始路径，不能修改旧BELA / Dense分数；
- V100必须 fp16，禁止 bf16与FlashAttention2；模型并行真正正确；
- 所有 test prompt、image ID、five-crop preprocess 稳定 SHA固定。

Stage A：单模型开局，选 7B-T160（94.4%）：
- A0,A2,A3,A4 4臂，补 A1 只有出现值得研究的paper budget异动；
- 独立开发面板每任务小批，优先 GQA、POPE、TextVQA(OCR)、MME，
  128–256真实item/任务按固定ID与seed，覆盖图/文字/计数；
- 别用官方 test split反复挑参数，正式MME官方或GQA test
  如果只能提供测试集，就另找 calibration split或不重叠验证题；
- **比较同一个Dense**，而不只是R6/R9；panel输出实际 IDs。
- 若QA B8过慢，记录/剖析，不偷偷关QA让不同臂跑不同方法。
- A0作为固定selector基线；先选一到两个更好日程。

Stage B：7B-T320，13B-T160/T320独立小屏：
- 7B T320：A0,A2,A4；最多额外A1（论文token-var）
- 13B T160：A0,A2,A4；最多额外A1
- 13B T320：A0,A2,A4；最多额外A1
- **不以7B赢家的K值代替13B；同候选比例可以当设计指导**
  但13B40层剪枝位置15/30和7B32层12/20不同。
- 高分辨率真实 image cropping/input special tokens同模型独立校验，
  不得直接把7B的hidden states reshape用于13B。

Stage C：每模型每预算**日程先冻结**，再做精简 QA/Crop/Query：
- T160: QA_B = [0,8,16]；T320:[0,8,24,32]，保持其最优日程。
  B0是必要消融，不能只选择性能最好的复杂方法。
- Stage1 crop quota 先 uniform; 如果 OCR/定位数据显示有高分辨率
  细节损失，再试一次 global-thumbnail-weight=1.5，
  same K0 same W; 不同时改QA和crop。
- Stage2 Q_ATTN_ALL先固定；question-only只作为最后的最小辅助
  实验，并确认 tokenizer spans 中**排除 assistant/template**；
  RDRIFT仅当Q_ATTN表现差且已作配对rank日志，再试weak历史
  w=.8/.9/1.0；不做历史w=.2全面扫。
- Stage1 `slots=32`, donor16, drop12初轮冻结，后续仅在真实
  运行瓶颈证据显示QA donor pool / feature width需要调整时做单因子消融。
- 不基于STAR或MMTok采纳其温度/α 参数并冒称BEL A超参已调好。

Stage D：最终锁四个winner（模型×比例），报告官方基准：
- 至少 GQA/ScienceQA-IMG/TextVQA/POPE/MME/MMBench-EN,
  MMBench-CN/MMVet；再按历史9任务补 AI2D/MMMU/VizWiz，
  必须注明不同benchmark的母体与评分脚本。
- 论文STAR的10-task mean将MME归一化（MME总分/20），
  BELA历史R6/R9是相对Dense保留率，不能直接把不同平均数横比。
- 所有最优参数在dev冻结后只跑一次最终 full eval；
  返回每 task绝对分、matchedDense、relative ratio、平均和
  Q/K/selector runtime，gather FLOPs/TTFT/end-to-end latency p50p90。
- 任何变量过多难以归因，先保证4个winner公平、重现。

## 5. 预算与NeXT原生AnyRes约束

`N_ref=2880` 是论文固定五裁剪参照；不是声明每个NeXT输入
都会有2880有效视觉 patch。
在native AnyRes真实样本N_i不同的时候：
- 如果N_i>=320, K0等可先按 `N_i/2880`比例缩放日程，
  并用 `dynamic_ratio_plan`逼近每张图的目标 W_i，
  记录**实际 W_i, N_i, compression ratio, achieve avg**。
- 此仅作为独立`NATIVE_ANYRES`协议和次级泛化实验。
  对于N_i非常少、group拆分不完整导致目标比率不可实现，
  fail closed，不能增加或伪造不存在视觉patch。
- 每个截图/题的原始patch数量和newline都必须逐条上报。
- 有显式视觉组时优先 crop-wise COVER,
  尤其保护thumbnail+局部OCR，防止global TopK把整组清空。
- **不允许把几个 image_newline 当 patch算进去**；
  尽管patch_W相同，不同newline策略真实FLOPs不等。

## 6. 四卡 V100真实工作安排

- **先运行** `tools/v100_preflight.py`，填上机器当前确实空闲4个
  physical GPU ID，真实模型config路径；禁止凭历史0/2/4/7仍空闲。
- V100 compute7.0: `torch_dtype=float16`、eager/兼容SDPA，
  不使用BF16、FlashAttn2/Ampere-only kernel。
- 7B FP16权重约14GB(十进制)；13B约26GB，另有视觉编码器和
  推理KV/峰值激活。单V10016GB通常不宜承担全部模型；
  13B加载模式随16GB/32GB显存决定。
- **不能** `torchrun --nproc_per_node=4` 便宣布4卡能装13B，
  torchrun基本是DDP复制模型。Cursor必须确认 source
  支持`device_map=balanced` / accelerate真正模型并行，
  并验证vision/projector/LLM层间跨卡tensor正确。
- 如真实每卡32GB可考虑 7B在1GPU并发4个独立job，
  13B在2GPU并发2个job（必须额外显存余量）；
  如果每卡16GB，保守7B2GPU/group，13B可能需要4GPU/group，
  不要承诺能够并发多个13B进程。
- 模型加载先1个小题 smoke，测 reserved/allocated memory；
  OOM时减批次到1、分组分层、关闭不必要attention cache等，
  但不得悄悄降低图像分辨率或改变benchmark。
- 4GPU每卡利用率及CUDA device_map都需要出示截图/JSON，
  记录真实进程ID和当前GPU memory监测，不抢别人卡。
- 优先 V100 7B-T160按一个组完成并返回P0数值，之后排队其它arm；
  不在dev未锁定情况下同时跑四套正式官方大测试。

## 7. 交回取证包与科学结论格式

`BELA_NEXT_V100_ROUND1_RETURN_YYYYMMDD.tar.gz` 至少包含：
1. `GPU_INVENTORY.json`: 4张GPU型号/显存占用/device IDs；
2. `CHECKPOINT_AND_SOURCE_LOCK.json`: 7B/13B checkpoint、
   tokenizer、processor、vision config、代码SHA；
3. `NEXT_ANYRES_CROP_AUDIT.jsonl`: 各模型16张真实图原patch ID
   /crop groups/newlines/unpad/prompt hash;
4. `DENSE_NO_PRUNE_LOGIT_PARITY.jsonl`（含同输入相对误差）；
5. `QK_AND_STAGE_TRACE.jsonl`: 两剪枝边界真实Q/K、每crop ID嵌套、
   newlines、pos/kv 数值校验与 token-layer W；
6. `DEV_PANEL_IDS.sha256`、`DEV_RESULTS.csv`、`DEV_PAIRED.jsonl`
   (同样题目dense vs每arm harness correctness;不要mix官方最终数据);
7. `QA_SWAPS_AND_OVERHEAD.csv`、`REAL_TTFT.csv`,
   `PATCH_SPECIAL_FULL_FLOPS.csv`（vision+decoder+selector分开）；
8. `ACTUAL_SOURCE_DIFF.patch` + `git rev-parse HEAD`或文件SHA；
9. `WINNERS_LOCK.json` 每model/budget锁定配置及选择依据；
10. `FULL_EVAL_FINAL.csv`只包含在冻结后正式跑完的分数；
    未跑完的留 NA，不得虚构。

## 8. 规则

失败的 P0必须显式返回不能掩盖；每模型两个 budget，四个模型×预算单元。
禁用所有未获证实的hidden-state代理、CPU fallback和性能数字。
不得自动下载超大模型到不可控磁盘，也不能自动给用户远程GPU做
任何没有真实账号连接的操作；Cursor在V100本地完成执行。
以上新代码已经离线运行/CPU验证；没有装到远程V100，
不能说本助手现在已经“开始了GPU测试”。
