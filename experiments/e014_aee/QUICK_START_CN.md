# E014-AEE 中文交付说明 / Cursor 快速执行

日期：2026-10-10
工程状态：**基于两份完整 E013 源码重新生成；静态检查和 CPU 测试已通过；真实 GPU 实验尚未执行。**

## 0. 这轮到底实现了什么？

E014 的核心是固定 Stage1 `172` 个**可被 QED 选择的视觉 token**，
执行图像自适应证据交换，而不是再增加 `transient` token。

- `Core`: 原始 CLS attention Top172
- `Residual`: 未进入 Core 的其余 404 个 CLIP patch
- `Add`: 在 Residual 中根据 COVER proxy 选择 `b` 个
- `Remove`: 在 Core 中依据 redundancy-aware eviction risk 删除 `b` 个
- `Final`: (Core - Remove) + Add，总数 **恰好 172**，全部 QED-eligible
- `b` 候选：`0,8,16,24,32`，通过轻量 gain-risk proxy 决策
- 主路径：仍为 `E007 ORIGINAL_172 → E010 QED → E012 S=D`，不修改 QED/SDPA 数学实现
- 首要：`K96 / L8`；之后检验 `K64/K32`，以及可选 `L12` 桥接
- 8 卡任务：K96 36 个四数据集全量任务，K64 20 个，K32 20 个，共 76

这个实现是 **AEE-v0 的可测原型**。图像自适应收益需要用完整数据验证，
不能因为写了控制器或选出了不同的 `b` 就宣称准确率或真实延迟提升。

## 1. 与你们 E013 结果的关系

历史 E013 完整评测为：
GQA 12578 / POPE 9000 / MME 2374 / MMBench 4329 条（对应当时的 split/filter）。
已有信号：`REPLACE_COVER16` 对 Top172 的 GQA / MME 有正向收益，
但没有证明它显著超过所有固定替换预算，也没有证明 adaptive 一定更强。

**特别注意 COVER 的严格可比性：** 两份新上传的主源码没有附带
`e013_stage1_support.py` 的实际实现。
因此 E014 中 `FIXED_COVER` **直接调用你服务器原有的 E013 helper**。
`FIXED_COVER_PROXY` 是单独标注的近似策略，不能混称“精确 E013 COVER”。
需要做一次相同图片、相同请求、相同 S=D 的 token-ID identity 验证。

## 2. 全部源码与文件位置

交付包的 `llava/...` 两条主文件是基于你提供的 **完整 E013** 直接修改，
不是从片段重写；每一处 E014 接入带有 `E014-AEE-BEGIN/END` 注释。

核心文件：
- `llava/model/multimodal_encoder/clip_encoder.py`
- `llava/model/language_model/vitcop_modeling_llama.py`
- `llava/model/multimodal_encoder/e014_config.py`
- `llava/model/multimodal_encoder/e014_stage1_support.py`
- `E013_to_E014.patch`
- `E014_CURSOR_HANDOFF.md`
- `scripts/run_e014_suite.py` 和各 K 启动器
- `tests/test_e014_selector.py`, `tests/test_e014_contract.py`
- `configs/expected_n_e013_protocol.json`

**原版 SHA256 必须在服务器核对：**
- `clip_encoder_e013.py`: `b89208803528019460a47100fb5984e178c85cf12ca5f8ddb2981fda420ea73c`
- `vitcop_modeling_llama_e013.py`: `6a4f4c31334fa73f9e99395c93a6bff64e60c8b7b93f951f8e54c8c804651896`

## 3. 上服务器：先应用补丁再测试

在项目根目录：

```bash
git status --short
sha256sum llava/model/multimodal_encoder/clip_encoder.py \
  llava/model/language_model/vitcop_modeling_llama.py
git apply --check /PATH/TO/E014_AEE/E013_to_E014.patch
git apply /PATH/TO/E014_AEE/E013_to_E014.patch
git diff --check
python -m unittest discover -s tests -p 'test_e014_*.py' -v
```

如果 `git apply --check` 失败，说明服务器实际源码与本次上传的 E013
版本不完全一致。**不要强行覆盖**，先逐段 reconcile。
不要同时“复制整份新文件”和“应用 Patch”：二选一即可。
服务器原有仓库 README、模型权重、benchmark 数据/启动脚本均不在此包中。

## 4. 用原来的模型/评估入口启动

```bash
# K96 / L8，单图真实模型 smoke，允许审计同步
E014_VARIANT=AEE E014_AUDIT=True E012_KV_AUDIT=True \
  scripts/run_e014_k96.sh python YOUR_EXISTING_SMOKE.py

# 固定 B16 精确 E013 COVER 对照，要求旧 e013_stage1_support.py 仍在
E014_VARIANT=FIXED_COVER E014_B=16 \
  scripts/run_e014_k96.sh python YOUR_EXISTING_EVAL.py

# GPU 延时测量禁止 audit / KV audit
E014_VARIANT=AEE E014_AUDIT=False E012_KV_AUDIT=False \
  scripts/run_e014_k96.sh python YOUR_EXISTING_LATENCY.py
```

这里的 `YOUR_EXISTING_...py` 必须替换为你服务器上 E013 已验证的真实脚本，
并保持模型版本、FP16/SDPA、prompt/question-span、batch=1、seed、
warmup/repeats 完全一致。包里没有编造这些入口。

## 5. 四全量 + 八卡调度

先配置 `configs/commands.example.json` 中四个真实命令（不要直接执行占位文件）。
确认 FULL N 无过滤或截断后，再用历史协议的 expected-N：

```bash
python scripts/run_e014_suite.py --plan all --dry-run

python scripts/run_e014_suite.py \
  --plan k96 --gpus 0,1,2,3,4,5,6,7 \
  --cwd /YOUR/REAL/REPO \
  --commands-json /YOUR/commands.json \
  --expected-n-json configs/expected_n_e013_protocol.json \
  --out logs/e014/k96
```

K96 GPU/KV/cache/identity smoke 通过后，再运行 `--plan k64`、`--plan k32`。
每个 evaluator 必须在 `{out_dir}/eval_summary.json` 写入真实 `n` 和主指标 `score`。
如 split 变化，应修改 expected-N 文件为经过核对的真实数量，**不能为了验收改写统计 n**。

对照：
BASE、FIXED_COVER(B0/B8/B16/B24/B32)、FIXED_RISK、ADAPTIVE_B_ONLY、AEE。
重点不是仅比较 AEE vs BASE，而是**同 K 的最强固定 COVER**。

## 6. 验收与诚实边界

已在交付环境完成：
- 12 项 CPU 单测；
- 两个完整主文件及新增模块的 Python 编译；
- Shell `bash -n`；
- Patch 应用后文件字节一致性；
- ZIP 完整性及 SHA256 校验。

**尚未**完成：
- 模型权重加载、真实 GPU prefill / decode1 / decode2；
- E013 exact COVER 对照跨版本 token-ID identity；
- K96/K64/K32 的真实 cache/KV topology；
- GQA/POPE/MME/MMBench 四全量 ACC；
- TTFT / Fixed32 / E2E / GPU peak memory。

推荐按半天至一天为一个实验推进周期持续回传：
先 smoke，后四全量，再 paired ACC/latency 分析。
出现失败优先定位差异，不应为了进度把小样本结果冒充正式结论，
也不宜一遇负向 pilot 就完全停止探索。

最终需判断：AEE 是否在相同 token 预算下击败最强固定方案，
且未造成无法接受的真实推理开销。详细指令见 `E014_CURSOR_HANDOFF.md`。
