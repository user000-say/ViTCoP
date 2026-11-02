# BELA NeXT-7B：P0 可执行修复交付包 V3（2026-11-02）

本包以用户上传的 `BELA_NEXT7B_CORE_CODE_FOR_AGENT_20261102.zip` 为输入，包含**完整可替换 Python 模块、单任务入口、预算配置、CPU 回归测试、原始文件副本、修改差异、Cursor Handoff 与审计工具**。

## 重要边界

- **已验证**：33 个 CPU/PyTorch 合成与数学单元测试、全源码 `compileall`、Shell 静态语法、W5120 算式；运行时原生源码哈希的校验逻辑已接入。
- **未验证**：没有你们 GPU/checkpoint、实际 NeXT ViT/LLM、lmms-eval 完整数据以及旧 W1024 Decoder 源码，因此**未完成任何真实 NeXT 精度恢复证明**。不能声称性能达到 95% 或已经超过论文。
- **不允许直接覆盖正在运行的 s2qk 队列**；独立目录和新日志、空闲 GPU 上执行。
- 仅支持：HF transformers **5.20.2**（源码哈希必须匹配）、`llava-hf/llava-v1.6-vicuna-7b-hf`、单图/单样本、32 层 L12/L20、固定预算、贪心解码。超出这些条件**显式报错**。

## 结构

- `bela_next_forward/`：主生产 Stage1/Stage2、native packed UID 跟踪与生成入口。
- `bela_next_fix/`, `bela_next_budget/`, `bela_query_anchor/`, `bela_legacy_cover/`：本地闭环依赖和原 BELA 算法。
- `next_configs/arms.json`：A0–A4 原始等预算矩阵；不伪称已验证的最优参数。
- `next_scripts/lmms_bela_next_entry.py`：携带真实 lmms-eval tokenizer 的入口。
- `next_scripts/run_one_task.sh`：只在显式选定 GPU 上运行一个任务，不管理或终止其他任务。
- `next_scripts/gate_next7_arm_r4.py`：唯一结果文件 + Dense 相对分数闸门。
- `tools/preflight.py`：实际源码哈希及环境指纹。
- `tests/`：无 GPU 的 CPU 测试。`audit_tools/`：此前 V2 的独立轨迹与数值 oracle。
- `reference/`：上传时的运行源码快照、成绩 JSON、原脚本和原文档（**仅比对，不作为替换环境代码**）。
- `PATCHES/`：源文件差异。`docs/`：Cursor Handoff、审计报告、论文级实验门槛。

## 快速验证（不得启动现有任务队列）

```bash
cd /path/to/BELA_NEXT7B_SUBMISSION_FIX_V3_20261102
python -m pytest -q tests
python -m compileall -q bela_next_forward bela_next_budget bela_next_fix bela_query_anchor
python tools/preflight.py --arm NEXT7_T160_A0_PAPER
# 有相同的 HF 5.20.2 环境才能执行：
python tools/preflight.py --runtime --arm NEXT7_T160_A0_PAPER
```

GPU smoke 前必须先阅读 `docs/HANDOFF_TO_CURSOR_CN.md`，处理真正的工作环境路径、stop criteria 和原生编码器 oracle。一个示例（选择**实际空闲的** GPU，输出目录不得已有其他实验结果）：

```bash
export LMMS_EVAL_ROOT=/path/to/lmms-eval
export BELA_PYTHON=/path/to/validated/python
export BELA_LMMS_LIMIT=4
export BELA_NATIVE_DENSE_ORACLE=1
export BELA_TRACE_FULL_UID=1
CUDA_VISIBLE_DEVICES=<free-gpu> bash next_scripts/run_one_task.sh NEXT7_T160_A0_PAPER mme /path/to/NEW_SMOKE_DIR
```

请不要把上面的 `<free-gpu>` 当作 shell 中的合法参数；替换为自己确实有权限且空闲的 GPU 编号。

**正确的历史边界**：LLaVA-1.5-7B W1024、`64×12 + 17×8 + 10×12`、94.44% token-layer 压缩；NeXT-7B W5120、`320×12 + 91×8 + 46×12`、以名义 2880×32 为 Dense 参考时也是 94.44%。高分辨率绝对保留更多 Patch，但准确率仍要实测。
