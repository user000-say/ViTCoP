# Cursor 请在真实 V100 节点执行：BELA NeXT 7B/13B × T160/T320

**研究目标**：LLaVA-NeXT Vicuna-7B 和 Vicuna-13B，在 94.4%（T160）
与 88.9%（T320）两档高分辨率图像压缩下检验 BELA
Dual COVER+B8 QA+Q_ATTN 的泛化，并通过同 W日程、QA B级别、
少量crop配额消融寻找各档最优。完全不涉及A800视频。

**立即阅读** `README.md`, `docs/00_MASTER_HANDOFF_CN.md`,
`docs/01_RESEARCH_PAPER_PARAMETER_MATRIX.md`,
`docs/02_ENGINEERING_NE_XT_ANYRES.md`,
`docs/03_V100_FOUR_GPU_RUNBOOK.md`,
`docs/04_SELECTION_AND_RESULTS_PROTOCOL.md`。

## P0 — 现在就做，但先不要占满GPU跑大基准

(1) 在真实V100工程中找 **两个确切checkpoint** 和原image
BELA入口：`llava-v1.6-vicuna-7b`(32layers) 与
`llava-v1.6-vicuna-13b`(40layers)。如果checkpoint不是这两个或
model/processor SHA与论文不同，记录差异，勿假装同配置复现。
找到真实模型目录和 eval command、`llava_arch.py`、
image_grid_pinpoints、`mm_patch_merge_type`, 各模型 patch newline 处理。

(2) 取得当前4张**真正空闲** V100 physical GPUs。
之前实验0/2/4/7曾空闲，但现在不能假设；检查运行任务并用
`v100_preflight.py --gpu-ids CURRENT_FOUR_IDS --config-7b PATH --config-13b PATH`
输出证据。V100禁止 bf16/FlashAttn2，用fp16和eager/兼容SDPA。
注意13B 4GPU装载必须真正 model-parallel，`torchrun`单纯4个
rank是模型复制不是切分。不要抢其他研究人员显卡。

(3) 将zip解压到真实repo的一个**新experiments目录**，不覆盖老
LLaVA-1.5-7B已冻结实验和旧NeXT模型。执行本包17?项本地
单测（具体以运行输出为准），再在V100执行同一套。

(4) **Cursor必须自己写实际代码integration patch**：
`BELA_NEXT_V2_ENABLE=0` 旧forward完全不变。
`=1` 时把本包精确 arm `K0/K1/K2, cut` 接进真实
LLaVA-NeXT forward及原BELA Stage1、Stage2。
Stage1以5组 ×576真patch为目标分组，保留 crop_id、
original_patch_id、unpad后的sequence位置、
image_newline和text完整。
Stage2对真实same-layer Q/K after RoPE 用Q_ATTN_ALL，
正确维护物理剪枝metadata/causal/cache。QA B8
按历史donor16,drop12,slots32,seed17但**需定义清楚B8是整图
还是每crop**，避免实际5×B8造成额外运行量。
这一步需要读取现场源码，不能把本包standalone planner误作已应用补丁。

(5) 16 个真实样本：no-prune Dense forward logits parity，
各5crop raw ID、special newline, full seq lengths、
Q/K numeric reference, position/KV cache/generation、
W精确验证。任何P0失败STOP并回传最小复现。不能先跑分后修。

## P1 — 开始4卡的第一轮开发集小屏（P0通过才算真正开始）

先 NeXT-7B T160（最难的94.4%）:
- A0 `320→91→46` L12/L20 W5120（**BELA method + paper schedule**）
- A2 `320→82→52` (same K0)
- A3 `288→136→48` (mild Stage1 reduction)
- A4 `256→160→64` (more late candidates)
- A1 `320→100→40` 可选paper token-var

保持所有 arm `Stage1 Dual COVER + QA B8`, `Stage2 Q_ATTN_ALL`,
uniform-crop quota、固定同model/同processor、采样ID与seed。
开发集只用**非最终评测split**，覆盖GQA/POPE/TextVQA/MME；
先128–256题/任务，不要直接对所有arm跑10任务正式测分。
记录独立的 Dense 与 paired prediction/answer correctness。
成功后按相同做 7B-T320和13B-T160/T320，默认A0/A2/A4三臂，
A1备选。所有精确日程在 `configs/arms.json|tsv`。

## P2 — 同日程下小的QA和crop调参

选每个模型/预算开发集最好的日程后再测试：
- T160:B0/B8/B16（控制QA轻重）
- T320:B0/B8/B24/B32
- 仍是原BELA true-dual, source donor来自整个原始patch pool，
  没有gold answer或生成答案泄漏。
- 仅在诊断OCR/位置问题时测 uniform-crop vs
  thumbnail-boost(1.5)配额，K0/W不变。
- 第一次部署不测RDRIFT；后来如必要同赢家对比
  Q100/RDRIFT W90/W80，用真实同task输入的Attention排序。

## P3 — 最终只跑4个winner的完整任务

四个最终winner=2 models × 2 budgets，
在完整dev冻结配置后跑官方任务并测真实加速：
GQA/ScienceQA-IMG/TextVQA/POPE/MME/MMBench EN+CN/MMVet；
能加AI2D/MMMU/VizWiz更好。
所有任务报告pruned abs, matched dense abs, relative ratio,
真实视觉patch W、special token W、TTFT/fixeddecode p50/p90、
QA与Stage1/Stage2耗时、显存/FLOPs。
有条件真实跑官方STAR-Pro NeXT image code做参考，
如不能，清晰标注论文数值而非自己复现。

## 返回
请打包 `BELA_NEXT_V100_ROUND1_RETURN_YYYYMMDD.tar.gz`:
- `GPU_INVENTORY.json`
- `MODEL_7B13B_VERSION_LOCK.json`
- `ANYRES_5CROP_AND_NEWLINE_AUDIT.jsonl`
- `P0_DENSE_PATCH_PARITY.jsonl`
- `QK_KV_POSITION_TRACE.jsonl`
- `DEV_IDS_SHA256` + per-arm full paired dev prediction / correctness
- `DEV_RESULTS.csv`+ `BEST_SCHEDULES.json`
- `ARM_ACTUAL_W_AND_EFF.csv`
- `TRUE_DIFF.patch` + changed live source scripts + SHA
- `BLOCKERS.md` if no model weights, VRAM, source incompatibility.

严禁把print config当作已运行，不允许偷偷换低分辨率、
量化、checkpoint、test prompt或盲目缩小K而不报告。

## 快速本地使用（需按真实路径修改）
```bash
PKG=/ABSOLUTE/PATH/BELA_NEXT_7B13B_V100_CURSOR_20261101
export PYTHONPATH="$PKG/src:$PKG:${PYTHONPATH:-}"
python -m unittest discover -s "$PKG/tests" -v
python "$PKG/tools/plan_priority_queue.py"
python "$PKG/tools/print_arm_env.py" --arm NEXT7_T160_A0_PAPER
python "$PKG/tools/print_arm_env.py" --arm NEXT13_T320_A2_LATE
# 在V100主机真实检查物理4GPU和模型config:
python "$PKG/tools/v100_preflight.py" --gpu-ids 0,2,4,7 \
 --config-7b /REAL/7b/config.json --config-13b /REAL/13b/config.json
# 注意上面示例GPU id是历史的，只能使用V100机器当前空闲卡！
```
