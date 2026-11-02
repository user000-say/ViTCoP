# NeXT-7B 核心代码包（供 Agent 修改）

- 生成时间: 2026-11-02T21:08:04+08:00
- git: `fc4c08c463f71d2df083b622842f29ae6de9d5cf`
- 环境: `vitcop-v100-4model` / transformers **5.20.2**
- 模型: `llava-hf/llava-v1.6-vicuna-7b-hf`
- **这不是** P0 指令 Handoff；这是可改的生产源码快照。

## 目录

| 目录 | 作用 |
|------|------|
| `next_forward/` | 生产补丁：`apply_llava_next.py` Stage1/2、`dual_cover.py`、`stage2.py`、`native_stage1.py`、`bela_next_compat.py` |
| `next_fix/` | Native packing / legacy bridge / progressive hooks |
| `next_budget/` | 多 crop 配额 |
| `next_configs/arms.json` | A0–A4：`320→91→46` 等 |
| `next_scripts/` | lmms entry、s2qk 队列、R4 gate（single_floor=88） |
| `qa_cover/` | QA-COVER：`query_anchor.py` + `llava_adapter.py`（NeXT 经 compat 调用） |
| `legacy_cover/` | `e013_stage1_support.py`（Dual COVER 核心） |
| `runtime_hf_tf520/` | 实际 import 的 HF LlavaNext/Llama（未改权重；NeXT 用 monkey-patch） |
| `smoke_evidence/` | 烟雾/闸门 JSON，非完整 ACC |

## 真实调用链（改代码时按此找点）

1. `lmms_bela_next_entry.py` → HF `LlavaNextForConditionalGeneration`
2. `apply_llava_next.apply_patch` hook `forward` / `generate`
3. Stage1: `native_stage1.run_native_stage1` → `bela_next_compat.original_dual` / `original_qa`
4. Dual: `dual_cover.dual_cover_one_crop` → `e013_stage1_support`
5. QA: `llava_adapter.apply_llava_query_anchor` → `select_question_anchored`
6. Stage2: `_llama_forward_with_cuts` @ L12/L20 → `stage2.select_and_prune`（当前：layer-**input** Q/K 打分，再剪 output）

## 已知缺陷（改之前必读）

见 `docs/05_NEXT_ACC_DROP_ENG_FIX_20261102.md` 与仓库审计。要点：
- 曾：slots×N crop、QA 模板稀释、Stage2 post-layer Q/K probe；部分已改但 **s2qk A0 MME/MMBench&lt;88% 已 SKIP**，**不能当修复已验证**。
- Dense `gen_parity=false` 仍未闭环。
- 单任务 rel&lt;88% 立即 SKIP，禁止继续空跑。

## 约束

- 不要混入 W4096 / 1.5 日程当 NeXT 默认。
- A0 名义 W=5120：`320×12+91×8+46×12`；1.5 对照是 W1024 `64/17/10`。
- 修改后先烟雾（K 事件、qa_swaps、slots_per_crop），再 R4。
