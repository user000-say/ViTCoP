# PROVEN_DEFECTS / PROPOSED_FIX（只记录，本轮不落地新修）

## PROVEN
1. **Dual slots×N crops**：每 crop 满额 slots=32 → COVER 约 5×；已在 `bela_next_compat.original_dual` 改为全局 32 均分（smoke 已证）。
2. **QA 模板稀释**：非 image token 全进 query → swaps=0；已改为 USER…ASSISTANT 原 token 切片（未 re-encode）。
3. **Stage2 post-layer Q/K probe**：与 1.5 `_e018_callback(pre-layer Q/K)` 合同不符；已改为 input Q/K 再剪 output（s2qk）。
4. **mid_layer=10** 曾与 ~82 R4 同捆；已回退 **mid_layer=9**。
5. **s2qk A0**：MME rel 81.61%、MMBench 84.36% → 按 single_floor=88 已 SKIP；完整 R4 未齐，**不得声称修复有效**。

## PROPOSED_FIX（下一轮，需独立 worktree + 消融）
1. Stage2 query 限制为真实 question span（对齐 e006_span），非 `max(patch)+1` 之后全部。
2. Dual：CLS→patch attention saliency + β=0.5 归一化 concat；unpad tile 用 `origin_local` 映射进 24×24 COVER（勿对 packed row 硬套 grid）。
3. QA：接入 frozen content_mask；dual_metric 用与 COVER 相同的归一化描述子。
4. Dense gen_parity：修 ENABLE=0 generate 旁路直至与未 patch 模型一致。
5. 每次只改一项，R4 单任务 <88% 立即 SKIP。
