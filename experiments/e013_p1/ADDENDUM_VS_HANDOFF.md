# E013: 原 Handoff vs Review Addendum（2026-10-09）+ 执行调整

**User locks（始终优先）：**
- ACC：只认四数据集全量 N（GQA12578 / POPE9000 / MME2374 / MMBench4329）。Panel/limit **不算** acc。
- Latency：Fixed32 相对 |Δ| ≤ **15%** 作为可接受门槛；headline 在 ABBA 解释清 E012~90ms 前 **HOLD**（可测可报数，不宣称算法加速）。

**Addendum 总原则：** KEEP CURRENT RUNNING JOBS；不热替换 live 进程；补充诊断放在当前 panel/full 之后。

## 对比表

| 主题 | 原 Handoff | Addendum | 我们怎么做 |
|---|---|---|---|
| KV audit EXTRA | 要求减 `172+b−96` | 原包公式错；需 KVFIX；错 audit 的 EXTRA smoke=`AUDIT_INVALID`；勿热补正在跑的 acc | Live 已是 `visual_before=172+b`（SHA `6a4f4c31…`）。Smoke EXTRA 已见 `visual_before=188` 且 PASS→**记为 KVFIX-equivalent smoke VALID**。运行中 full/ABBA **不换源码**。包内未上传 `*_KVFIX.py`，不另热补。 |
| screen1/2 headroom | 必做 | 必做；强调 sample/seed/SHA；REPLACE_ATTN≠简单Top172；panel MME 不与全量混算 | 已完成 panel；**acc 结论只看 full** |
| Full ACC | 晋级 1–2 臂 | 保留现有 headroom；不因 addendum 砍掉对照 | 按你要求跑 **全量对照集**（BASE/B0/REPLACE×3/EXTRA×5），不中断 |
| Bridge L12 | 可选（E008 复核不了才做） | **改为必做**；Δ8 与 Δ12 各自相对同 S 的 BASE；可复用已跑 S8 | Panel S8 已有 BASE/EXTRA_ATTN/RANDOM→复用算 Δ8。Full 结束后必跑 bridge12 panel（S12）。**Bridge 不作正式全量 acc 替代** |
| CROSSIMAGE | 未要求 | 新分支+donor mapping；无 env 冒充；无则不能声称内容特异性 | **不做**（无补丁）。RETURN 标明未隔离图像语义 |
| 表示路径 / fixed-ID | 未要求 | 入围样本离线 Jaccard / hidden shift；fixed-ID 新分支禁混主 run | Full 出数后离线脚本；不改主模型 |
| Utility / logprob | 提 gold-logprob 若有 | 禁止伪造 Δlogp；先用 Δcorrect + cluster bootstrap；logprob 二期 | Full 后做 `Δcorrect` 配对分析 |
| ABBA / latency | ABBA 后 Stage1 latency | ~90ms 未解释前 **latency headline HOLD**；acc 可继续 | ABBA 继续；gate=你的 **15%**；Stage1 仍测；RETURN 写 HOLD/15% 双栏 |
| 结论措辞 | 勿宣称 COVER 新方法 | 同上 + 勿写 temporary evidence 被吸收 / utility 可预测 | RETURN 严格执行 |

## 不中断的当前作业
- Full ACC queue（GPU 0–3,5–7）
- ABBA on GPU4（E012_S12D12 B …）
- Watchdog 保活

## 调整后的后续顺序（SSH-safe）
1. **不碰** 正在跑的 full / ABBA 源码与进程。
2. ABBA 四段 → gate@15% → Stage1 latency（测+归档；headline HOLD until ABBA 叙事清楚）。
3. Full ACC 齐 → `FULL_ACCURACY_TABLE.json`（唯一 acc 晋级依据）。
4. **必做 bridge12 panel**（同冻结 panel）：BASE_S12, EXTRA_ATTN16_S12, EXTRA_RANDOM16_S12（+可选 COVER）；与已有 S8 panel 算 Δ8/Δ12。
5. 离线：Δcorrect / cheap-feature（有日志才算）；无 CROSSIMAGE / 无 fixed-ID 干预。
6. `E013_RETURN.md` 按 addendum 交付清单写齐。
