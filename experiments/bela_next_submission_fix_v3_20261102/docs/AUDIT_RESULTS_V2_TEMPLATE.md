# NeXT-7B P0+深度 UID Index Audit 回报模板

## 运行溯源

- `run_id`：
- 当前 NeXT checkpoint / processor / tokenizer：
- 真实 `inspect.getfile`、模块 `__file__`、所有 monkey patch 生效时间：
- 当前及最近 s2qk A0 R4 的 `git_sha / file_sha256 / config_sha256`：
- 历史 7B W1024 `64→17→10` Q_ATTN 与 RDRIFT 的原始记录路径和 source hash：
- 独立 native Dense 源码、版本、图像协议与日志基线：
- 活跃 GPU job 状态（只读观察；不作假设）：

## E01–E10 总门槛

对原 `AUDIT_RETURN_TABLE.md` 填 PASS/FAIL/UNKNOWN 和逐条证据；**任何 FAIL 优先解决，UNKNOWN 需要具体取证方案**。

## D01–D52 细粒度表

| ID | 状态 PASS/FAIL/UNKNOWN | 真实运行文件:函数:行号 | run_id / SHA | 具体真实证据/trace片段 | 复现步骤 | 最小可验证修复 | 风险 |
|---|---|---|---|---|---|---|---|
| D01 | UNKNOWN | | | | | | |
| D02 | UNKNOWN | | | | | | |
| D03 | UNKNOWN | | | | | | |
| D04 | UNKNOWN | | | | | | |
| D05 | UNKNOWN | | | | | | |
| D06 | UNKNOWN | | | | | | |
| D07 | UNKNOWN | | | | | | |
| D08 | UNKNOWN | | | | | | |
| D09 | UNKNOWN | | | | | | |
| D10 | UNKNOWN | | | | | | |
| D11 | UNKNOWN | | | | | | |
| D12 | UNKNOWN | | | | | | |
| D13 | UNKNOWN | | | | | | |
| D14 | UNKNOWN | | | | | | |
| D15 | UNKNOWN | | | | | | |
| D16 | UNKNOWN | | | | | | |
| D17 | UNKNOWN | | | | | | |
| D18 | UNKNOWN | | | | | | |
| D19 | UNKNOWN | | | | | | |
| D20 | UNKNOWN | | | | | | |
| D21 | UNKNOWN | | | | | | |
| D22 | UNKNOWN | | | | | | |
| D23 | UNKNOWN | | | | | | |
| D24 | UNKNOWN | | | | | | |
| D25 | UNKNOWN | | | | | | |
| D26 | UNKNOWN | | | | | | |
| D27 | UNKNOWN | | | | | | |
| D28 | UNKNOWN | | | | | | |
| D29 | UNKNOWN | | | | | | |
| D30 | UNKNOWN | | | | | | |
| D31 | UNKNOWN | | | | | | |
| D32 | UNKNOWN | | | | | | |
| D33 | UNKNOWN | | | | | | |
| D34 | UNKNOWN | | | | | | |
| D35 | UNKNOWN | | | | | | |
| D36 | UNKNOWN | | | | | | |
| D37 | UNKNOWN | | | | | | |
| D38 | UNKNOWN | | | | | | |
| D39 | UNKNOWN | | | | | | |
| D40 | UNKNOWN | | | | | | |
| D41 | UNKNOWN | | | | | | |
| D42 | UNKNOWN | | | | | | |
| D43 | UNKNOWN | | | | | | |
| D44 | UNKNOWN | | | | | | |
| D45 | UNKNOWN | | | | | | |
| D46 | UNKNOWN | | | | | | |
| D47 | UNKNOWN | | | | | | |
| D48 | UNKNOWN | | | | | | |
| D49 | UNKNOWN | | | | | | |
| D50 | UNKNOWN | | | | | | |
| D51 | UNKNOWN | | | | | | |
| D52 | UNKNOWN | | | | | | |

## 工程问题分类和参数搜索准入

- **PROVEN_DEFECTS**：只能写代码和真实运行证据已经证明的 defect，包括合法范围内 UID 错位。
- **SUSPECTED_ONLY**：列前提与可证伪实验，不冒充已证实。
- **PATCHES**：按最小语义保真修复分别给出 .patch 和 regression tests，不改变原有 GPU 运行。
- **PARAM_SEARCH_GATE**：在 E01、E02、E03、E05、E06、E07、E08、E09、E10 与所有相关 D 门槛验收前标为 BLOCKED。
- **REFERENCE_RESULTS**：历史 95.34%/95.43% 需按原始 run 恢复关联。NeXT 95%+ 是期望目标，不是本轮成功声明。
