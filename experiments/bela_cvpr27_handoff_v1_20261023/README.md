# BELA / ELVA CVPR2027 — 冻结 Cursor 交接 V1.0 (2026-10-23)

**先读 [00_CURSOR_START_HERE.md](00_CURSOR_START_HERE.md)**，再看 01–06 文件。该包仅提供**研究审计和实验决策**，不是现役 GPU 模型代码补丁，不包含任何新 GPU 提点结果。

禁用旧的 Candidate-risk / Evidence Guard 实现作为本轮默认动作；只有当前交接书是执行依据。需跑实际实验时 Cursor 应依据服务器现役版本和评测器接入读取/分析工具。

离线验证：
```bash
python -m unittest discover -s tests -v
python scripts/audit.py schedules config/schedules.json
```

待真实数据导出后：
```bash
python scripts/audit.py trace outputs/TRACE.jsonl --arm A2 --schedule config/schedules.json
python scripts/paired.py outputs/A0_GQA.jsonl outputs/A2_GQA.jsonl
python scripts/paired.py outputs/A0_POPE.jsonl outputs/A2_POPE.jsonl --pope
```

`outputs/` 与 `SOURCE_DIFF.patch` 需 Cursor 在服务器上生成，本包不会伪造。
