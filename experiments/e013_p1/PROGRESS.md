# E013 live progress (auto-checked)

Rules: ACC=full-4 only; latency |rel|≤15% + headline HOLD until ABBA explains ~90ms.

| Track | State |
|---|---|
| Smoke | PASS (KVFIX-eq VALID) |
| Panel screen1/2 | done (diagnostic) |
| Full ACC | running — sole ACC evidence |
| ABBA | LEGACY_B in progress (last of 4) |
| Stage1 latency | after ABBA gate@15% |
| Bridge12 | after full ACC |
| RETURN | watchdog writes continuously / final on COMPLETE |

Watch: `watchdog.log`, `master_driver.log`, `queue.log`, `FULL_ACCURACY_TABLE.json`, `E013_RETURN.md`.
