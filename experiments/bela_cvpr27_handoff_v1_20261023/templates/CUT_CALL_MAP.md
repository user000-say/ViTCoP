# CUT_CALL_MAP（来自真实源码，禁止按论文名称猜）
| Event | after decoder layer index (zero-based) | input→output K | module:function | scorer source layer | text query aggregation | gather/mask/position/KV call | Verified |
|---|---|---|---|---|---|---|---|
| E1 | ? | ? | ? | ? | ? | ? | UNKNOWN |
| E2 | ? | ? | ? | ? | ? | ? | UNKNOWN |
| E3 (only C1) | ? | ? | ? | ? | ? | ? | NOT_RUN |

结论：RDRIFT 究竟在 E1、E2、还是两者？必须标明 `file:line` 和调用栈，附原始 patch ID traces 和三个 smoke test。
