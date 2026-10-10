# E013 accuracy policy (user lock, 2026-10-09)

**Hard rule:** Any accuracy comparison used for H1/H2 / arm promotion / paper claims
MUST use the official full four-dataset splits:

| task | N |
|---|---|
| gqa | 12578 |
| pope | 9000 (report F1 + accuracy) |
| mme | 2374 (perception + cognition) |
| mmbench_en_dev | 4329 |

**Forbidden as ACC evidence:** panel/limit=512 (or any partial subset). Those runs are
diagnostics only (smoke plumbing, rough trends). They do **not** represent accuracy.

Full compare arms (controls included): BASE_S8, B0_S8, REPLACE_{RANDOM,ATTN,COVER}16,
EXTRA_{RANDOM,ATTN,COVER,NULL,DUPLICATE}16.
