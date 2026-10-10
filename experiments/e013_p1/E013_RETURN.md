# E013_RETURN
generated_utc: 2026-10-10T02:36:30.273071+00:00

## 1) Source
- git: `627f63214ef9c67158beb7744d7dff087d43e6df`
- live llama SHA: `6a4f4c31334fa73f9e99395c93a6bff64e60c8b7b93f951f8e54c8c804651896`
- live clip SHA: `b89208803528019460a47100fb5984e178c85cf12ca5f8ddb2981fda420ea73c`
- live e013_stage1_support SHA: `e6704f81345b26cc01a7bb138ff26cd4c7b7d1fac9b533a2d2071375dedf9341`
- torch/transformers/CUDA: `2.1.2 4.37.2 11.8`
- GPUs:
```
0, GPU-18a68279-2877-381a-68ef-30de98314f23, Tesla V100-PCIE-32GB
1, GPU-d1ee3c7d-8137-5265-13f5-33cd54181ee7, Tesla V100-PCIE-32GB
2, GPU-fded14bf-1920-7ddc-58da-c70411e0bb62, Tesla V100-PCIE-32GB
3, GPU-0b36ade1-4c1f-8cd6-72e4-282769531f1b, Tesla V100-PCIE-32GB
4, GPU-011a2a1a-2fe7-3d52-73f9-3ff636bd1ed2, Tesla V100-PCIE-32GB
5, GPU-444947e2-11d4-6982-43dd-55206f4a369f, Tesla V100-PCIE-32GB
6, GPU-40027d6e-49f8-a925-cbe0-49e426de4a4c, Tesla V100-PCIE-32GB
7, GPU-5774ecc0-d45c-b8e8-dc39-732bf617a41e, Tesla V100-PCIE-32GB
```
- runner: `e013_boot.py` → `e010_eval_job.py` / lmms_eval; latency `e013_latency.py` timed region=`raw.generate`; input_policy=`model_inference`
- ACC policy: full four-dataset only (user). Latency gate: Fixed32 |rel Δ| ≤ 15% (user).
- backup: `/home/xiangyue/ViTCoP/logs/e013_p1/backups/20261009T140258Z`

## 2) Install & smoke
- SMOKE_STATUS: `PASS`
- COMPARE: `{"status": "PASS", "fails": []}`
- logs: `logs/e013_p1/smoke_driver.log`, `logs/e013_p1/smoke/*/smoke.json`

## 3) Panel (diagnostic only — NOT ACC evidence)
- PANEL_SUMMARY present: `True`
- screen1/screen2 metrics under `logs/e013_p1/panel/`; N=512 per task; do not use for H1/H2.

## 4) Full official metrics
- FULL_ACCURACY_TABLE: `/home/xiangyue/ViTCoP/logs/e013_p1/FULL_ACCURACY_TABLE.json`
- complete: `True`
- missing: []
- bad_n: []
```json
{
  "BASE_S8": {
    "arm": "BASE_S8",
    "gqa": {
      "exact_match": 0.5878518047384322,
      "effective": 12578,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/BASE_S8/gqa/metrics.json"
    },
    "pope": {
      "pope_accuracy": 0.861,
      "pope_f1_score": 0.8483452539701781,
      "effective": 9000,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/BASE_S8/pope/metrics.json"
    },
    "mme": {
      "mme_perception_score": 1452.8056222488995,
      "mme_cognition_score": 324.6428571428571,
      "mme_total": 1777.4484793917566,
      "effective": 2374,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/BASE_S8/mme/metrics.json"
    },
    "mmbench_en_dev": {
      "gpt_eval_score": 63.4020618556701,
      "effective": 4329,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/BASE_S8/mmbench_en_dev/metrics.json"
    }
  },
  "B0_S8": {
    "arm": "B0_S8",
    "gqa": {
      "exact_match": 0.5878518047384322,
      "effective": 12578,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/B0_S8/gqa/metrics.json"
    },
    "pope": {
      "pope_accuracy": 0.861,
      "pope_f1_score": 0.8483452539701781,
      "effective": 9000,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/B0_S8/pope/metrics.json"
    },
    "mme": {
      "mme_perception_score": 1452.8056222488995,
      "mme_cognition_score": 324.6428571428571,
      "mme_total": 1777.4484793917566,
      "effective": 2374,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/B0_S8/mme/metrics.json"
    },
    "mmbench_en_dev": {
      "gpt_eval_score": 63.4020618556701,
      "effective": 4329,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/B0_S8/mmbench_en_dev/metrics.json"
    }
  },
  "REPLACE_RANDOM16": {
    "arm": "REPLACE_RANDOM16",
    "gqa": {
      "exact_match": 0.5876132930513596,
      "effective": 12578,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/REPLACE_RANDOM16/gqa/metrics.json"
    },
    "pope": {
      "pope_accuracy": 0.8615555555555555,
      "pope_f1_score": 0.848639455782313,
      "effective": 9000,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/REPLACE_RANDOM16/pope/metrics.json"
    },
    "mme": {
      "mme_perception_score": 1442.5793317326932,
      "mme_cognition_score": 337.1428571428571,
      "mme_total": 1779.7221888755503,
      "effective": 2374,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/REPLACE_RANDOM16/mme/metrics.json"
    },
    "mmbench_en_dev": {
      "gpt_eval_score": 63.31615120274914,
      "effective": 4329,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/REPLACE_RANDOM16/mmbench_en_dev/metrics.json"
    }
  },
  "REPLACE_ATTN16": {
    "arm": "REPLACE_ATTN16",
    "gqa": {
      "exact_match": 0.5880903164255048,
      "effective": 12578,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/REPLACE_ATTN16/gqa/metrics.json"
    },
    "pope": {
      "pope_accuracy": 0.8648888888888889,
      "pope_f1_score": 0.8528201404018397,
      "effective": 9000,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/REPLACE_ATTN16/pope/metrics.json"
    },
    "mme": {
      "mme_perception_score": 1457.482092837135,
      "mme_cognition_score": 321.7857142857143,
      "mme_total": 1779.2678071228493,
      "effective": 2374,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/REPLACE_ATTN16/mme/metrics.json"
    },
    "mmbench_en_dev": {
      "gpt_eval_score": 62.97250859106529,
      "effective": 4329,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/REPLACE_ATTN16/mmbench_en_dev/metrics.json"
    }
  },
  "REPLACE_COVER16": {
    "arm": "REPLACE_COVER16",
    "gqa": {
      "exact_match": 0.5900779138177771,
      "effective": 12578,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/REPLACE_COVER16/gqa/metrics.json"
    },
    "pope": {
      "pope_accuracy": 0.8644444444444445,
      "pope_f1_score": 0.8523359961268457,
      "effective": 9000,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/REPLACE_COVER16/pope/metrics.json"
    },
    "mme": {
      "mme_perception_score": 1467.8701480592238,
      "mme_cognition_score": 331.07142857142856,
      "mme_total": 1798.9415766306524,
      "effective": 2374,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/REPLACE_COVER16/mme/metrics.json"
    },
    "mmbench_en_dev": {
      "gpt_eval_score": 63.144329896907216,
      "effective": 4329,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/REPLACE_COVER16/mmbench_en_dev/metrics.json"
    }
  },
  "EXTRA_RANDOM16": {
    "arm": "EXTRA_RANDOM16",
    "gqa": {
      "exact_match": 0.5895213865479408,
      "effective": 12578,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_RANDOM16/gqa/metrics.json"
    },
    "pope": {
      "pope_accuracy": 0.8621111111111112,
      "pope_f1_score": 0.850138872116894,
      "effective": 9000,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_RANDOM16/pope/metrics.json"
    },
    "mme": {
      "mme_perception_score": 1457.5978391356543,
      "mme_cognition_score": 304.2857142857143,
      "mme_total": 1761.8835534213686,
      "effective": 2374,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_RANDOM16/mme/metrics.json"
    },
    "mmbench_en_dev": {
      "gpt_eval_score": 63.745704467353946,
      "effective": 4329,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_RANDOM16/mmbench_en_dev/metrics.json"
    }
  },
  "EXTRA_ATTN16": {
    "arm": "EXTRA_ATTN16",
    "gqa": {
      "exact_match": 0.5883288281125775,
      "effective": 12578,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_ATTN16/gqa/metrics.json"
    },
    "pope": {
      "pope_accuracy": 0.8632222222222222,
      "pope_f1_score": 0.8513823493903175,
      "effective": 9000,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_ATTN16/pope/metrics.json"
    },
    "mme": {
      "mme_perception_score": 1452.9616846738697,
      "mme_cognition_score": 320.0,
      "mme_total": 1772.9616846738697,
      "effective": 2374,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_ATTN16/mme/metrics.json"
    },
    "mmbench_en_dev": {
      "gpt_eval_score": 63.487972508591064,
      "effective": 4329,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_ATTN16/mmbench_en_dev/metrics.json"
    }
  },
  "EXTRA_COVER16": {
    "arm": "EXTRA_COVER16",
    "gqa": {
      "exact_match": 0.5896803943393226,
      "effective": 12578,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_COVER16/gqa/metrics.json"
    },
    "pope": {
      "pope_accuracy": 0.8626666666666667,
      "pope_f1_score": 0.8505441354292623,
      "effective": 9000,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_COVER16/pope/metrics.json"
    },
    "mme": {
      "mme_perception_score": 1462.7558023209283,
      "mme_cognition_score": 330.3571428571429,
      "mme_total": 1793.1129451780712,
      "effective": 2374,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_COVER16/mme/metrics.json"
    },
    "mmbench_en_dev": {
      "gpt_eval_score": 63.487972508591064,
      "effective": 4329,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_COVER16/mmbench_en_dev/metrics.json"
    }
  },
  "EXTRA_NULL16": {
    "arm": "EXTRA_NULL16",
    "gqa": {
      "exact_match": 0.5765622515503259,
      "effective": 12578,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_NULL16/gqa/metrics.json"
    },
    "pope": {
      "pope_accuracy": 0.8558888888888889,
      "pope_f1_score": 0.841112336150925,
      "effective": 9000,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_NULL16/pope/metrics.json"
    },
    "mme": {
      "mme_perception_score": 1409.592236894758,
      "mme_cognition_score": 327.8571428571429,
      "mme_total": 1737.4493797519008,
      "effective": 2374,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_NULL16/mme/metrics.json"
    },
    "mmbench_en_dev": {
      "gpt_eval_score": 60.39518900343642,
      "effective": 4329,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_NULL16/mmbench_en_dev/metrics.json"
    }
  },
  "EXTRA_DUPLICATE16": {
    "arm": "EXTRA_DUPLICATE16",
    "gqa": {
      "exact_match": 0.5888058514867228,
      "effective": 12578,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_DUPLICATE16/gqa/metrics.json"
    },
    "pope": {
      "pope_accuracy": 0.8597777777777778,
      "pope_f1_score": 0.8460975609756097,
      "effective": 9000,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_DUPLICATE16/pope/metrics.json"
    },
    "mme": {
      "mme_perception_score": 1432.8473389355743,
      "mme_cognition_score": 330.0,
      "mme_total": 1762.8473389355743,
      "effective": 2374,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_DUPLICATE16/mme/metrics.json"
    },
    "mmbench_en_dev": {
      "gpt_eval_score": 63.144329896907216,
      "effective": 4329,
      "path": "/home/xiangyue/ViTCoP/logs/e013_p1/full/full/EXTRA_DUPLICATE16/mmbench_en_dev/metrics.json"
    }
  }
}
```

## 5) ABBA crossover
- gate: `{
  "status": "PASS",
  "rule": "user_relative_15pct_fixed32",
  "max_rel_percent": 15.0,
  "e012_minus_legacy_pair1_rel_mean_ci": [
    0.04339791279983074,
    0.03925407388198627,
    0.04754175171767521
  ],
  "e012_minus_legacy_pair2_rel_mean_ci": [
    0.0013037792711361722,
    -0.002258158952822764,
    0.0048657174950951084
  ],
  "e012_minus_legacy_pair1_ms_mean": -52.46857137812499,
  "e012_minus_legacy_pair2_ms_mean": -1.4686637718750113,
  "legacy_ab_stability_rel_mean_ci": [
    0.07007755682918654,
    0.06718177219221608,
    0.072973341466157
  ],
  "e012_ab_stability_rel_mean_ci": [
    0.027167329049227803,
    0.02262980528551024,
    0.03170485281294537
  ],
  "legacy_a_mean_fixed32_ms": 1209.59861295625,
  "e012_a_mean_fixed32_ms": 1262.067184334375,
  "verdict": "|rel Fixed32 mean| <= 15.0% on both ABBA pairs; Stage1 latency attribution allowed"
}`
- latency dirs: ['BASE_S8', 'E012_S12D12_A', 'E012_S12D12_B', 'EXTRA_ATTN16', 'EXTRA_COVER16', 'LEGACY_L12_A', 'LEGACY_L12_B', 'REPLACE_ATTN16', 'REPLACE_COVER16']
- rule: user 15% relative Fixed32 vs LEGACY

## 6) Stage1 latency
- done arms: ['BASE_S8', 'REPLACE_ATTN16', 'REPLACE_COVER16', 'EXTRA_ATTN16', 'EXTRA_COVER16']
- required after crossover PASS: BASE_S8 REPLACE_ATTN16 REPLACE_COVER16 EXTRA_ATTN16 EXTRA_COVER16
- threshold vs BASE: user ≤15% Fixed32 relative overhead for acceptability notes

## 7) Decision
- MASTER_STATUS: `PASS`
- H1=reject/inconclusive (COVER vs BASE/RANDOM/ATTN on full GQA+POPE F1; also inspect MME/MMBench in table). H2=reject/inconclusive (COVER vs RANDOM/ATTN/NULL/DUPLICATE). COVER is a test heuristic — no algorithmic novelty claim.

## 8) Rollback / archive
- rollback: `bash e013_rollback.sh $(cat logs/e013_p1/ACTIVE_BACKUP_PATH.txt)`
- return path: `/home/xiangyue/ViTCoP/logs/e013_p1/E013_RETURN.md`
- next: only if H2 accepts on full metrics → E014 counterfactual utility predictor; else report soft negative.
