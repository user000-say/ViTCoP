# E020.1 R2 Incident Log

Auto-recorded pipeline failures during collect/fit/learned ACC.
Format: timestamp · symptom · root cause · impact on prior ACC · fix · status.

---

## INC-001 — CUDA_VISIBLE_DEVICES polluted by log capture
- **When:** 2026-10-17T06:55:37+08:00
- **Symptom:** `SMOKE FAILED rc=1` within ~4s; log showed
  `collect/fit on physical GPU=[2026-10-17T06:55:36+08:00] wait GPUs...`
  and torch reported `cuda False` / no device.
- **Root cause:** `log()` used `tee -a "$LOG"` (stdout).  
  `GPU=$(wait_gpus_free)` therefore captured log lines into `CUDA_VISIBLE_DEVICES`,
  so the device string was garbage (timestamps + free_gpus text), not `0`/`2`/`4`/`6`.
- **Impact on prior ACC:** **None.** E019 remaining (144/144) and E020 priority (104/104)
  launch via `e014_queue` fill scripts with explicit `export CUDA_VISIBLE_DEVICES=<int>`.
  Only the post-E019 R2 smoke path in `orchestrate_e019_then_r2.sh` was hit.
- **Fix:** `log(){ … | tee -a "$LOG" >&2; }` so command substitutions never see logs;
  resume script sets `CUDA_VISIBLE_DEVICES` from `pick_gpu` only.
- **Status:** fixed

## INC-002 — USE_VITCOP / STAGE3_POLICY not set before model construct
- **When:** 2026-10-17T11:33–11:36
- **Symptom:** `ValueError: E018 CLIP adapter requires one tensor image and USE_VITCOP=True`
  then `ValueError: [E006] STAGE3_POLICY must be skip, got original`
- **Root cause:** `CLIPVisionTower.use_vitcop` and Llama `STAGE3_POLICY` are read at
  `__init__`. `apply_arm_env` ran only inside `run_teacher_forced`, after load.
- **Impact on prior ACC:** **None.** ACC path uses `e014_boot` → `apply_profile` →
  `apply_arm_env` before eval; constructors see correct env.
- **Fix:** call `apply_arm_env("E020_COVER_FIRST_SHIFT_REF32")` in
  `ProjectE020Backend.__init__` before `load_pretrained_model`.
- **Status:** fixed

## INC-003 — generate_process_count missing on teacher-forced decode
- **When:** 2026-10-17T11:37
- **Symptom:** After model load + first prefill OK,
  `AttributeError: 'LlavaLlamaModel' object has no attribute 'generate_process_count'`
  on answer-token decode steps (`images=None`, `seq_len==1`).
- **Root cause:** Normal ACC uses `model.generate()`, which zeros
  `self.model.generate_process_count`. Pairdata teacher-forced path calls `forward`
  directly and never initializes that counter.
- **Impact on prior ACC:** **None.**
- **Fix:** set `core.generate_process_count = 0` (and default image_shape lists)
  before the teacher-forced forward in `project_e020_backend.py`.
- **Status:** fixed (restarting smoke)

---

## Ops rule (going forward)
On any smoke/collect/fit failure: append an INC entry, auto-triage env vs code,
apply the smallest fix, restart the pipeline, and report — do not leave the campaign idle.

## AUTO 2026-10-17T11:40:24+08:00 — SMOKE attempt 1 rc=1
```
[e0201r2_backend] loading liuhaotian/llava-v1.5-7b on cuda:0
/home/xiangyue/envs/vitcopreback/vitcopreback/lib/python3.10/site-packages/huggingface_hub/file_download.py:949: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
You are using a model of type llava to instantiate a model of type llava_llama. This is not supported for all configurations of models and can yield errors.

  return self.fget.__get__(instance, owner)()

[e0201r2_backend] 256 calibration examples
Traceback (most recent call last):
  File "/home/xiangyue/ViTCoP/experiments/e020_1_r2/pairdata.py", line 166, in <module>
    print(json.dumps(collect(adapter,args.out,max_states=args.states,actions_per_state=args.actions,seed=args.seed,
  File "/home/xiangyue/ViTCoP/experiments/e020_1_r2/pairdata.py", line 87, in collect
    duplicate=checked_branch(adapter,s,ids[selected],prefix,force_recompute=True)
  File "/home/xiangyue/ViTCoP/experiments/e020_1_r2/pairdata.py", line 46, in checked_branch
    raise ValueError('invalid branch NLL or mismatched actual prefix')
ValueError: invalid branch NLL or mismatched actual prefix
```

## AUTO 2026-10-17T11:40:51+08:00 — SMOKE attempt 2 rc=1
```
[e0201r2_backend] loading liuhaotian/llava-v1.5-7b on cuda:0
/home/xiangyue/envs/vitcopreback/vitcopreback/lib/python3.10/site-packages/huggingface_hub/file_download.py:949: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
You are using a model of type llava to instantiate a model of type llava_llama. This is not supported for all configurations of models and can yield errors.

  return self.fget.__get__(instance, owner)()

[e0201r2_backend] 256 calibration examples
Traceback (most recent call last):
  File "/home/xiangyue/ViTCoP/experiments/e020_1_r2/pairdata.py", line 166, in <module>
    print(json.dumps(collect(adapter,args.out,max_states=args.states,actions_per_state=args.actions,seed=args.seed,
  File "/home/xiangyue/ViTCoP/experiments/e020_1_r2/pairdata.py", line 87, in collect
    duplicate=checked_branch(adapter,s,ids[selected],prefix,force_recompute=True)
  File "/home/xiangyue/ViTCoP/experiments/e020_1_r2/pairdata.py", line 46, in checked_branch
    raise ValueError('invalid branch NLL or mismatched actual prefix')
ValueError: invalid branch NLL or mismatched actual prefix
```

## AUTO 2026-10-17T11:41:17+08:00 — SMOKE attempt 3 rc=1
```
[e0201r2_backend] loading liuhaotian/llava-v1.5-7b on cuda:0
/home/xiangyue/envs/vitcopreback/vitcopreback/lib/python3.10/site-packages/huggingface_hub/file_download.py:949: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
You are using a model of type llava to instantiate a model of type llava_llama. This is not supported for all configurations of models and can yield errors.

  return self.fget.__get__(instance, owner)()

[e0201r2_backend] 256 calibration examples
Traceback (most recent call last):
  File "/home/xiangyue/ViTCoP/experiments/e020_1_r2/pairdata.py", line 166, in <module>
    print(json.dumps(collect(adapter,args.out,max_states=args.states,actions_per_state=args.actions,seed=args.seed,
  File "/home/xiangyue/ViTCoP/experiments/e020_1_r2/pairdata.py", line 87, in collect
    duplicate=checked_branch(adapter,s,ids[selected],prefix,force_recompute=True)
  File "/home/xiangyue/ViTCoP/experiments/e020_1_r2/pairdata.py", line 46, in checked_branch
    raise ValueError('invalid branch NLL or mismatched actual prefix')
ValueError: invalid branch NLL or mismatched actual prefix
```

## INC-004 — prefix_hash included selected/force (broke no-op + paired branches)
- **When:** 2026-10-17T11:40–11:41
- **Symptom:** `ValueError: invalid branch NLL or mismatched actual prefix` at
  `force_recompute=True` no-op in `pairdata.checked_branch`.
- **Root cause:** Backend hashed `selected` + `force` into `prefix_hash`. Natural
  baseline has `force=None`; recomputed same keep has `force=<ids>` → hash mismatch.
  Same bug would break every alternative keep branch (pairdata requires identical
  prefix_hash across branches).
- **Impact on prior ACC:** **None** (collect never produced models; ACC unused).
- **Fix:** Hash only prompt/checkpoint identity: image_id, question_id, schedule,
  stage1, patch_ids (pre-choice universe).
- **Status:** fixed; restarting smoke

## INC-005 — learned ACC arm parse failed on multi-underscore kinds
- **When:** 2026-10-17T11:55–12:49
- **Symptom:** `full_e020_learned` queue `COMPLETE_WITH_FAILURES done=0 failed=112`.
  Every job: `Unknown E020 schedule in arm …: EARLY_CFFULL_G1R4` (etc.).
- **Root cause:** `apply_arm_env` took `parts[-1]` as kind. Kind keys like
  `CFFULL_G1R4` contain `_`, so schedule absorbed the kind tokens.
- **Impact:** Collect/fit OK; learned ACC produced **no** metrics. Prior E019/E020
  priority ACC unaffected (no CFFULL_* kinds).
- **Fix:** Longest-suffix match of `_E020_KINDS` when parsing E020 arms.
- **Status:** fixed; clearing attempts and relaunching `full_e020_learned`

## INC-006 — learned ACC rejected G1R4 vs calibrated G1R1 policy
- **When:** 2026-10-17T15:21
- **Symptom:** `deployment differs from calibrated policy: rounds` (then same for all 112).
- **Root cause:** R2 fit stores `DEFAULT_POLICY` rounds=1/group_size=1; arms still named
  `CFFULL_G1R4` / `G4R2` and env set rounds=4. Dispatch correctly refuses mismatch.
  Also `accept_enabled`/`threshold` were not copied from checkpoint into the hook spec.
- **Impact:** Collect/fit unchanged and valid. Learned ACC still 0 metrics.
- **Fix:** Continuation kinds → `*_G1R1`; `apply_arm_env` freezes rounds/threshold/accept
  from npz; `_e020_from_env` passes `candidate_limit` + `accept_enabled`.
- **Status:** fixed; relaunching `full_e020_learned` (new arm names)

## INC-007 — PairScorer CPU buffers vs CUDA features
- **When:** 2026-10-17T15:25
- **Symptom:** After G1R1 policy fix, `RuntimeError: Expected all tensors to be on the same device, cuda:0 and cpu!` in `PairScorer.forward`.
- **Root cause:** Checkpoint buffers registered on CPU; token features from LLaVA are CUDA.
- **Fix:** `PairScorer.forward` moves module to `pairs.device/dtype` on first use.
- **Status:** fixed; clearing attempts and relaunching
