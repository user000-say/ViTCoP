"""E009-FINAL runtime. QED and HREC formulas are not edited here."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/home/xiangyue/ViTCoP/logs/e009_dualrole/20261008_130500")
RUNTIME = ROOT / "code_runtime"
E006_RUNTIME = Path("/home/xiangyue/ViTCoP/logs/e006_qed_r1/code_candidate_v1")
E007_RUNTIME = Path("/home/xiangyue/ViTCoP/logs/e007_elp/20261008_011358/code_runtime")
E006_RUN = Path("/home/xiangyue/ViTCoP/logs/e006_qed_r1/20261007_194300")
E007_RUN = Path("/home/xiangyue/ViTCoP/logs/e007_elp/20261008_011358")
E008_RUN = Path("/home/xiangyue/ViTCoP/logs/e008_causal/20261008_111500")
PANEL = Path("/home/xiangyue/ViTCoP/logs/e004_b/20260924_192700/manifest/frontier_panel_128.jsonl")
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
MODEL_7B = "liuhaotian/llava-v1.5-7b"
MODEL_13B = "liuhaotian/llava-v1.5-13b"
LLAMA_SHA = "0d2546a852819ae6badcea1d6efcbfb99806c08204a9c6a42442b494d7824e8e"
CLIP_SHA = "8e5c47098ced710ec68c4a6dcee4e44ae6cf9a17fa641ba50f38e5ee07274ef5"
E007_LLAMA_SHA = "19d3d012b48e19ddd8be7c6162b801b7ef86813304798f685d8d6f547aa76c87"
E006_LLAMA_SHA = "3d6277f874ce206024627b11c27a7ca6deb7f9e33d16e1b232d51f327fac3bab"
UUIDS = {
    0: "GPU-18a68279-2877-381a-68ef-30de98314f23",
    1: "GPU-d1ee3c7d-8137-5265-13f5-33cd54181ee7",
    2: "GPU-fded14bf-1920-7ddc-58da-c70411e0bb62",
    3: "GPU-0b36ade1-4c1f-8cd6-72e4-282769531f1b",
    4: "GPU-011a2a1a-2fe7-3d52-73f9-3ff636bd1ed2",
    5: "GPU-444947e2-11d4-6982-43dd-55206f4a369f",
    6: "GPU-40027d6e-49f8-a925-cbe0-49e426de4a4c",
    7: "GPU-5774ecc0-d45c-b8e8-dc39-732bf617a41e",
}
TASKS = (
    "gqa",
    "pope",
    "mme",
    "mmbench_en_dev",
    "mmbench_cn_dev",
    "scienceqa_img",
    "textvqa_val",
    "textvqa_val_ocr",
    "vqav2_val",
    "vqav2_val_lite",
)
# Counts filled from local Arrow / lmms-eval splits; update if cache rebuilt.
FULL_N = {
    "gqa": 12578,
    "pope": 9000,
    "mme": 2374,
    "mmbench_en_dev": 4329,
    "mmbench_cn_dev": 4329,  # same MMBench-CN dev size as EN in lmms-lab
    "scienceqa_img": 2017,
    "textvqa_val": 5000,
    "textvqa_val_ocr": 5000,
    "vqav2_val": 214354,
    "vqav2_val_lite": 500,
}
# Frozen before any E009 full-accuracy metric was read.
STRONG_MARGIN_PP = 0.25
RETENTION_STRONG = 0.75
RETENTION_FAIL = 0.50
# A single benchmark collapses when its S0-relative change is at or below this.
COLLAPSE_PP = -1.0
# Null explains the gain when its macro effect is at least this share of HREC's.
NULL_LARGE_FRACTION = 0.50
# Latency is acceptable when both endpoints stay within this paired overhead vs S0.
LATENCY_ACCEPT_PERCENT = 5.0

E009_ARMS = ("NULL188_CONTEXT", "RANDOM188_CONTEXT", "RAW188_CONTEXT", "HREC16_CONTEXT")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hard(message: str) -> None:
    print("E009_FUNCTIONAL_HARD_FAIL " + message, flush=True)
    raise SystemExit(99)


def apply_profile(profile: str, *, audit: bool = False, strict: bool = True) -> None:
    from e006_env import _common

    _common()
    e009 = profile in E009_ARMS or profile in {"HREC64", "HREC96_13B"}
    if profile == "ANCHOR_A":
        stage1, visual, selector, target_k, layers = "ORIGINAL_172", 172, "QED", 96, 32
        e007 = True
    elif profile == "ANCHOR_D":
        stage1, visual, selector, target_k, layers = "HR_TOPK_192", 192, "QED", 96, 32
        e007 = True
    elif profile in E009_ARMS or profile == "HREC64":
        stage1, visual, selector, layers = "ORIGINAL_172", 188, "QED", 32
        target_k = 64 if profile == "HREC64" else 96
        e007 = True
    elif profile == "HREC96_13B":
        stage1, visual, selector, target_k, layers = "ORIGINAL_172", 188, "QED", 96, 40
        e007 = True
    elif profile == "G1_Q_ATTN":
        stage1, visual, selector, target_k, layers, e007 = "ORIGINAL_172", 172, "Q_ATTN", 96, 32, False
    elif profile == "G2_Q_RESP":
        stage1, visual, selector, target_k, layers, e007 = "ORIGINAL_172", 172, "Q_RESP", 96, 32, False
    elif profile == "G4_QED_VISUAL":
        stage1, visual, selector, target_k, layers, e007 = "ORIGINAL_172", 172, "QED_VISUAL", 96, 32, False
    elif profile == "CORE_QED64":
        stage1, visual, selector, target_k, layers, e007 = "ORIGINAL_172", 172, "QED", 64, 32, True
    elif profile == "HR192_QED64":
        stage1, visual, selector, target_k, layers, e007 = "HR_TOPK_192", 192, "QED", 64, 32, True
    elif profile == "S0":
        stage1, visual, selector, target_k, layers, e007 = "ORIGINAL_172", 576, "QED", 96, 32, False
    else:
        hard(f"profile {profile}")
    mode = "HREC16_CONTEXT" if profile in {"HREC64", "HREC96_13B"} else profile
    os.environ.update(
        USE_VITCOP="True",
        USE_DEBIAS="False",
        USE_COORD_TOKENS="False",
        USE_SPATIAL_AWARE="False",
        USE_DYNAMIC_RATIO="False",
        USE_ADAPTIVE_SELECT="False",
        USE_ZONE_ROUTING="False",
        USE_STAGE2_SRE="False",
        USE_STAGE2_LITE_ROUTER="False",
        USE_STAGE2_WEIGHTED_MERGE="False",
        USE_T3GUIDE="False",
        SHALLOW_PRUNED_LAYER="2" if profile == "S0" else "12",
        DEEP_PRUNED_LAYER="22",
        STAGE3_POLICY="original" if profile == "S0" else "skip",
        E006_ENABLE="False" if profile == "S0" else "True",
        E006_SELECTOR=selector,
        E006_LAYER="12",
        E006_TARGET_K=str(target_k),
        E006_ALLOW_FALLBACK_64="True" if target_k == 64 and not e009 else "False",
        E006_REQUIRE_EXPLICIT_QUESTION_SPAN="False" if profile == "S0" else "True",
        E006_PROFILE="False",
        E006_DEBUG="False",
        E006_EXPECT_LAYERS=str(layers),
        E001_METHOD_ID="E009",
        E001_TOKEN_AUDIT="False",
        E001_WARMUP_REQUESTS="0",
        E007_ENABLE="True" if e007 else "False",
        E007_STAGE1_MODE=stage1,
        E007_EXPECT_STAGE1=str(visual if visual != 188 else 172),
        E007_SKIP_UNUSED_VIC="True",
        E007_TOKEN_AUDIT="False",
        E008_ENABLE="False",
        E009_ENABLE="True" if e009 else "False",
        E009_MODE=mode if e009 else "HREC16_CONTEXT",
        E009_AUDIT="True" if audit else "False",
        E009_STRICT_CHECK="True" if (e009 and strict) else "False",
        E009_EXPECT_VISUAL=str(visual),
        E009_PRETRAINED=MODEL_13B if layers == 40 else MODEL_7B,
    )
    if profile == "HREC64":
        os.environ["E009_ALLOW_REGISTERED_K64"] = "True"
    else:
        os.environ.pop("E009_ALLOW_REGISTERED_K64", None)
    import e006_span

    e006_span.STAGE1_VISUAL = visual if profile != "S0" else 172


def expected_llama_sha() -> str:
    recorded = ROOT / "runtime_llama.sha"
    if os.environ.get("E009_ALLOW_REGISTERED_K64") == "True" and recorded.exists():
        return recorded.read_text().strip()
    return LLAMA_SHA


def configure_runtime_constants() -> None:
    import e006_env

    e006_env.SOURCE = str(RUNTIME)
    e006_env.EXPECTED_MODELING_SHA256 = expected_llama_sha()


def install_runtime_audit() -> None:
    import e006_lmms_patch as patch

    expected = int(os.environ["E009_EXPECT_VISUAL"])
    target_k = int(os.environ["E006_TARGET_K"])
    layers = int(os.environ["E006_EXPECT_LAYERS"])
    s0 = os.environ.get("E006_ENABLE") == "False"

    def _audit(raw_model) -> None:
        core = raw_model.get_model()
        last = getattr(core, "e006_last", {}) or {}
        reasons = []
        stage1 = int(getattr(core, "e001_last_stage1_visual_tokens", -1))
        if not s0 and stage1 != expected:
            reasons.append(f"stage1={stage1}")
        if int(core.config.num_hidden_layers) != layers:
            reasons.append(f"layers={core.config.num_hidden_layers}")
        if core.config._attn_implementation != "sdpa":
            reasons.append(f"attention={core.config._attn_implementation}")
        if bool(core.config.output_attentions):
            reasons.append("output_attentions")
        if not s0:
            if core.stage3_policy != "skip":
                reasons.append(f"stage3={core.stage3_policy}")
            final = int(getattr(core, "e001_last_final_visual_tokens", -1))
            if final != target_k:
                reasons.append(f"final={final}")
            visual_in = int(last.get("visual_in") or -1)
            if visual_in != expected:
                reasons.append(f"visual_entering_layer12={visual_in}")
            if int(last.get("prefill_count", -1)) != 1 or int(last.get("decode_count", -1)) != 0:
                reasons.append(f"prefill={last.get('prefill_count')} decode={last.get('decode_count')}")
            if last.get("span_source") != "explicit":
                reasons.append(f"span_source={last.get('span_source')}")
            if not bool(last.get("sdpa_preserved", getattr(core, "_use_sdpa", False))):
                reasons.append("sdpa_not_preserved")
        if reasons:
            patch._hard("; ".join(reasons))
        patch._JOB_AUDIT["requests_audited"] += 1
        patch._JOB_AUDIT["stage1_tokens"] = stage1
        patch._JOB_AUDIT["visual_entering_layer12"] = int(last.get("visual_in") or -1)
        patch._JOB_AUDIT["final_visual_tokens"] = target_k
        patch._JOB_AUDIT["question_span_source"] = last.get("span_source")
        patch._JOB_AUDIT["selector_prefill_count"] = int(last.get("prefill_count", -1))
        patch._JOB_AUDIT["selector_decode_count"] = int(last.get("decode_count", -1))
        patch._JOB_AUDIT["num_hidden_layers"] = layers
        patch._JOB_AUDIT["attention_backend"] = "sdpa"
        patch._JOB_AUDIT["stage3_policy"] = str(core.stage3_policy)

    patch._audit = _audit


def load_model(gpu_index: int):
    configure_runtime_constants()
    llama = RUNTIME / "LLaVA/llava/model/language_model/vitcop_modeling_llama.py"
    clip = RUNTIME / "LLaVA/llava/model/multimodal_encoder/clip_encoder.py"
    if sha256(clip) != CLIP_SHA or sha256(llama) != expected_llama_sha():
        hard(f"runtime hash llama={sha256(llama)} clip={sha256(clip)}")
    frozen = E006_RUNTIME / "LLaVA/llava/model/language_model/vitcop_modeling_llama.py"
    e007 = E007_RUNTIME / "LLaVA/llava/model/language_model/vitcop_modeling_llama.py"
    if sha256(frozen) != E006_LLAMA_SHA or sha256(e007) != E007_LLAMA_SHA:
        hard("E006 or E007 runtime tree was modified")
    from e006_model import assert_physical_gpu, load_13b

    layers = int(os.environ["E006_EXPECT_LAYERS"])
    model_id = os.environ["E009_PRETRAINED"]
    uuid = assert_physical_gpu(gpu_index, UUIDS[gpu_index])
    name = "llava-v1.5-13b" if layers == 40 else "llava-v1.5-7b"
    eng, lm, raw, core, torch = load_13b(model_id, layers, name)
    return uuid, eng, lm, raw, core, torch


def panel_lookup(eng, lm):
    panel = [json.loads(line) for line in PANEL.read_text().splitlines() if line.strip()]
    if len(panel) != 128:
        hard(f"panel {len(panel)}")
    per, tasks = {}, []
    for row in panel:
        per[row["dataset"]] = per.get(row["dataset"], 0) + 1
        if row["dataset"] not in tasks:
            tasks.append(row["dataset"])
    reqs, meta, _counts = eng._load_tasks(lm, tasks, max(per.values()))
    lookup = {}
    for task in tasks:
        for req, item in zip(reqs[task], meta[task]):
            lookup[(task, str(item["sample_id"]), int(item["sample_index"]))] = req
    keys = [(row["dataset"], str(row["sample_id"]), int(row["sample_index"])) for row in panel]
    missing = [key for key in keys if key not in lookup]
    if missing:
        hard(f"panel missing {missing[:3]}")
    return lookup, keys
