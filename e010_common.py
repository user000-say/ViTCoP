"""E010 runtime binding. Does not change the QED formula."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path("/home/xiangyue/ViTCoP/logs/e010_layer/20261008_173500")
RUNTIME = ROOT / "code_runtime"
E009_RUNTIME = Path("/home/xiangyue/ViTCoP/logs/e009_dualrole/20261008_130500/code_runtime")
REPO = Path("/home/xiangyue/ViTCoP")
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"

S0 = {
    "gqa": 0.5730640801399268,
    "pope_f1_score": 0.8034363379920503,
    "mme_total": 1771.3248299319728,
    "mmbench_en_dev": 62.371134020618555,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_runtime() -> None:
    llama_name = "LLaVA/llava/model/language_model/vitcop_modeling_llama.py"
    clip_name = "LLaVA/llava/model/multimodal_encoder/clip_encoder.py"
    if not (E009_RUNTIME / "LLaVA/llava/model/language_model/llava_llama.py").exists():
        raise SystemExit("E009 runtime is missing the question-span forwarder")
    if not (RUNTIME / "LLaVA/llava/model/language_model/llava_llama.py").exists():
        if RUNTIME.exists():
            shutil.rmtree(RUNTIME)
        shutil.copytree(E009_RUNTIME, RUNTIME)
    for name in (
        llama_name,
        clip_name,
        "LLaVA/llava/model/vitcop_llava_arch.py",
        "LLaVA/llava/model/e010_runtime_support.py",
        "LLaVA/llava/model/e010_context_bank.py",
    ):
        src = REPO / name
        dst = RUNTIME / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or sha256(dst) != sha256(src):
            shutil.copy2(src, dst)
    # Rewrite the recorded digest only when it changes. Concurrent jobs
    # were reading a partially written one-line hash and aborting.
    for name, filename in (
        (llama_name, "runtime_llama.sha"),
        (clip_name, "runtime_clip.sha"),
    ):
        digest = sha256(RUNTIME / name)
        recorded = ROOT / filename
        if recorded.exists() and recorded.read_text().strip() == digest:
            continue
        temporary = recorded.with_suffix(recorded.suffix + ".tmp")
        temporary.write_text(digest + "\n")
        os.replace(temporary, recorded)


def bind() -> None:
    prepare_runtime()
    import e009_common as common

    common.RUNTIME = RUNTIME
    common.LLAMA_SHA = (ROOT / "runtime_llama.sha").read_text().strip()
    common.CLIP_SHA = (ROOT / "runtime_clip.sha").read_text().strip()


def set_e010(
    *,
    enabled: bool,
    layer: int = 12,
    exit_layer: int = -1,
    keep: bool = False,
    full576: bool = False,
    side_only: bool = False,
    influence: bool = False,
    context_mode: str = "E009",
) -> None:
    os.environ["E010_ENABLE"] = "True" if enabled else "False"
    os.environ["E010_QED_LAYER"] = str(layer)
    os.environ["E010_CONTEXT_EXIT_LAYER"] = str(exit_layer)
    os.environ["E010_KEEP_CONTEXT_AFTER_QED"] = "True" if keep else "False"
    os.environ["E010_ANALYSIS_FULL576"] = "True" if full576 else "False"
    os.environ["E010_QED_SIDE_ONLY"] = "True" if side_only else "False"
    os.environ["E010_LOCAL_INFLUENCE"] = "True" if influence else "False"
    os.environ["E010_CONTEXT_MODE"] = context_mode
    os.environ["E010_CONTEXT_BANK"] = os.environ.get("E010_CONTEXT_BANK", "")


GPU4_CLAIM = ROOT / "results" / "gpu4_l6_owner.json"


def _pid_is_ancestor(pid: int) -> bool:
    cur = os.getpid()
    seen = set()
    while cur > 1 and cur not in seen:
        if cur == pid:
            return True
        seen.add(cur)
        try:
            text = Path(f"/proc/{cur}/status").read_text()
        except OSError:
            return False
        nxt = 0
        for line in text.splitlines():
            if line.startswith("PPid:"):
                nxt = int(line.split()[1])
                break
        if nxt <= 0:
            return False
        cur = nxt
    return False


def iter_claims():
    paths = []
    if GPU4_CLAIM.exists():
        paths.append(GPU4_CLAIM)
    claim_dir = ROOT / "results" / "claims"
    if claim_dir.exists():
        paths.extend(sorted(claim_dir.glob("*.json")))
    for path in paths:
        try:
            claim = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if "gpu" not in claim and path == GPU4_CLAIM:
            claim["gpu"] = 4
        claim["_path"] = str(path)
        yield claim


def gpu4_claim_owner(output: str | None = None, gpu: int | None = None) -> int:
    """Live owner pid when this process is not that owner. Zero means free."""
    for claim in iter_claims():
        pid = int(claim.get("pid") or 0)
        if pid <= 0 or not Path(f"/proc/{pid}").exists():
            continue
        if output is not None:
            owned = {str(Path(item)) for item in claim.get("outputs") or []}
            if str(Path(output)) not in owned:
                continue
        if gpu is not None and int(claim.get("gpu", -1)) != int(gpu):
            continue
        if _pid_is_ancestor(pid):
            continue
        return pid
    return 0


def wait_for_gpu4_owner() -> None:
    while True:
        pid = gpu4_claim_owner(gpu=4)
        if not pid:
            return
        print(f"[gpu4] waiting for owner pid={pid}", flush=True)
        time.sleep(30)
