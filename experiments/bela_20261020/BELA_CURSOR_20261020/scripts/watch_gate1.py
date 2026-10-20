"""Poll Gate1 ACC+TTFT until complete, then write official_scores.json."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
PKG = REPO / "experiments/bela_20261020/BELA_CURSOR_20261020"
ACC = REPO / "logs/e014_aee/full/bela_gate1_8task"
TIMING = PKG / "server_results" / "timing"
LOG = PKG / "logs" / "gate1_watch.log"
ARMS = ("W2048_V5_COVER_QATTN", "W2048_V5_COVER_RDRIFT", "W1024_V4_PCOV_QATTN")
TASKS = (
    "gqa", "pope", "mme", "mmbench_en_dev",
    "scienceqa_img", "mmmu_val", "ai2d", "textvqa_val_ocr", "vizwiz_vqa_val",
)


def counts():
    acc = sum(1 for a in ARMS for t in TASKS if (ACC / a / t / "metrics.json").exists())
    ttft = sum(1 for a in ARMS if (TIMING / a / "latency_summary.json").exists())
    return acc, ttft


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    while True:
        acc, ttft = counts()
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} acc={acc}/27 ttft={ttft}/3\n"
        LOG.open("a").write(line)
        if acc >= 27 and ttft >= 3:
            break
        time.sleep(300)
    subprocess.check_call([sys.executable, str(PKG / "scripts" / "summarize_gate1.py")])
    (PKG / "state" / "gate1_complete.flag").write_text("COMPLETE\n")
    print("GATE1 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
