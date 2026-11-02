"""Read-only V100 CUDA/memory/model-config compatibility probe.

This probe does not reserve GPUs or download model checkpoints.
Call directly on target V100 host before any run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

def parse_ids(value):
    if not re.fullmatch(r"\d+(?:,\d+){3}",value):
        raise ValueError("Exactly FOUR explicit physical GPU IDs required (e.g., 0,2,4,7)")
    ids=[int(x) for x in value.split(",")]
    if len(set(ids))!=4:
        raise ValueError("Repeated GPU IDs")
    return ids

def gpu_inventory():
    exe=shutil.which("nvidia-smi")
    if not exe:
        raise RuntimeError("nvidia-smi unavailable; run this on CUDA V100 server")
    query=["index","name","memory.total","memory.free"]
    output=subprocess.run([exe,"--query-gpu="+",".join(query),"--format=csv,noheader,nounits"],
                          check=True,capture_output=True,text=True,timeout=20)
    gpu={}
    for line in output.stdout.splitlines():
        vals=[v.strip() for v in line.split(",")]
        if len(vals)!=4: raise RuntimeError("Unexpected nvidia-smi output "+line)
        i=int(vals[0])
        gpu[i]=dict(index=i,name=vals[1],total_mib=int(vals[2]),free_mib=int(vals[3]))
    return gpu

def model_info(config_path,expect_layers):
    if not config_path:
        return {"status":"CHECK_MODEL_CONFIG_REQUIRED", "expected_layers":expect_layers}
    p=Path(config_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Missing model config.json: {p}")
    conf=json.loads(p.read_text(encoding="utf-8"))
    tc=conf.get("text_config",{}) or {}
    layers=tc.get("num_hidden_layers",conf.get("num_hidden_layers"))
    if layers is None or int(layers)!=expect_layers:
        raise ValueError(f"Model layer mismatch {layers}, expected {expect_layers}")
    model_type=conf.get("model_type","unknown")
    model_arch=conf.get("architectures",[])
    return {
        "status":"VERIFIED_CONFIG_LAYERS",
        "model_type":model_type,"architectures":model_arch,
        "layers":int(layers),
        "image_grid_pinpoints":conf.get("image_grid_pinpoints","unknown"),
        "mm_patch_merge_type":conf.get("mm_patch_merge_type","unknown"),
        "mm_vision_select_layer":conf.get("mm_vision_select_layer","unknown"),
        "config_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
    }

def preflight(ids,*,min_free_mib=10000,config_7b=None,config_13b=None,allow_busy=False):
    listing=gpu_inventory()
    selected=[]
    for i in ids:
        if i not in listing: raise ValueError(f"GPU {i} not found")
        info=listing[i]
        if "V100" not in info["name"].upper():
            raise ValueError(f"GPU {i} is not NVIDIA V100; found {info['name']}")
        if info["free_mib"]<min_free_mib and not allow_busy:
            raise RuntimeError(f"GPU {i} only free {info['free_mib']} MiB < {min_free_mib} MiB")
        selected.append(info)
    warnings=[
        "NVIDIA V100 compute capability 7.0: FP16 permitted; BF16 not supported",
        "Do not enable FlashAttention-2 (requires Ampere or newer)",
        "torchrun with four processes is data-parallel, NOT model/tensor parallel",
        "13B fp16 weights alone ~26GB decimal; V100 16GB x2 may still OOM when activations/vision encoder count",
        "Actual four-GPU model parallel grouping must be verified using the existing code and devices",
        "4 available cards do not imply they can be used concurrently by other jobs",
        "No CUDA kernels or model inference were launched by this preflight",
    ]
    return {"selected_ids":ids,"gpus":selected,
            "model7":model_info(config_7b,32),
            "model13":model_info(config_13b,40),
            "min_free_mib":min_free_mib,
            "available_gpu_memory_is_NOT_reservation":True,
            "warnings":warnings,"status":"HOST_GPU_CHECK_PASSED"}

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--gpu-ids",required=True,
        help="Physical GPU IDs, exactly 4, e.g. 0,2,4,7; no autodetection selection")
    a.add_argument("--min-free-mib",type=int,default=10000)
    a.add_argument("--config-7b")
    a.add_argument("--config-13b")
    a.add_argument("--output")
    a.add_argument("--allow-busy",action="store_true",
                   help="Print warnings only; never counts as allowed for experiment run")
    args=a.parse_args()
    report=preflight(parse_ids(args.gpu_ids),
                     min_free_mib=args.min_free_mib,config_7b=args.config_7b,
                     config_13b=args.config_13b,allow_busy=args.allow_busy)
    txt=json.dumps(report,indent=2,ensure_ascii=False)
    print(txt)
    if args.output:Path(args.output).write_text(txt+"\n",encoding="utf-8")
    if not args.config_7b or not args.config_13b:
        print("CAUTION: MODEL CONFIGS NOT PROVIDED, REAL EXPERIMENTS STILL BLOCKED",
              file=sys.stderr)
if __name__=="__main__":
    main()
