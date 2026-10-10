#!/usr/bin/env bash
# Filled with verified E012-era runners adapted to E013 boot/latency.
# Do not invent new eval implementations — only env + e013_boot / e013_smoke / e013_latency.
E013_SMOKE_CMD='python /home/xiangyue/ViTCoP/e013_smoke.py --arm {arm} --gpu {gpu}'
E013_PANEL_CMD='python /home/xiangyue/ViTCoP/e013_boot.py --profile ANCHOR_A --task {task} --output {out_dir} --gpu-index {gpu} --layer {layer} --limit 512'
E013_FULL_CMD='python /home/xiangyue/ViTCoP/e013_boot.py --profile ANCHOR_A --task {task} --output {out_dir} --gpu-index {gpu} --layer {layer}'
E013_LATENCY_CMD='python /home/xiangyue/ViTCoP/e013_latency.py --arm {arm} --gpu {gpu}'
