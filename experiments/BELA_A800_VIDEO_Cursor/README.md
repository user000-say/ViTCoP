# BELA / CVPR 2027 — LLaVA-Video-7B / dual A800 experiment kit

Read **[the complete Cursor Handoff](docs/CURSOR_HANDOFF.md)** first.

Includes a frozen copy of the original QA-COVER operator, a frame-wise video selector adapter, 28-layer STAR-Pro-matched schedules, per-frame token audit, deterministic selector-only wrong-video-query controls, official paper reference scores and benchmark plan, and a safe read-only A800 preflight script.

**No SSH login, remote model patch, GPU inference, or video accuracy was performed**. Cursor must connect this code to the exact live LLaVA-Video checkout and establish off-path equivalence, positional alignment and physically shortened sequences before benchmark Accuracy is valid.

```bash
bash scripts/offline_verify.sh
bash scripts/a800_preflight.sh
python3 -m bela_video_a800.schedule --tier T32 --frames 64
python3 -m bela_video_a800.schedule --tier T16 --frames 64
```

No login usernames, host addresses, passwords, secrets or SSH configuration are stored in this package.
