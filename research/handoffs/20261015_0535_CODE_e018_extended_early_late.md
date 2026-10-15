# CODE → MASTER: E018 extended early/late STATIC ACC+TTFT

Installed `experiments/e018_extended_4gpu/` from Stage2 Controller zip.
checks.py 12/12 (incl. torch parity). truth_manifest recovered ONE/TWO; EARLY≡TWO; LATE=`4:102,16:32` W=2564.
Skip smoke per user; launch EARLY/LATE ACC (EARLY seeded from TWO) + LATE TTFT on GPUs 0/2/4/6.
Controller online training deferred until adapter; this round is fixed early/late envelope.
