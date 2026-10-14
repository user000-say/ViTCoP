#!/usr/bin/env python3
"""Fit a frozen-VLM continuation predictor. Image-group splits; finite calibration.

Training uses numpy only. JSON weights are portable to CUDA inference. No fake
or pretrained controller is bundled: input MUST be your collected labels.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from e018_tools import contract, read_rows

C = contract()


def train(x, y, mask, epochs=300, width=32, seed=20261014):
    rng = np.random.default_rng(seed)
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-3)
    z = np.clip((x - mean) / scale, -8, 8) * mask
    weights = dict(w1=rng.normal(0, 0.12, (z.shape[1], width)), b1=np.zeros(width),
                   w2=rng.normal(0, 0.08, (width, 1)), b2=np.zeros(1))
    m, v = {k: np.zeros_like(w) for k, w in weights.items()}, {k: np.zeros_like(w) for k, w in weights.items()}
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(z))
        for start in range(0, len(z), 128):
            ids = order[start:start + 128]
            a, target = z[ids], y[ids, None]
            h = np.tanh(a @ weights["w1"] + weights["b1"])
            pred = h @ weights["w2"] + weights["b2"]
            # Huber derivative reduces the effect of occasional very large NLL labels.
            d = np.clip(pred - target, -0.25, 0.25) / len(ids)
            dh = (d @ weights["w2"].T) * (1 - h * h)
            grads = dict(w2=h.T @ d, b2=d.sum(0), w1=a.T @ dh, b1=dh.sum(0))
            step += 1
            for key, grad in grads.items():
                grad = grad + 1e-5 * weights[key] if key.startswith("w") else grad
                m[key] = 0.9 * m[key] + 0.1 * grad
                v[key] = 0.999 * v[key] + 0.001 * grad ** 2
                weights[key] -= 0.003 * (m[key] / (1 - 0.9 ** step)) / (np.sqrt(v[key] / (1 - 0.999 ** step)) + 1e-8)
    return dict(mean=mean, scale=scale, **weights)


def predict(x, weights, mask):
    z = np.clip((x - weights["mean"]) / weights["scale"], -8, 8) * mask
    return (np.tanh(z @ weights["w1"] + weights["b1"]) @ weights["w2"] + weights["b2"]).reshape(-1)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--ablate", choices=["none", "stage1", "history", "receiver", "mass_only"], default="none")
    args = p.parse_args()
    rows = [r for path in args.labels for r in read_rows(path)]
    if not rows or len({r["fingerprint"] for r in rows}) != 1:
        p.error("Require nonempty labels with a single frozen protocol fingerprint")
    seen, images = set(), {}
    for r in rows:
        key = (str(r["image_id"]), str(r["question_id"]), r["event"], r.get("state_id", "initial"))
        if key in seen:
            p.error("Duplicate counterfactual row")
        seen.add(key)
        if r.get("schema") != C.SCHEMA or r.get("feature_names") != list(C.FEATURE_NAMES):
            p.error("Label feature/schema mismatch")
        split = C.group_split(r["image_id"])
        if r["split"] != split:
            p.error("Split differs from deterministic image grouping")
        images[str(r["image_id"])] = split
    x, y = np.array([r["features"] for r in rows]), np.array([r["utility"] for r in rows])
    if x.shape != (len(rows), len(C.FEATURE_NAMES)) or not np.isfinite(x).all() or not np.isfinite(y).all():
        p.error("Invalid/nonfinite labels")
    groups = {s: np.array([i for i, r in enumerate(rows) if r["split"] == s], dtype=int)
              for s in ("train", "calibration", "test")}
    if any(len(v) == 0 for v in groups.values()):
        p.error("Need real train/calibration/test image groups")
    mask = np.ones(len(C.FEATURE_NAMES))
    s1 = {"added_fraction", "drop_added_fraction", "drop_added_mass", "drop_cls_mass", "keep_cls_mass", "s1_exchange_fraction"}
    history = {"response_drift", "has_history", "receiver_cosine_drift"}
    for i, name in enumerate(C.FEATURE_NAMES):
        if ((args.ablate == "stage1" and name in s1) or (args.ablate == "history" and name in history) or
            (args.ablate == "receiver" and (name.startswith("receiver_") or name == "receiver_cosine_drift")) or
            (args.ablate == "mass_only" and name not in {"layer_fraction", "delta_work", "raw_mass_mean", "drop_raw_mass", "drop_response_mass"})):
            mask[i] = 0
    w = train(x[groups["train"]], y[groups["train"]], mask, args.epochs, seed=args.seed)
    predictions = predict(x, w, mask)
    # Max residual per calibration image handles correlated questions and both events.
    residuals = {}
    for i in groups["calibration"]:
        gid = str(rows[i]["image_id"])
        residuals[gid] = max(residuals.get(gid, -float("inf")), float(y[i] - predictions[i]))
    q = C.conformal_upper(list(residuals.values()), args.alpha)
    report = {}
    for s, ids in groups.items():
        report[s] = dict(rows=len(ids), images=len({str(rows[i]["image_id"]) for i in ids}),
                         mae=float(np.mean(np.abs(y[ids] - predictions[ids]))),
                         rmse=float(np.sqrt(np.mean((y[ids] - predictions[ids]) ** 2))),
                         upper_coverage=float(np.mean(y[ids] <= predictions[ids] + q)))
    artifact = dict(schema=C.SCHEMA, feature_names=list(C.FEATURE_NAMES), fingerprint=rows[0]["fingerprint"],
                    calibrated=True, alpha=args.alpha, q_upper=q, feature_mask=mask.tolist(),
                    seed=args.seed, epochs=args.epochs, ablation=args.ablate, report=report,
                    split_rule="sha256 image grouping 70/15/15; image-max calibration residual",
                    **{k: v.tolist() for k, v in w.items()})
    if not all(np.isfinite(v).all() for v in w.values()):
        p.error("Nonfinite trained parameters")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("x") as f:
        json.dump(artifact, f, indent=2)
    print(json.dumps(dict(q_upper=q, report=report), indent=2))


if __name__ == "__main__":
    main()
