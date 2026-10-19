"""Offline Stage1 diagnostics from source-bound scores and original patch IDs.

This module does not run a VLM or infer token utility from attention. Pixel
boxes must refer to the exact processed crop shown, never an assumed reshape.
No image or score is generated when real input is absent. Fixtures are explicit.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, to_rgba
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle


SCHEMA = "elva_stage1_visual_v1"
INK, BASE, NEW, BOTH = "#183e4b", "#c97a43", "#437fac", "#477f6b"


def _ids(value, name):
    a = np.asarray(value)
    if (a.ndim != 1 or not len(a) or a.dtype.kind not in "iu"
            or (a < 0).any() or (a > np.iinfo(np.int64).max).any()
            or len(np.unique(a)) != len(a)):
        raise ValueError(f"{name}: nonempty unique nonnegative integer IDs required")
    return a.astype(np.int64)


def _rect(value, width, height):
    a = np.asarray(value, dtype=float)
    if (a.ndim != 2 or a.shape[1] != 4 or not len(a) or not np.isfinite(a).all()
            or (a[:, :2] < 0).any() or (a[:, 2] > width).any()
            or (a[:, 3] > height).any() or (a[:, 2:] <= a[:, :2]).any()):
        raise ValueError("boxes must be finite positive xyxy inside the processed crop")
    # A crop's patch boxes must not overlap. Otherwise summed region coverage
    # is invalid; separate crops must be separate input records.
    lo = np.maximum(a[:, None, :2], a[None, :, :2])
    hi = np.minimum(a[:, None, 2:], a[None, :, 2:])
    area = np.maximum(0., hi-lo).prod(-1)
    np.fill_diagonal(area, 0.)
    if (area > 1e-7).any():
        raise ValueError("overlapping patch boxes; split image crops into separate records")
    return a


def validate_record(record, *, allow_fixture=False):
    if record.get("schema") != SCHEMA or type(record.get("fixture")) is not bool:
        raise ValueError("explicit schema and fixture flag required")
    if record["fixture"] and not allow_fixture:
        raise ValueError("fixture input requires --allow-fixture")
    for key in ("image_key", "crop_key", "case_key", "image_path"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"nonempty {key} required")
    source = record.get("source", {})
    for key in ("model_sha", "processor_sha", "selector_sha", "saliency_rule",
                "schedule_hash", "split"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise ValueError(f"source.{key} required")
    depth, guide, pen = (source.get(k) for k in
                         ("vision_depth", "guide_completed", "pen_completed"))
    if (any(type(k) is not int for k in (depth, guide, pen))
            or depth < 2 or not 1 <= guide <= depth or pen != depth-1):
        raise ValueError("source must bind completed guide layer and actual penultimate layer")
    if source.get("payload") != "pen":
        raise ValueError("this audit compares selection only; payload must be raw pen")
    if type(record.get("work")) is not int or record["work"] < 1:
        raise ValueError("positive declared token-layer work required")
    image_path = Path(record["image_path"])
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    ids = _ids(record.get("patch_ids"), "patch_ids")
    boxes = _rect(record.get("boxes_xyxy"), rgb.shape[1], rgb.shape[0])
    if len(boxes) != len(ids):
        raise ValueError("one actual pixel box per original patch ID required")
    scores = {}
    for key in ("guide_scores", "pen_scores"):
        a = np.asarray(record.get(key), dtype=float)
        if a.shape != ids.shape or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError(f"{key}: aligned nonnegative finite source scores required")
        scores[key] = a
    methods, all_ids = {}, set(map(int, ids))
    for method in record.get("methods", []):
        name = method.get("name")
        if not isinstance(name, str) or not name or name in methods:
            raise ValueError("unique named methods required")
        if not isinstance(method.get("config_hash"), str) or not method["config_hash"]:
            raise ValueError("each actual keep set must bind a configuration hash")
        keep = _ids(method.get("keep_ids"), name)
        if not set(map(int, keep)).issubset(all_ids):
            raise ValueError("unknown/deleted keep ID; never interpret IDs as row numbers")
        methods[name] = {**method, "keep_ids": keep}
    if len(methods) < 2 or len({len(v["keep_ids"]) for v in methods.values()}) != 1:
        raise ValueError("at least two methods with identical entry K required")
    regions = []
    for region in record.get("regions", []):
        if not isinstance(region.get("name"), str) or not region["name"]:
            raise ValueError("annotated region name required")
        box = _rect([region.get("box_xyxy")], rgb.shape[1], rgb.shape[0])[0]
        regions.append({"name": region["name"], "box": box})
    return {**record, "rgb": rgb, "patch_ids": ids, "boxes_xyxy": boxes,
            **scores, "methods": methods, "regions": regions}


def load_record(path, *, allow_fixture=False):
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    image = Path(record.get("image_path", ""))
    if not image.is_absolute():
        record["image_path"] = str((path.parent/image).resolve())
    return validate_record(record, allow_fixture=allow_fixture)


def keep_mask(record, method):
    return np.isin(record["patch_ids"], record["methods"][method]["keep_ids"])


def matched_random_keep(patch_ids, baseline_ids, candidate_ids, *, seed):
    """CPU reference for a matched random SUPPLEMENT control.

    Freeze the exact baseline tokens removed by the dual method, replace its
    additions by an equally sized uniform sample outside the baseline. Keep
    original sequence order. This is neither a quota selector nor random
    decoder pruning. A server must bind/record its device-native equivalent.
    """
    ids = _ids(patch_ids, "patch_ids")
    baseline, candidate = _ids(baseline_ids, "baseline"), _ids(candidate_ids, "candidate")
    if (len(baseline) != len(candidate) or not set(baseline).issubset(ids)
            or not set(candidate).issubset(ids) or type(seed) is not int or seed < 0):
        raise ValueError("same-K actual subsets and nonnegative integer seed required")
    common = np.intersect1d(baseline, candidate)
    count = len(baseline)-len(common)
    outside = ids[~np.isin(ids, baseline)]
    added = np.random.default_rng(seed).choice(outside, size=count, replace=False)
    keep = ids[np.isin(ids, np.r_[common, added])]
    return keep, dict(changed=count, seed=seed, removed_ids=baseline[~np.isin(baseline, common)].tolist(),
                      added_ids=added.tolist())


def pair_metrics(record, baseline, candidate):
    a, b = keep_mask(record, baseline), keep_mask(record, candidate)
    common, union = int((a & b).sum()), int((a | b).sum())
    regions = []
    for region in record["regions"]:
        rect = region["box"]
        area = np.maximum(0., np.minimum(record["boxes_xyxy"][:, 2:], rect[2:])
                          - np.maximum(record["boxes_xyxy"][:, :2], rect[:2])).prod(-1)
        denominator = float(np.prod(rect[2:]-rect[:2]))
        regions.append({"name": region["name"],
                        "baseline_area_fraction": float(area[a].sum()/denominator),
                        "candidate_area_fraction": float(area[b].sum()/denominator)})
    return dict(image_key=record["image_key"], crop_key=record["crop_key"],
                case_key=record["case_key"], baseline=baseline, candidate=candidate,
                k=int(a.sum()), work=record["work"], common=common, union=union,
                jaccard=common/union, retention_overlap=common/int(a.sum()),
                baseline_only=list(map(int, record["patch_ids"][a & ~b])),
                candidate_only=list(map(int, record["patch_ids"][b & ~a])),
                annotated_region_coverage=regions or None)


def _axis(ax, rgb, title):
    ax.imshow(rgb)
    ax.set(xlim=(0, rgb.shape[1]), ylim=(rgb.shape[0], 0), title=title)
    ax.set_aspect("equal")
    ax.axis("off")


def _boxes(ax, boxes, *, values=None, cmap="magma", norm=None,
           facecolors="none", edgecolor="none", alpha=1., linewidth=.8):
    collection = PatchCollection([Rectangle((x, y), right-x, bottom-y)
                                  for x, y, right, bottom in boxes],
                                 cmap=cmap, norm=norm,
                                 facecolors=None if values is not None else facecolors,
                                 edgecolors=edgecolor, linewidths=linewidth, alpha=alpha)
    if values is not None:
        collection.set_array(np.asarray(values))
    ax.add_collection(collection)
    return collection


def _save(fig, output, *, fixture):
    if fixture:
        fig.get_layout_engine().set(rect=(0, 0, 1, .92))
        fig.text(.5, .985, "SYNTHETIC FIXTURE — NOT VLM EVIDENCE", ha="center", va="top",
                 fontsize=13, color="#a44131", bbox=dict(facecolor="white", edgecolor="none", pad=3))
    fig.savefig(str(output)+".png", dpi=160, facecolor="white")
    fig.savefig(str(output)+".svg", facecolor="white")
    plt.close(fig)


def render_case(record, output, baseline, candidate, vmax):
    rgb, boxes = record["rgb"], record["boxes_xyxy"]
    guide = record["source"]["guide_completed"]
    pen = record["source"]["pen_completed"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    titles = ["Processed crop", f"CLS score: completed block {guide}",
              f"CLS score: pen / block {pen}", baseline, candidate,
              "Actual selected identities"]
    for ax, title in zip(axes.flat, titles):
        _axis(ax, rgb, title)
    norm = Normalize(0, vmax)
    for ax, key in ((axes[0, 1], "guide_scores"), (axes[0, 2], "pen_scores")):
        _boxes(ax, boxes, values=record[key], norm=norm, alpha=.7)
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap="magma")
    fig.colorbar(scalar, ax=list(axes[0, 1:]), label="Raw score; shared scale across this input batch", shrink=.75)
    a, b = keep_mask(record, baseline), keep_mask(record, candidate)
    _boxes(axes[1, 0], boxes[a], edgecolor=BASE, linewidth=1.5)
    _boxes(axes[1, 1], boxes[b], edgecolor=NEW, linewidth=1.5)
    selected = a | b
    colors = [to_rgba(BOTH if aa and bb else BASE if aa else NEW, .65)
              for aa, bb in zip(a[selected], b[selected])]
    _boxes(axes[1, 2], boxes[selected], facecolors=colors, edgecolor="white", linewidth=.4)
    fig.legend(handles=[Patch(color=BOTH, label="Both"),
                        Patch(color=BASE, label="Baseline only"),
                        Patch(color=NEW, label="Candidate only")],
               loc="outside lower center", ncol=3, frameon=False)
    _save(fig, output/"panel", fixture=record["fixture"])
    methods = list(record["methods"])
    rows = math.ceil(len(methods)/3)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3.9*rows), squeeze=False,
                             layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, name in zip(axes.flat, methods):
        _axis(ax, rgb, f"{name} / K={int(keep_mask(record, name).sum())}")
        _boxes(ax, boxes[keep_mask(record, name)], edgecolor=NEW, linewidth=1.3)
    _save(fig, output/"selection_grid", fixture=record["fixture"])
    # Keep all numeric values, including zero masses. There is no automatic
    # within-visual normalization that could conceal absolute mass changes.
    np.savez_compressed(output/"raw_values.npz", patch_ids=record["patch_ids"],
                        boxes_xyxy=boxes, guide_scores=record["guide_scores"],
                        pen_scores=record["pen_scores"])


def audit(paths, output, *, baseline, candidate, allow_fixture=False):
    records = [load_record(p, allow_fixture=allow_fixture) for p in paths]
    if not records:
        raise ValueError("no actual input records")
    if len({r["fixture"] for r in records}) != 1:
        raise ValueError("do not mix fixture and real records")
    names = list(records[0]["methods"])
    if baseline not in names or candidate not in names or baseline == candidate:
        raise ValueError("two distinct actual method names required")
    signature = lambda r: (r["work"], len(r["methods"][baseline]["keep_ids"]),
                            tuple(r["source"][k] for k in
                                  ("model_sha", "processor_sha", "selector_sha", "saliency_rule",
                                   "schedule_hash", "split", "vision_depth", "guide_completed", "pen_completed")))
    unique = {}
    for r in records:
        if set(r["methods"]) != set(names) or signature(r) != signature(records[0]):
            raise ValueError("one budget, method panel and source binding per audit batch required")
        key = (r["image_key"], r["crop_key"])
        if key in unique:
            raise ValueError("duplicate image/crop: Stage1 panels must not repeat questions as independent images")
        unique[key] = r
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    vmax = max(float(r[k].max()) for r in records for k in ("guide_scores", "pen_scores")) or 1.
    rows, per_image = [], defaultdict(list)
    for r in records:
        case_hash = hashlib.sha256((r["image_key"]+"\0"+r["crop_key"]).encode()).hexdigest()[:16]
        folder = output/case_hash
        folder.mkdir(exist_ok=True)
        render_case(r, folder, baseline, candidate, vmax)
        metrics = pair_metrics(r, baseline, candidate)
        metrics.update(fixture=r["fixture"], source=r["source"], shared_raw_vmax=vmax,
                       category=r.get("category"), predictions=r.get("predictions"),
                       note="Attention, selection and geometric coverage diagnostics; not causal token utility")
        (folder/"metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False)+"\n")
        rows.append(metrics)
        matrix = np.array([[pair_metrics(r, aa, bb)["jaccard"] for bb in names] for aa in names])
        per_image[r["image_key"]].append(matrix)
    # Average crops inside each canonical image before averaging images.
    overlap = np.stack([np.mean(values, axis=0) for values in per_image.values()]).mean(0)
    fig, ax = plt.subplots(figsize=(max(6., len(names)*.9), max(5., len(names)*.8)), layout="constrained")
    im = ax.imshow(overlap, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_title("Selected-set Jaccard; each canonical image weighted equally")
    for i, j in itertools.product(range(len(names)), repeat=2):
        ax.text(j, i, f"{overlap[i, j]:.2f}", ha="center", va="center",
                color="white" if overlap[i, j] < .55 else INK, fontsize=9)
    fig.colorbar(im, ax=ax, label="Jaccard", shrink=.8)
    _save(fig, output/"overlap_matrix", fixture=records[0]["fixture"])
    with (output/"overlap.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["image_key", "crop_key", "case_key", "k", "work", "baseline", "candidate",
                  "common", "union", "jaccard", "retention_overlap"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    result = dict(schema=SCHEMA, fixture=records[0]["fixture"], image_count=len(per_image),
                  crop_count=len(records), methods=names, shared_raw_vmax=vmax,
                  work=records[0]["work"], overlap_matrix=overlap.tolist(),
                  source=records[0]["source"], score_zero=all(not r[k].any() for r in records
                                                           for k in ("guide_scores", "pen_scores")))
    (output/"summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    return result


def render_sweep(path, output, *, allow_fixture=False):
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "elva_stage1_sweep_v1" or type(value.get("fixture")) is not bool:
        raise ValueError("sweep schema and explicit fixture flag required")
    if value["fixture"] and not allow_fixture:
        raise ValueError("fixture sweep requires --allow-fixture")
    for key in ("metric_name", "source_hash", "scorer_sha", "frozen_panel_hash", "timing_protocol_hash"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"bound {key} required")
    groups = defaultdict(list)
    for row in value.get("rows", []):
        if row.get("selection_mode") != "weighted_evidence" or type(row.get("cover")) is not bool:
            raise ValueError("do not mix quota selection, fusion and weighting on one beta curve")
        beta, metric, ttft = (row.get(k) for k in ("beta", "metric", "wall_ttft_p50_ms"))
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (beta, metric, ttft))
                or not 0 <= beta <= 1 or ttft <= 0 or type(row.get("work")) is not int
                or row["work"] <= 0 or row.get("payload") != "pen"
                or row.get("split") not in ("development", "confirmation")):
            raise ValueError("finite beta/official metric/positive wall TTFT and frozen protocol required")
        groups[(row["work"], row["cover"], row["split"])].append(row)
    if not groups:
        raise ValueError("empty sweep; real evaluation records required")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for (work, cover, split), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["beta"])
        x = [r["beta"] for r in rows]
        if len(set(x)) != len(x):
            plt.close(fig)
            raise ValueError("aggregate true repeats first; duplicate beta rows in one comparison")
        label = f"W={work}, COVER={cover}, {split}"
        axes[0].plot(x, [r["metric"] for r in rows], marker="o", label=label)
        axes[1].plot(x, [r["wall_ttft_p50_ms"] for r in rows], marker="o", label=label)
    for ax in axes:
        ax.set_xlabel("Guide-layer evidence weight beta (not a token quota)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(value["metric_name"])
    axes[1].set_ylabel("End-to-end wall TTFT p50 (ms)")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _save(fig, output/"beta_quality_ttft", fixture=value["fixture"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    image = commands.add_parser("images")
    image.add_argument("records", nargs="+")
    image.add_argument("--baseline", required=True)
    image.add_argument("--candidate", required=True)
    sweep = commands.add_parser("sweep")
    sweep.add_argument("ledger")
    for sub in (image, sweep):
        sub.add_argument("--out", required=True)
        sub.add_argument("--allow-fixture", action="store_true")
    args = parser.parse_args()
    if args.command == "images":
        print(json.dumps(audit(args.records, args.out, baseline=args.baseline,
                               candidate=args.candidate, allow_fixture=args.allow_fixture), indent=2))
    else:
        render_sweep(args.ledger, args.out, allow_fixture=args.allow_fixture)


if __name__ == "__main__":
    main()
