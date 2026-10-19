"""Create clearly labeled synthetic data ONLY for testing the plotting path."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from stage1_visual_audit import SCHEMA


def create(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    height, width, rows, cols = 180, 240, 3, 4
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack((180+40*xx/width, 190+35*yy/height, 185+15*xx/width), -1).astype(np.uint8)
    Image.fromarray(rgb).save(output/"fixture_crop.png")
    boxes = np.array([[x*width/cols, y*height/rows, (x+1)*width/cols, (y+1)*height/rows]
                      for y in range(rows) for x in range(cols)])
    ids = (np.arange(rows*cols)[::-1]*7+11).tolist()
    guide = np.array([.20, .16, .03, .01, .14, .13, .04, .02, .01, .02, .03, .01])
    pen = guide[::-1].copy()
    methods = []
    for name, beta in (("pen", 0.), ("guide", 1.), ("dual025", .25), ("dual050", .5), ("dual075", .75)):
        scores = (1-beta)*pen+beta*guide
        keep = np.sort(np.argsort(-scores, kind="stable")[:4])
        methods.append(dict(name=name, beta=beta, cover=False, config_hash="FIXTURE_"+name,
                            keep_ids=[ids[i] for i in keep]))
    record = dict(schema=SCHEMA, fixture=True, image_key="FIXTURE_IMAGE", crop_key="FIXTURE_CROP",
                  case_key="FIXTURE_CASE", image_path="fixture_crop.png", patch_ids=ids,
                  boxes_xyxy=boxes.tolist(), guide_scores=guide.tolist(), pen_scores=pen.tolist(),
                  methods=methods, work=1024, category="FIXTURE_ONLY",
                  source=dict(model_sha="FIXTURE", processor_sha="FIXTURE", selector_sha="FIXTURE",
                              saliency_rule="FIXTURE", schedule_hash="FIXTURE", split="development",
                              vision_depth=24, guide_completed=9, pen_completed=23, payload="pen"),
                  regions=[dict(name="FIXTURE_REGION", box_xyxy=[0, 0, 120, 120])])
    (output/"record_FIXTURE_ONLY.json").write_text(json.dumps(record, indent=2)+"\n")
    ledger = dict(schema="elva_stage1_sweep_v1", fixture=True, metric_name="Fixture score (arbitrary)",
                  source_hash="FIXTURE", scorer_sha="FIXTURE", frozen_panel_hash="FIXTURE",
                  timing_protocol_hash="FIXTURE", rows=[dict(beta=b, metric=m, wall_ttft_p50_ms=t,
                  work=1024, cover=False, payload="pen", split="development", selection_mode="weighted_evidence")
                  for b, m, t in [(0., 2., 10.), (.25, 2.2, 10.4), (.5, 2.3, 10.4), (.75, 2.1, 10.4), (1., 1.8, 10.2)]])
    (output/"sweep_FIXTURE_ONLY.json").write_text(json.dumps(ledger, indent=2)+"\n")
    return record


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    create(parser.parse_args().out)
