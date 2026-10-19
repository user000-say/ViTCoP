"""List complete four-task results; normalized R4 only with bound dense references."""
import argparse
import json
import math
from pathlib import Path
from make_catalog import TASKS
from elva.workflow import result_key, validate_result


def summarize(configs, binding, results):
    rows = []
    references = binding.get("normalized_metric_reference", {})
    for c in configs:
        scores, missing = {}, []
        normalized = []
        for task in TASKS:
            key = result_key(c, binding, task)
            folder = Path(results)/key
            try:
                value = json.loads((folder/"result.json").read_text())
                validate_result(value, c, binding, task, key, folder)
                scores[task] = value["metrics"]
                ref = references.get(task, {})
                name, dense = ref.get("metric"), ref.get("dense")
                if name in value["metrics"] and type(dense) in (int, float) and math.isfinite(dense) and dense > 0:
                    normalized.append(value["metrics"][name]/dense)
            except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
                missing.append(dict(task=task, reason=str(error)))
        rows.append(dict(config=c["id"], phase=c["phase"], budget=c["method"]["budget"],
                         cover_family=c["method"]["stage1"]["cover_family"], metrics=scores,
                         missing=missing, normalized_R4=100*sum(normalized)/4 if not missing and len(normalized)==4 else None))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="configs/catalog.json")
    parser.add_argument("--binding", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default="results_summary.json")
    args = parser.parse_args()
    load = lambda p: json.loads(Path(p).read_text())
    rows = summarize(load(args.catalog)["configs"], load(args.binding), args.results)
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False)+"\n")
