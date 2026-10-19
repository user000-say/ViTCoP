"""Freeze a completed Stage1 choice; preserve all four-task evidence hashes."""
import argparse
import json
from pathlib import Path
from elva.workflow import freeze, validate_binding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="configs/catalog.json")
    parser.add_argument("--binding", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--budget", type=int, choices=(2048, 1024), required=True)
    parser.add_argument("--winner", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--out", default="stage1_decisions.json")
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text())
    binding, catalog = load(args.binding), load(args.catalog)
    validate_binding(binding)
    decision = freeze(catalog["configs"], binding, args.results, args.budget, args.winner, args.reason)
    out = Path(args.out)
    if out.exists():
        old = load(out)
        if old.get("schema") != decision["schema"] or old.get("binding_hash") != decision["binding_hash"]:
            raise ValueError("do not merge decisions from different experiments")
        old["budgets"].update(decision["budgets"])
        decision = old
    out.write_text(json.dumps(decision, indent=2, ensure_ascii=False)+"\n")
    print(str(out))
