"""Reproducible local checks. All decoder outputs here are FIXTURES."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import compileall
import io
import json
import platform
import unittest
import numpy as np
from elva.budget import Schedule, feasible_suffixes
from elva.toy_adapter import ToyAdapter
from elva.policy import propose, collect_catalog
from make_catalog import build_catalog, main as generate


def main():
    root = Path(__file__).resolve().parent
    out = root/"validation"
    out.mkdir(exist_ok=True)
    generate()
    assert compileall.compile_dir(str(root/"elva"), quiet=1)
    assert compileall.compile_file(str(root/"run_queue.py"), quiet=1)
    catalog = build_catalog()
    budgets = []
    for c in catalog:
        p = Schedule(**c["method"]["schedule"])
        w = c["method"]["budget"]
        assert sum(p.trace) == p.work == w
        assert (1-p.compression(576))*576*p.depth == w
        budgets.append(dict(id=c["id"], counts=p.counts, lengths=p.lengths,
                            trace=p.trace, events=p.events, work=w,
                            compression_exact=str(p.compression(576)), **p.proxies(0)))
    (out/"budgets.json").write_text(json.dumps(budgets, indent=2))
    suffix = feasible_suffixes(depth=32, completed=10, spent=600, current_k=60,
                               budget=1024, boundaries=(18, 20, 26), finals=(12,))
    (out/"feasible_suffixes.json").write_text(json.dumps([
        dict(counts=p.counts, lengths=p.lengths, remaining_work=p.work,
             total_work=600+p.work) for p in suffix], indent=2))
    adapter = ToyAdapter()
    prefix = adapter.capture(2)
    ids = np.array(adapter.inspect(prefix)["active_ids"])
    scores = np.arange(len(ids))/len(ids)
    action_catalog = propose(scores, ids, np.arange(7), 2, group_size=2, limit=4,
                              begin=2, end=5, depth=8, proposal_rule="FIXTURE_ONLY")
    rows = []
    for mode in ("native", "aligned_support"):
        rows.extend(collect_catalog(adapter, prefix, action_catalog, ids, scores[:, None],
                                    begin=2, end=5, depth=8, image_key="FIXTURE_IMAGE",
                                    question_key="FIXTURE_QUESTION", feature_names=list("abcd"), cache_mode=mode))
    assert all(r["fixture"] for r in rows)
    (out/"toy_swaps_FIXTURE_ONLY.json").write_text(json.dumps(rows, indent=2))
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(root/"tests"))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (out/"tests.txt").write_text(stream.getvalue())
    summary = dict(python=platform.python_version(), tests=result.testsRun,
                   passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
                   skipped=[dict(test=str(test), reason=reason) for test, reason in result.skipped],
                   failures=len(result.failures), errors=len(result.errors),
                   exhaustive_suffix_catalog_cases=1000,
                   config_count=len(catalog), blocks=dict(Counter(c["block"] for c in catalog)),
                   unconditional=sum(not c["method"]["conditional"] for c in catalog),
                   conditional=sum(c["method"]["conditional"] for c in catalog),
                   four_task_capacity=4*len(catalog), formal_two_methods_three_budgets_nine_tasks=2*3*9,
                   dense_nine_tasks_separate=9,
                   old_shift_work=172*3+116*10+32*19,
                   old_shift_compression_percent=float(100*(1-Fraction(2284, 576*32))),
                   pretrained_vlm_accuracy_run=False, gpu_ttft_run=False,
                   torch_path_runtime_verified=False,
                   note="CPU reference, finite catalog and random causal decoder only; no pretrained-model claims")
    (out/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if not result.wasSuccessful():
        raise SystemExit(stream.getvalue())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
