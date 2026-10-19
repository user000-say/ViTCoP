"""Per-sample evidence and explicit Stage1 freeze gates for real VLM jobs."""
import json
import math
from pathlib import Path
from .cover_binding import spec_fingerprint, origin
from .budget import Schedule


REQUIRED = ("source_commit", "model_sha", "processor_sha", "tokenizer_sha", "prompt_sha",
            "dataset_sha", "scorer_sha", "backend", "cache_mode", "native_selectors")


def validate_binding(binding, *, allow_fixture=False):
    if binding.get("fixture", True) and not allow_fixture:
        raise ValueError("real server binding required; fixture bindings cannot launch VLM evaluation")
    if any(not binding.get(key) for key in REQUIRED):
        raise ValueError("complete source/model/data/scorer binding required")
    if (binding.get("original_tokens"), binding.get("depth")) != (576, 32):
        raise ValueError("this two-budget catalog is explicitly for 576x32")
    if binding.get("cache_mode") != "native":
        raise ValueError("this primary two-budget comparison uses native KV behavior")
    checks = binding.get("preflight", {})
    if not all(checks.get(key) is True for key in ("off_path_equal", "clip_layers_checked", "pen_payload_equal", "positions_and_kv_checked")):
        raise ValueError("native OFF path, actual CLIP layer and pen payload checks must pass first")


def blocked_reason(config, binding):
    method = config["method"]
    if binding.get("preflight", {}).get("trace_verified", {}).get(str(method["budget"])) is not True:
        return "BLOCKED_TRACE: full VLM pre-block trace not checked for this schedule budget"
    selector = method["stage2"]["first_selector"]
    if not binding["native_selectors"].get(selector):
        return "BLOCKED_SELECTOR: original decoder selector not bound"
    family = method["stage1"]["cover_family"]
    if family == "none":
        return None
    spec = binding.get("native_cover", {}).get(family)
    k = method["schedule"]["counts"][0]
    if not spec or spec.get("family") != family or k not in spec.get("supported_k", []):
        return "BLOCKED_COVER_CAPACITY: no explicit source-bound COVER for requested entry K"
    if family == "protected_coverage":
        if spec.get("implementation_parity_verified") is not True:
            return "BLOCKED_COVER_PARITY: protected-control NumPy/Torch and real CLIP preflight not verified"
        from .low_budget_cover import POLICY
        if (spec.get("coverage_policy") != POLICY or method["stage1"].get("coverage_policy") != POLICY
                or method["stage1"].get("coverage_slots") not in spec.get("allowed_slots", [])):
            return "BLOCKED_COVER_POLICY: unverified protected coverage policy/slots"
    elif spec.get("legacy172_parity_verified") is not True:
        return "BLOCKED_COVER_PARITY: original K172 identity regression not verified"
    try:
        spec_fingerprint(spec)
    except (ValueError, OSError, RuntimeError) as error:
        return f"BLOCKED_COVER_SOURCE: {error}"
    return None


def trace_for(config):
    return list(Schedule(**config["method"]["schedule"]).trace)


def _integer_ids(value, k):
    return (isinstance(value, list) and len(value) == k and all(type(v) is int and v >= 0 for v in value)
            and len(set(value)) == k)


def validate_result(value, config, binding, task, run_key, result_dir, *, allow_fixture=False):
    from make_catalog import digest
    method, stage = config["method"], config["method"]["stage1"]
    if digest(method) != config.get("config_hash"):
        raise ValueError("configuration edited without regenerating its fingerprint")
    if value.get("schema") != "elva_r3_vlm_result" or value.get("inference_kind") != "full_vlm_generate":
        raise ValueError("a CLIP-only pilot or old schema is not an official VLM result")
    if value.get("fixture", True) and not allow_fixture:
        raise ValueError("synthetic results cannot be accepted as real VLM ACC")
    expected = dict(config_hash=config["config_hash"], binding_hash=digest(binding),
                    run_key=run_key, task=task, backend=binding["backend"],
                    cache_mode=method["cache_mode"], scorer_sha=binding["scorer_sha"])
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("stale configuration, source, task or scorer fingerprint")
    count, metrics = value.get("samples"), value.get("metrics")
    if type(count) is not int or count < 1 or not isinstance(metrics, dict) or not metrics:
        raise ValueError("actual scored sample count and official metrics required")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in metrics.values()):
        raise ValueError("nonfinite official metric")
    declared_count = binding.get("task_samples", {}).get(task)
    if type(declared_count) is not int or count != declared_count:
        raise ValueError("evaluated samples must match the frozen task/panel manifest")
    root = Path(result_dir).resolve()
    filename = value.get("predictions_file")
    if not isinstance(filename, str):
        raise ValueError("per-sample predictions and traces are mandatory")
    predictions = (root/filename).resolve()
    if root not in predictions.parents or not predictions.is_file():
        raise ValueError("predictions file must exist inside this run directory")
    import hashlib
    if hashlib.sha256(predictions.read_bytes()).hexdigest() != value.get("predictions_sha256"):
        raise ValueError("prediction file fingerprint mismatch")
    family = stage["cover_family"]
    covered, k = family != "none", method["schedule"]["counts"][0]
    source_sha = spec_fingerprint(binding["native_cover"][family]) if covered else None
    seen, actual = set(), 0
    with predictions.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = row.get("image_key"), row.get("question_key")
            if any(not isinstance(v, str) or not v for v in key) or key in seen:
                raise ValueError("canonical nonduplicate image/question keys required")
            seen.add(key)
            if (not isinstance(row.get("prediction"), str) or row.get("trace") != trace_for(config)
                    or any(type(v) is not int for v in row["trace"])):
                raise ValueError("missing generation or wrong actual pre-block token trace")
            if not _integer_ids(row.get("stage1_ids"), k):
                raise ValueError("actual Stage1 identities have wrong K or duplicates")
            audit = row.get("stage1_audit", {})
            spec = binding.get("native_cover", {}).get(family, {})
            native_k = 172 if family == "legacy_pool172_then_topk" else k
            slots = (stage.get("coverage_slots") if family == "protected_coverage" else
                     spec.get("replacement_slots", {}).get(str(native_k))) if covered else 0
            required = dict(cover_family=family, requested_k=k, cover_calls=int(covered),
                            native_calls=int(covered and family != "protected_coverage"),
                            origin=origin(family), replacement_slots=slots,
                            fallback=False, payload="pen", cover_source_sha=source_sha)
            if any(audit.get(key) != value for key, value in required.items()):
                raise ValueError("missing native COVER call proof or forbidden fallback")
            expected_native_k = (172 if family == "legacy_pool172_then_topk" else k) if covered else None
            if audit.get("native_k") != expected_native_k:
                raise ValueError("native COVER capacity does not match the declared implementation")
            actual += 1
    if actual != count:
        raise ValueError("prediction rows do not cover all scored samples")


def result_key(config, binding, task):
    from make_catalog import digest
    return digest(dict(method=config["method"], binding=binding, task=task))


def collect_complete(configs, binding, result_root, budget, *, allow_fixture=False):
    complete = []
    for c in configs:
        if c["phase"] != "stage1_core" or c["method"]["budget"] != budget:
            continue
        jobs = []
        for task in ("GQA", "POPE", "MME", "MMBench"):
            key = result_key(c, binding, task)
            folder = Path(result_root)/key
            path = folder/"result.json"
            value = json.loads(path.read_text())
            validate_result(value, c, binding, task, key, folder, allow_fixture=allow_fixture)
            jobs.append(dict(task=task, run_key=key, result_hash=__import__('hashlib').sha256(path.read_bytes()).hexdigest()))
        complete.append(dict(config=c, jobs=jobs))
    if len(complete) != 5:
        raise ValueError("all five matched Stage1 groups and all four tasks must be complete before freeze")
    return complete


def freeze(configs, binding, results, budget, winner, rationale, *, allow_fixture=False):
    from make_catalog import digest
    if budget not in (2048, 1024) or not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("explicit supported budget and evidence-based rationale required")
    complete = collect_complete(configs, binding, results, budget, allow_fixture=allow_fixture)
    selected = next((c for c in configs if c["id"] == winner and c["phase"] in ("stage1_core", "stage1_optional")
                     and c["method"]["budget"] == budget), None)
    if selected is None:
        raise ValueError("winner must be a Stage1 configuration at this budget")
    winner_evidence = []
    for task in ("GQA", "POPE", "MME", "MMBench"):
        key = result_key(selected, binding, task)
        folder = Path(results)/key
        path = folder/"result.json"
        value = json.loads(path.read_text())
        validate_result(value, selected, binding, task, key, folder, allow_fixture=allow_fixture)
        winner_evidence.append(dict(task=task, run_key=key, result_hash=__import__('hashlib').sha256(path.read_bytes()).hexdigest()))
    return dict(schema="elva_r3_stage1_freeze", binding_hash=digest(binding),
                budgets={str(budget):dict(status="FROZEN", stage1=selected["method"]["stage1"],
                         winner=winner, rationale=rationale, core_evidence=complete,
                         winner_config=selected, winner_evidence=winner_evidence)})


def validate_decision(decision, binding, configs, results, budget, *, allow_fixture=False):
    from make_catalog import digest
    if decision.get("schema") != "elva_r3_stage1_freeze" or decision.get("binding_hash") != digest(binding):
        raise ValueError("Stage1 freeze belongs to a different model/data/source binding")
    item = decision.get("budgets", {}).get(str(budget), {})
    if item.get("status") != "FROZEN":
        raise ValueError("Stage1 is not frozen at this budget")
    actual = collect_complete(configs, binding, results, budget, allow_fixture=allow_fixture)
    if item.get("core_evidence") != actual:
        raise ValueError("Stage1 results changed or are incomplete; regenerate the decision")
    winner = next((c for c in configs if c["id"] == item.get("winner")
                   and c["phase"] in ("stage1_core", "stage1_optional") and c["method"]["budget"] == budget), None)
    if not winner or item.get("stage1") != winner["method"]["stage1"] or item.get("winner_config") != winner:
        raise ValueError("frozen Stage1 does not match its recorded result")
    current = freeze(configs, binding, results, budget, winner["id"], item.get("rationale"), allow_fixture=allow_fixture)
    if current["budgets"][str(budget)]["winner_evidence"] != item.get("winner_evidence"):
        raise ValueError("winner results changed after the Stage1 freeze")
