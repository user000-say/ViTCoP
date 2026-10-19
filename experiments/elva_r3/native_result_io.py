"""Call from the real project runner after generation, outside TTFT timing."""
import hashlib
import json
from pathlib import Path
from make_catalog import digest
from elva.workflow import validate_result


def sample_record(*, image_key, question_key, prediction, stage1_ids, stage1_audit,
                  trace, question=None, reference=None):
    if hasattr(stage1_ids, "detach"):
        stage1_ids = stage1_ids.detach().cpu().tolist()
    elif hasattr(stage1_ids, "tolist"):
        stage1_ids = stage1_ids.tolist()
    return dict(image_key=str(image_key), question_key=str(question_key), prediction=prediction,
                question=question, reference=reference, stage1_ids=stage1_ids,
                stage1_audit=stage1_audit, trace=list(trace))


def finish_task(output_path, *, records, official_metrics, config, binding, task, run_key,
                fixture=False):
    """Official scorer supplies metrics; this helper never invents a score."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions = out.parent/"predictions.jsonl"
    with predictions.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
    value = dict(schema="elva_r3_vlm_result", inference_kind="full_vlm_generate", fixture=fixture,
                 config_hash=config["config_hash"], binding_hash=digest(binding), task=task,
                 run_key=run_key, backend=binding["backend"], cache_mode=config["method"]["cache_mode"],
                 scorer_sha=binding["scorer_sha"], samples=len(records), metrics=official_metrics,
                 predictions_file=predictions.name,
                 predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest())
    validate_result(value, config, binding, task, run_key, out.parent, allow_fixture=fixture)
    out.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    return value
