"""Adapt the EXISTING successful ACC/TTFT runner; no new VLM implementation."""
from typing import Protocol


class Bridge(Protocol):
    working_directory: str

    def command(self,job:dict,outdir:str)->list[str]:
        """Return existing project runner argv. Pass job.json or explicit fields.

        Process sees exactly one physical GPU through CUDA_VISIBLE_DEVICES; use
        local cuda:0, not the physical index. Set dataset/panel and exact schedule.
        Install new evidence selector only at FINAL deletion event. For temporal
        modes cache query-group response at penultimate event, original patch IDs
        and original question-token group identity; align on device before use.
        Reference must call original path unchanged. Write official scoring output
        plus raw records and provenance. Do not invoke a shell command string.
        """
        ...

    def read_result(self,job:dict,outdir:str)->dict:
        """Return job_hash, metric_key, metric_value, panel_ids_hash, absolute
        records_path. Optional per-job selector/TTFT fields are diagnostic only.
        Aggregate normalized metric units: e.g. POPE 84.42, not .8442.
        Real record file must preserve per-sample predictions and identities.
        """
        ...

    def cached_result(self,job:dict):
        """Optional. Return only a truly matching result, otherwise None.
        Reuse EARLY/TWO actual records; never add duplicate independent samples.
        Full outputs can yield a development subset only through actual filtering
        and rescoring. An old aggregate cannot be treated as a subset result.
        """
        ...
