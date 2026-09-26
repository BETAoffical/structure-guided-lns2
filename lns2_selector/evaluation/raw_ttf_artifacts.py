"""Read-only identity and outcome checks for the registered raw TTF lanes."""
from __future__ import annotations

import math

from experiments._common import json_fingerprint
from lns2_selector.runtime.contracts import require_bool, require_int


ARTIFACT_FILES = frozenset(("initial.json", "final.json", "trace.jsonl.gz"))


def validate_raw_ttf_identity(row, job, binding):
    if not isinstance(row, dict):
        raise ValueError("timed result must be an object")
    expected = {key: job[key] for key in
                ("job_id", "pair_id", "replica", "comparison_arm", "solver_seed", "budget_seconds")}
    expected.update(
        schema="lns2.sa_raw_ttf.episode.v1", status="ok", binding=binding,
        task_id=job["case"]["task_id"], map_id=job["case"]["map_id"],
        initial_fingerprint=job["expected_initial"],
        policy_sha256=job["model"]["policy_sha256"] if job["model"] else job["arm"],
        rng_stream_id=json_fingerprint([job["plan"]["config"]["stream_seed"],
                                       job["phase"], job["pair_id"], job["replica"]]))
    for key, value in expected.items():
        if key not in row or row[key] != value:
            raise ValueError("timed result identity mismatch: " + key)
    for key in ("replica", "solver_seed"):
        require_int(row[key], field=key, minimum=0)
    for key in ("feasible", "success_within_budget", "delivered_within_budget"):
        require_bool(row.get(key), field=key)
    files = row.get("files")
    if not isinstance(files, dict) or set(files) != ARTIFACT_FILES:
        raise ValueError("timed artifact file coverage mismatch")
    for digest in files.values():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid timed artifact SHA256")


def validate_raw_ttf_outcome(row, initial, final, pp_seconds):
    """Compare saved scalar metrics with authenticated states/trace aggregates."""
    for key, state in (("initial_conflicts", initial), ("final_conflicts", final)):
        value = require_int(row.get(key), field=key, minimum=0)
        if value != state["num_of_colliding_pairs"]:
            raise ValueError("timed outcome mismatch: " + key)
    for key in ("decisions", "legal_noops", "pp_calls", "generated", "soc", "makespan", "wait_steps"):
        require_int(row.get(key), field=key, minimum=0)
    for key in ("reset_seconds", "selection_seconds", "step_wall_seconds", "native_pp_seconds",
                "trace_seconds", "bookkeeping_seconds", "search_end_seconds", "finalization_seconds",
                "delivery_seconds", "setup_seconds", "cold_worker_seconds", "budget_seconds"):
        value = row.get(key)
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid or (key == "budget_seconds" and value == 0):
            raise ValueError("invalid timed metric: " + key)
    # The producer adds the per-step double values sequentially. Only tolerate
    # floating-point summation differences, never a missing or replaced total.
    if not math.isfinite(pp_seconds) or pp_seconds < 0 or not math.isclose(
            row["native_pp_seconds"], pp_seconds, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("native PP time differs from trace")
