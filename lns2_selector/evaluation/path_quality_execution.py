"""Prepared scheduling and durable path capture; no cohort launch CLI.

Existing first-phase selection is reused without rewriting proposal or ranking.
The process entry is opt-in and is exercised only with tiny functional fixtures.
"""
from __future__ import annotations

import copy
import itertools
import math
import multiprocessing
import time
from pathlib import Path
from typing import Any

from experiments._common import json_fingerprint, read_json, sha256_file, write_json
from lns2_selector.evaluation.anytime_handoff import continue_with_official_anytime
from lns2_selector.evaluation.path_quality_preflight import audit_paths, contained, read_grid, scenario_prefix

CONTROLLERS = ("official_adaptive", "v2-full", "dual16")


def execution_schedule(cases: list[dict], protocol: dict) -> list[dict]:
    """Use all statically admitted cases, never inspect solver outcomes."""
    permutations = list(itertools.permutations(CONTROLLERS))
    modes = [("first_feasible", protocol["first_feasible_budget_seconds"])]
    modes += [("fixed_budget", b) for b in protocol["total_planning_budgets_seconds"]]
    out = []
    for mode, budget in modes:
        if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
            raise ValueError("invalid prepared budget")
        key_index = 0
        for case in sorted(cases, key=lambda c: (c["map_id"], c["task_id"])):
            if case["status"] == "quarantined":
                continue
            if case["status"] != "static_ready_runtime_unverified":
                raise ValueError("unknown case admission status")
            for seed in case["solver_seeds"]:
                paired_seed = int(json_fingerprint({"task": case["task_id"], "seed": seed, "stage": 2})[:8], 16) % (2**31)
                for position, controller in enumerate(permutations[key_index % 6]):
                    identity = {"task_id": case["task_id"], "solver_seed": seed, "controller": controller,
                                "protocol": mode, "budget_seconds": float(budget)}
                    out.append({**identity, "map_id": case["map_id"], "family": case["family"],
                                "job_id": json_fingerprint(identity)[:24], "schedule_index": len(out),
                                "within_pair_position": position, "stage2_seed": paired_seed,
                                "repair_iteration_cap": None, "timed_workers": 1,
                                "execution_authorized": False})
                key_index += 1
    if len({row["job_id"] for row in out}) != len(out):
        raise ValueError("duplicate prepared jobs")
    return out


def controller_job(root: Path, case: dict, item: dict, template: dict, output: Path, fingerprint: str) -> dict:
    """Create the existing worker's input, without loading native or models."""
    if item["controller"] not in CONTROLLERS or item["task_id"] != case["task_id"] or case["status"] == "quarantined":
        raise ValueError("invalid controller/case binding")
    learned = item["controller"] != "official_adaptive"
    proposal = copy.deepcopy(template["proposal"])
    if item["controller"] == "dual16":
        from lns2_selector.runtime.structshell_dual16 import structshell_dual16_augmentation
        proposal["hybridstructpool"] = structshell_dual16_augmentation()
    environment = copy.deepcopy(template["environment"])
    environment.update(time_limit=item["budget_seconds"], max_repair_iterations=0)
    if environment.get("replan_algorithm") != "PP" or environment.get("use_sipp") is not True:
        raise ValueError("prepared worker requires official PP+SIPPS")
    return {
        "row": {"split": ".", "map_id": case["map_id"], "task_id": case["task_id"],
                "layout_mode": case["family"], "agent_count": case["static_audit"]["agent_count"],
                **case["files"]},
        "dataset_root": str(root), "policy": "realized_dynamic" if learned else "official_adaptive",
        "solver_seed": item["solver_seed"], "environment": environment, "proposal": proposal,
        "controller": "v2-full" if learned else "official_adaptive",
        "controller_bundle": str(root / "artifacts/initlns-closed-loop-controller-v2"),
        "frozen_models": str(root / template["frozen_models"]),
        "model_registration": copy.deepcopy(template["model_registration"]),
        "feature_backend": "native" if learned else "auto",
        "controller_runtime": "optimized" if learned else "reference",
        "verification_profile": "deployment" if learned else "audit",
        "proposal_state_verification": "sampled" if learned else "always",
        "feature_shadow_validation": False, "proposal_shadow_validation": False,
        "max_decisions": 0, "safety_max_decisions": None, "metric_iteration_budget": None,
        "wall_time_budget_seconds": item["budget_seconds"], "stopping_rule": "wall-clock",
        "repair_seed_policy": "episode_stream", "deterministic_pp_replay": False,
        "output_root": str(output / "first_phase"), "run_fingerprint": fingerprint, "resume": False,
    }


class PathJournal:
    """Atomically preserve independent path checkpoints before stage two."""
    def __init__(self, output: Path, binding: str, map_path: Path, scenario: Path):
        self.output, self.binding = output, binding
        self.grid = read_grid(map_path)
        self.map_path, self.scenario = map_path, scenario
        self.first: dict | None = None
        self.started: float | None = None

    def save(self, name: str, payload: dict) -> dict:
        envelope = {"schema": "lns2.path_quality_artifact.v1", "binding": self.binding, "payload": payload}
        envelope["sha256"] = json_fingerprint(envelope)
        path = self.output / f"{name}.json"
        if path.exists():
            if read_json(path) != envelope:
                raise ValueError("existing path artifact differs")
        else:
            write_json(path, envelope)
        return envelope

    def quality(self, state: dict) -> dict:
        agents = sorted(state["agents"], key=lambda a: a["id"])
        if [a["id"] for a in agents] != list(range(len(agents))):
            raise ValueError("invalid path agent ordering")
        rows = scenario_prefix(self.map_path, self.scenario, self.grid, len(agents))
        quality = audit_paths(self.grid, [(int(r[5]), int(r[4])) for r in rows],
                              [(int(r[7]), int(r[6])) for r in rows], [a["path"] for a in agents])
        if quality["feasible"] != state["feasible"] or quality["colliding_pairs"] != state["num_of_colliding_pairs"]:
            raise ValueError("path feasibility conflicts with observation")
        if quality["soc_steps"] != state["sum_of_costs"]:
            raise ValueError("path cost differs from observation")
        return quality

    def __call__(self, event: str, state: dict, started: float, completed: float, fingerprint: str) -> None:
        if event not in {"initial", "terminal"} or not math.isfinite(completed - started) or completed < started:
            raise ValueError("invalid path checkpoint event")
        if self.started is not None and self.started != started:
            raise ValueError("path checkpoint clock changed")
        self.started = started
        payload = {"event": event, "observation": state, "state_fingerprint": fingerprint,
                   "available_elapsed_seconds": completed - started, "quality": self.quality(state)}
        self.save(event, payload)
        if state["feasible"] and self.first is None:
            self.save("first_feasible", payload)
            self.first = copy.deepcopy(payload)


def read_artifact(path: Path, binding: str) -> dict:
    value = read_json(path)
    digest = value.pop("sha256", None)
    if value.get("schema") != "lns2.path_quality_artifact.v1" or value.get("binding") != binding or digest != json_fingerprint(value):
        raise ValueError("path artifact identity/SHA mismatch")
    return value["payload"]


def spec_fingerprint(spec: dict) -> str:
    return json_fingerprint({k: v for k, v in spec.items() if k != "binding"})


def prepare_episode_spec(root: Path, report: dict, item: dict, output: Path,
                         template: dict, native_sha256: str, process_timeout_seconds: float) -> dict:
    """Preparation only; a concrete native hash is mandatory before execution."""
    if len(native_sha256) != 64 or any(c not in "0123456789abcdef" for c in native_sha256):
        raise ValueError("concrete native SHA256 required")
    if item not in execution_schedule(report["cases"], report["protocol"]):
        raise ValueError("job is not in the prepared schedule")
    template_path = contained(root, report["runtime_template"]["manifest"])
    if sha256_file(template_path) != report["runtime_template"]["sha256"] or read_json(template_path) != template:
        raise ValueError("runtime template differs from preparation")
    case = next(c for c in report["cases"] if c["task_id"] == item["task_id"])
    output.resolve().relative_to((root / "build").resolve())
    spec = {"root": str(root.resolve()), "output": str(output.resolve()), "item": item, "case": case,
            "native_sha256": native_sha256, "process_timeout_seconds": process_timeout_seconds,
            "input_sha256": report["input_sha256"], "preflight_fingerprint": report["fingerprint"],
            "worker_job": controller_job(root, case, item, template, output, report["fingerprint"])}
    spec["binding"] = spec_fingerprint(spec)
    return spec


def _episode_child(spec: dict) -> None:
    from experiments.closed_loop_confirmation import _closed_loop_episode_worker
    from lns2_selector.solver.native import load_native_module, native_identity
    output, root = Path(spec["output"]), Path(spec["root"])
    journal = PathJournal(output, spec["binding"], contained(root, spec["case"]["files"]["map_file"]),
                          contained(root, spec["case"]["files"]["scenario_file"]))
    try:
        for relative, expected in spec["input_sha256"].items():
            if sha256_file(contained(root, relative)) != expected:
                raise ValueError("prepared input changed")
        module = load_native_module()
        if native_identity(module)["sha256"] != spec["native_sha256"]:
            raise ValueError("loaded native binary SHA mismatch")
        result = _closed_loop_episode_worker(spec["worker_job"], path_observer=journal)
        journal.save("first_phase_result", result)
        if result.get("status") != "ok":
            raise ValueError("first-phase worker failed; saved paths are diagnostic only")
        if journal.first is None:
            journal.save("result", {"status": "no_feasible_solution", "success_by_deadline": False})
            return
        first = journal.first
        if spec["item"]["protocol"] == "fixed_budget":
            handoff = continue_with_official_anytime(module, map_path=journal.map_path, scenario_path=journal.scenario,
                observation=first["observation"], expected_native_sha256=spec["native_sha256"],
                planning_started=journal.started, total_budget_seconds=spec["item"]["budget_seconds"],
                stage2_seed=spec["item"]["stage2_seed"], initial_feasible_elapsed_seconds=first["available_elapsed_seconds"])
            journal.save("final_paths", handoff)
            success = handoff["success_by_deadline"]
        else:
            elapsed = time.perf_counter() - journal.started
            success = first["available_elapsed_seconds"] <= spec["item"]["budget_seconds"]
            journal.save("final_paths", {"paths": [a["path"] for a in sorted(first["observation"]["agents"], key=lambda a: a["id"])],
                "final_quality": first["quality"], "dispatch_wall_seconds": elapsed,
                "first_feasible_elapsed_seconds": first["available_elapsed_seconds"], "success_by_deadline": success})
        ready = time.perf_counter() - journal.started
        dispatch = max(spec["item"]["budget_seconds"], ready) if spec["item"]["protocol"] == "fixed_budget" else ready
        journal.save("result", {"status": "completed", "success_by_deadline": success,
                                "paths_saved_elapsed_seconds": ready, "dispatch_wall_seconds": dispatch,
                                "first_feasible_elapsed_seconds": first["available_elapsed_seconds"]})
    except Exception as error:
        journal.save("result", {"status": "error", "error_type": type(error).__name__, "error": str(error),
                                "success_by_deadline": False})


def supervise_episode(spec: dict, *, authorized: bool = False, resume: bool = False,
                      child_entry: Any = _episode_child) -> dict:
    """Serial, isolated process; never auto-retry a timeout or a failed result."""
    if authorized is not True:
        raise PermissionError("case execution is not authorized")
    if spec.get("binding") != spec_fingerprint(spec):
        raise ValueError("prepared episode fingerprint changed")
    output = Path(spec["output"])
    budget = float(spec["item"]["budget_seconds"])
    timeout = float(spec["process_timeout_seconds"])
    if not math.isfinite(timeout) or not math.isfinite(budget) or timeout <= budget or budget <= 0:
        raise ValueError("invalid process fuse")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "run.lock"
    try:
        stream = lock.open("x")
    except FileExistsError as error:
        raise ValueError("episode already locked; stale locks require inspection") from error
    process = None
    try:
        with stream:
            stream.write(spec["binding"])
        binding_file = output / "binding.json"
        identity = {"binding": spec["binding"], "native_sha256": spec["native_sha256"], "item": spec["item"]}
        if binding_file.exists():
            if read_json(binding_file) != identity or not resume:
                raise ValueError("episode identity changed or resume not requested")
            for name in ("initial", "terminal", "first_feasible", "first_phase_result", "final_paths", "result", "supervisor"):
                if (output / f"{name}.json").exists():
                    read_artifact(output / f"{name}.json", spec["binding"])
            if (output / "supervisor.json").exists():
                return read_artifact(output / "supervisor.json", spec["binding"])
            raise ValueError("interrupted episode requires inspection; no silent mid-episode resume")
        write_json(binding_file, identity)
        process = multiprocessing.get_context("spawn").Process(target=child_entry, args=(spec,))
        process.start()
        process.join(timeout)
        expired = process.is_alive()
        if expired:
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
        status = "external_timeout" if expired else "worker_exit_error" if process.exitcode else "missing_result"
        result = None
        if (output / "result.json").exists():
            result = read_artifact(output / "result.json", spec["binding"])
            if not expired and process.exitcode == 0:
                status = result["status"]
        first = read_artifact(output / "first_feasible.json", spec["binding"]) if (output / "first_feasible.json").exists() else None
        summary = {"status": status, "process_exitcode": process.exitcode,
                   "first_feasible_preserved": first is not None,
                   "first_feasible_within_budget": first is not None and first["available_elapsed_seconds"] <= budget,
                   "success_by_deadline": bool(result and status == "completed" and result["success_by_deadline"]),
                   "error": status not in {"completed", "no_feasible_solution", "external_timeout"}}
        if result is not None and status == "completed":
            summary["dispatch_wall_seconds"] = result.get("dispatch_wall_seconds")
        # A preserved solution is not silently credited as an uninterrupted run.
        envelope = {"schema": "lns2.path_quality_artifact.v1", "binding": spec["binding"], "payload": summary}
        envelope["sha256"] = json_fingerprint(envelope)
        write_json(output / "supervisor.json", envelope)
        return summary
    finally:
        if process is not None:
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()
        lock.unlink(missing_ok=True)
