from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from experiments._common import (
    closed_loop_producer_identity,
    episode_id as closed_loop_episode_id,
    json_fingerprint,
    read_json,
    read_jsonl,
    registered_input,
    sha256_file,
    write_json,
)
from experiments.closed_loop_confirmation import (
    CLOSED_LOOP_SCHEMA,
    run_closed_loop_collection,
)
from experiments.closed_loop_trace_storage import trace_file_metadata
from experiments.run_output_guard import prepare_resumable_output
from experiments.stride_augcontrol_evaluation import _dataset_tasks
from lns2_selector.evaluation.trace_validation import validate_closed_loop_trace
from lns2_selector.runtime.structshell_single_family import (
    structshell_single_family_augmentation,
    validate_structshell_single_family_augmentation,
)


CONFIG_SCHEMA = "lns2.crossmap_profile_falsification_config.v1"
STATUS_SCHEMA = "lns2.crossmap_profile_falsification_status.v1"
REPORT_SCHEMA = "lns2.crossmap_profile_falsification_report.v1"
FRESHNESS_SCHEMA = "lns2.crossmap_profile_falsification_freshness_audit.v1"
EXPERIMENT_ID = "crossmap-profile-falsification-v1"
CONTROLLERS = ("v2_only", "component16", "hotspot16")
PROFILES = {
    "component16": "conflict_component",
    "hotspot16": "hotspot",
}
SOLVER_SEEDS = (19, 20)
WALL_TIME_SECONDS = 15.0
PROCESS_FUSE_SECONDS = 30.0
EXPECTED_GROUPS = (
    (
        "maze-32-32-4",
        "maze-32-32-4",
        "maze",
        "build/initlns-movingai-ood-dataset-v1",
        "movingai_ood",
        "maze-32-32-4__random_04__agents_0200",
        "component16",
    ),
    (
        "random-32-32-20-high-load",
        "random-32-32-20",
        "random",
        "build/stride-stage2-movingai-v1",
        "balanced_wall_clock",
        "random-32-32-20__random_01__agents_0400",
        "hotspot16",
    ),
)
EXPECTED_KEY_ORDER = (
    "maze-32-32-4@19",
    "random-32-32-20-high-load@19",
    "maze-32-32-4@20",
    "random-32-32-20-high-load@20",
)
EXPECTED_FRESHNESS = {
    "maze_consumed_solver_seeds": (1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18),
    "random_consumed_solver_seeds": (13, 14, 15, 16, 17, 18, 101, 202, 303),
}
EXPECTED_SEED_CORRECTION = {
    "rule": (
        "ascending_from_18_choose_first_two_seeds_with_zero_exact_task_artifacts_"
        "for_both_registered_tasks"
    ),
    "outcome_fields_read": False,
    "excluded": [
        {
            "solver_seed": 18,
            "reason": "qualification_only_exact_task_artifacts",
            "exact_manifest_match_count": 20,
            "controller_episode_match_count": 0,
            "source_experiment_ids": [
                "stride-structshell-overall-rollback-screen-v1",
                "stride-structshell-rollback-aware-ttf-v1",
                "stride-structshell-rollback-aware-ttf-v1-r2",
            ],
            "matches": [
                {
                    "source_experiment_id": (
                        "stride-structshell-overall-rollback-screen-v1"
                    ),
                    "qualification_manifest_match_count": 10,
                },
                {
                    "source_experiment_id": (
                        "stride-structshell-rollback-aware-ttf-v1"
                    ),
                    "qualification_manifest_match_count": 2,
                },
                {
                    "source_experiment_id": (
                        "stride-structshell-rollback-aware-ttf-v1-r2"
                    ),
                    "qualification_manifest_match_count": 8,
                },
            ],
        }
    ],
    "selected_solver_seeds": [19, 20],
}
STATUS_FILENAME = "collection_status.json"
REPORT_FILENAME = "falsification_report.json"
FRESHNESS_FILENAME = "freshness_audit.json"
FRESHNESS_SEARCH_SCOPE = (
    "workspace/build/**/episodes/**/*",
    "workspace/build/**/*manifest*.jsonl",
)
REGISTRATION_TIME_FULL_WORKSPACE_AUDIT = {
    "backend": "ripgrep_path_executable",
    "tool_sha256": (
        "0f5f7e6473e5374344ca8a895b89a79d7f25f68493d977ed539ddd825ccd0ffb"
    ),
    "version": "ripgrep 15.2.0 (rev e89fff89ac)",
    "inventory_file_count": 39,
    "inventory_sha256": (
        "b6b19666c4f866c685af751e1129f958c0ec8e6498a23192880b44bf564f8b75"
    ),
    "match_count": 0,
    "matches": [],
    "scan_errors": [],
    "task_ids": [
        "maze-32-32-4__random_04__agents_0200",
        "random-32-32-20__random_01__agents_0400",
    ],
    "solver_seeds": [19, 20],
    "excluded_output_root": "build/crossmap-profile-falsification-v1-r2",
    "completed_before_any_reset_or_controller": True,
    "identity_scope": "single_immediate_experiment_identity",
    "future_rerun_policy": (
        "new_experiment_identity_and_new_registration_audit_required"
    ),
    "resume_policy": "current_r2_only",
}
RIPGREP_DISCOVERY_CONTRACT = {
    "standard_executable_names": ["rg.exe", "rg"],
    "packaged_path_filename": "rg",
    "packaged_magic_hex": "7f454c46",
    "posix_standard_requires_elf": True,
    "copy_directory": "system_temporary_directory",
    "copy_name_template": "crossmap-profile-rg-<sha256>",
    "copy_mode_octal": "0700",
    "version_probe_timeout_seconds": 2.0,
    "search_timeout_seconds": 30.0,
    "no_verified_tool_policy": "fail_closed_before_reset",
}


def load_config(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    root = Path(__file__).resolve().parents[1]
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("cross-map profile falsification config must be an object")
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
    ):
        raise ValueError("cross-map profile falsification identity changed")

    runtime = dict(config.get("runtime") or {})
    if runtime != {
        "stopping_rule": "wall-clock",
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "wall_time_budget_seconds": WALL_TIME_SECONDS,
        "environment_time_limit_seconds": WALL_TIME_SECONDS,
        "episode_process_timeout_seconds": PROCESS_FUSE_SECONDS,
        "workers_for_qualification": 1,
        "workers_for_timed_episodes": 1,
        "execution_order": "rotating_strict_three_controller_serial",
        "stop_after_first_failed_completed_key": True,
    }:
        raise ValueError("cross-map profile falsification runtime contract changed")

    contract = dict(config.get("controller_contract") or {})
    if (
        contract.get("base")
        != "frozen_v2_full_native_features_optimized_copeland"
        or int(contract.get("nominal_size", -1)) != 16
        or int(
            contract.get(
                "single_family_maximum_added_candidates_per_decision", -1
            )
        )
        != 1
        or dict(contract.get("profiles") or {}) != PROFILES
        or "official_adaptive" not in set(map(str, contract.get("excluded") or ()))
        or "threshold_tuning" not in set(map(str, contract.get("excluded") or ()))
    ):
        raise ValueError("cross-map profile fixed16 controller contract changed")

    cohort = dict(config.get("cohort") or {})
    groups = list(cohort.get("groups") or ())
    observed_groups = tuple(
        (
            str(group.get("id")),
            str(group.get("map_id")),
            str(group.get("family")),
            str(group.get("dataset")),
            str(group.get("split")),
            str(group.get("task")),
            str(group.get("expected_winner")),
        )
        for group in groups
    )
    if (
        tuple(map(int, cohort.get("solver_seeds") or ())) != SOLVER_SEEDS
        or int(cohort.get("paired_key_count", -1)) != 4
        or int(cohort.get("episode_count", -1)) != 12
        or tuple(map(str, cohort.get("key_order") or ())) != EXPECTED_KEY_ORDER
        or observed_groups != EXPECTED_GROUPS
    ):
        raise ValueError("cross-map profile falsification cohort changed")

    freshness = dict(config.get("freshness_audit") or {})
    if (
        freshness.get("definition")
        != "no_existing_episode_artifact_for_the_exact_task_and_solver_seed_before_registration"
        or tuple(map(int, freshness.get("selected_solver_seeds") or ()))
        != SOLVER_SEEDS
        or any(
            tuple(map(int, freshness.get(field) or ())) != expected
            for field, expected in EXPECTED_FRESHNESS.items()
        )
        or any(
            seed in set(EXPECTED_FRESHNESS["maze_consumed_solver_seeds"])
            or seed in set(EXPECTED_FRESHNESS["random_consumed_solver_seeds"])
            for seed in SOLVER_SEEDS
        )
        or tuple(map(str, freshness.get("artifact_search_scope") or ()))
        != FRESHNESS_SEARCH_SCOPE
        or freshness.get("output_root_excluded") is not True
        or freshness.get("inventory_hash_definition")
        != "sha256_of_sorted_exact_seed_artifact_candidate_paths"
        or freshness.get("resume_policy")
        != "read_and_verify_frozen_audit_without_workspace_rescan"
        or int(freshness.get("required_match_count", -1)) != 0
        or dict(freshness.get("pre_controller_identity_correction") or {})
        != EXPECTED_SEED_CORRECTION
        or dict(freshness.get("registration_time_full_workspace_audit") or {})
        != REGISTRATION_TIME_FULL_WORKSPACE_AUDIT
    ):
        raise ValueError("cross-map profile solver-seed freshness audit changed")

    profile = dict(config.get("frozen_profile_hypothesis") or {})
    if profile != {
        "diagnostic_only_not_executed_by_this_runner": True,
        "active_conflict_agent_ratio_abstain_below": 0.35,
        "component16_minimum_fixed_size_to_lcc_ratio": 0.15,
        "hotspot16_minimum_active_conflict_agent_ratio": 0.60,
        "hotspot16_maximum_fixed_size_to_lcc_ratio": 0.10,
        "hotspot16_minimum_conflict_event_to_pair_ratio": 2.0,
        "otherwise": "v2_only",
        "post_result_threshold_changes_allowed": False,
    }:
        raise ValueError("cross-map profile hypothesis changed")

    gate = dict(config.get("falsification_gate") or {})
    if gate != {
        "maze_expected_winner": "component16",
        "random_expected_winner": "hotspot16",
        "expected_winner_must_be_strictly_fastest_on_each_key": True,
        "expected_winner_success_must_be_noninferior_on_each_key": True,
        "tie_is_failure": True,
        "first_failed_key_stops_remaining_schedule": True,
        "threshold_changes_after_failure_allowed": False,
        "claim_boundary": (
            "survival_only_allows_a_larger_fresh_test_and_never_a_speed_or_default_claim"
        ),
    }:
        raise ValueError("cross-map profile falsification gate changed")

    inputs = dict(config.get("inputs") or {})
    controller_manifest = registered_input(
        root,
        dict(inputs.get("controller_manifest") or {}),
        label="cross-map profile V2 controller manifest",
    )
    expected_bundle = (root / str(config.get("controller_bundle"))).resolve()
    if controller_manifest.parent != expected_bundle:
        raise ValueError("cross-map profile controller bundle changed")

    for group in groups:
        group_id = str(group["id"])
        runtime_key = str(group.get("runtime_config"))
        manifest_key = str(group.get("manifest"))
        task_key = str(group.get("task_input"))
        runtime_path = registered_input(
            root,
            dict(inputs.get(runtime_key) or {}),
            label=f"{group_id} runtime config",
        )
        manifest_path = registered_input(
            root,
            dict(inputs.get(manifest_key) or {}),
            label=f"{group_id} dataset manifest",
        )
        task_path = registered_input(
            root,
            dict(inputs.get(task_key) or {}),
            label=f"{group_id} task identity",
        )
        dataset = (root / str(group["dataset"])).resolve()
        split = str(group["split"])
        if manifest_path != (dataset / split / "manifest.jsonl").resolve():
            raise ValueError(f"{group_id} dataset manifest location changed")
        expected_task = (dataset / split / "tasks" / f"{group['task']}.json").resolve()
        if task_path != expected_task:
            raise ValueError(f"{group_id} task identity location changed")
        tasks = _dataset_tasks(dataset, split)
        task_id = str(group["task"])
        if task_id not in tasks:
            raise ValueError(f"{group_id} falsification task is absent")
        if str(tasks[task_id].get("map_id")) != str(group["map_id"]):
            raise ValueError(f"{group_id} falsification task map changed")
        group["_runtime_path"] = str(runtime_path)
    return path, root, config


def key_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups = {
        str(group["id"]): dict(group) for group in config["cohort"]["groups"]
    }
    result = []
    for key_index, label in enumerate(config["cohort"]["key_order"]):
        group_id, separator, raw_seed = str(label).rpartition("@")
        if not separator or group_id not in groups:
            raise ValueError(f"invalid falsification key identity: {label}")
        seed = int(raw_seed)
        if seed not in SOLVER_SEEDS:
            raise ValueError(f"unregistered falsification solver seed: {seed}")
        group = groups[group_id]
        result.append(
            {
                "key_index": key_index,
                "key_id": str(label),
                "group_id": group_id,
                "map_id": str(group["map_id"]),
                "family": str(group["family"]),
                "task_id": str(group["task"]),
                "solver_seed": seed,
                "expected_winner": str(group["expected_winner"]),
            }
        )
    return result


def schedule(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    controllers = tuple(map(str, config["controllers"]))
    for key in key_rows(config):
        offset = int(key["key_index"]) % len(controllers)
        for position in range(len(controllers)):
            controller = controllers[(offset + position) % len(controllers)]
            rows.append(
                {
                    **dict(key),
                    "within_key_position": position,
                    "controller": controller,
                }
            )
    return rows


def plan(config_path: str | Path) -> dict[str, Any]:
    _path, _root, config = load_config(config_path)
    rows = schedule(config)
    first_positions = [
        row["controller"] for row in rows if row["within_key_position"] == 0
    ]
    return {
        "schema": STATUS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "map_count": 2,
        "task_count": 2,
        "paired_key_count": 4,
        "controller_count": 3,
        "timed_episode_count": len(rows),
        "qualification_reset_count": 4,
        "solver_seeds": list(SOLVER_SEEDS),
        "controllers": list(CONTROLLERS),
        "keys": [row["key_id"] for row in key_rows(config)],
        "first_position_rotation": first_positions,
        "wall_time_budget_seconds": WALL_TIME_SECONDS,
        "episode_process_fuse_seconds": PROCESS_FUSE_SECONDS,
        "maximum_registered_timed_seconds": len(rows) * WALL_TIME_SECONDS,
        "maximum_timed_process_fuse_seconds": len(rows) * PROCESS_FUSE_SECONDS,
        "maximum_qualification_process_fuse_seconds": 4 * PROCESS_FUSE_SECONDS,
        "maximum_qualification_plus_timed_process_fuse_seconds": (
            len(rows) + 4
        )
        * PROCESS_FUSE_SECONDS,
        "strict_serial_timing": True,
        "rotating_controller_order": True,
        "early_stop_after_first_failed_completed_key": True,
        "solver_or_controller_invoked": False,
        "router_executed": False,
        "official_adaptive": False,
        "bootstrap": False,
        "auc_gate": False,
    }


def controller_kwargs(
    root: Path, config: Mapping[str, Any], controller: str
) -> dict[str, Any]:
    runtime = dict(config["runtime"])
    result: dict[str, Any] = {
        "controller": "v2-full",
        "controller_bundle": str(
            (root / str(config["controller_bundle"])).resolve()
        ),
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "stopping_rule": "wall-clock",
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "wall_time_budget_seconds": float(runtime["wall_time_budget_seconds"]),
        "environment_time_limit_seconds": float(
            runtime["environment_time_limit_seconds"]
        ),
        "episode_process_timeout_seconds": float(
            runtime["episode_process_timeout_seconds"]
        ),
    }
    if controller == "v2_only":
        return result
    try:
        profile = PROFILES[controller]
    except KeyError as error:
        raise ValueError(
            f"unknown cross-map profile controller: {controller}"
        ) from error
    augmentation = validate_structshell_single_family_augmentation(
        structshell_single_family_augmentation(profile, 16)
    )
    assert augmentation is not None
    result["hybridstructpool_augmentation"] = augmentation
    return result


def _nested_objects(value: Any, location: str = "$") -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        rows.append((location, value))
        for key, child in value.items():
            rows.extend(_nested_objects(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(_nested_objects(child, f"{location}[{index}]"))
    return rows


def _artifact_object_match(
    value: Mapping[str, Any], task_ids: set[str], solver_seeds: set[int]
) -> tuple[str, int] | None:
    task_id = str(value.get("task_id", ""))
    try:
        solver_seed = int(value.get("solver_seed", -1))
    except (TypeError, ValueError):
        return None
    if task_id not in task_ids or solver_seed not in solver_seeds:
        return None
    if not any(
        field in value
        for field in (
            "status",
            "trace_file",
            "summary",
            "state_fingerprint",
            "initial_conflicts",
        )
    ):
        return None
    return task_id, solver_seed


def _ripgrep_version(executable: Path) -> str | None:
    try:
        checked = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=float(
                RIPGREP_DISCOVERY_CONTRACT["version_probe_timeout_seconds"]
            ),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if checked.returncode != 0:
        return None
    lines = checked.stdout.splitlines()
    return lines[0].strip() if lines else "verified-ripgrep"


def _copy_packaged_ripgrep(source: Path, digest: str) -> Path:
    destination = Path(tempfile.gettempdir()) / f"crossmap-profile-rg-{digest}"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        if not destination.is_file() or sha256_file(destination) != digest:
            shutil.copyfile(source, temporary)
            os.chmod(temporary, 0o700)
            if sha256_file(temporary) != digest:
                raise RuntimeError("copied packaged ripgrep hash mismatch")
            os.replace(temporary, destination)
        os.chmod(destination, 0o700)
        if sha256_file(destination) != digest:
            raise RuntimeError("verified packaged ripgrep hash mismatch")
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return destination


def _discover_ripgrep() -> tuple[Path, dict[str, Any]]:
    """Return a verified ripgrep or fail closed before any solver reset."""

    seen: set[str] = set()
    for name in RIPGREP_DISCOVERY_CONTRACT["standard_executable_names"]:
        candidate_value = shutil.which(str(name))
        if candidate_value is None or candidate_value in seen:
            continue
        seen.add(candidate_value)
        candidate = Path(candidate_value)
        if os.name == "posix" and RIPGREP_DISCOVERY_CONTRACT[
            "posix_standard_requires_elf"
        ]:
            try:
                with candidate.open("rb") as stream:
                    if stream.read(4).hex() != RIPGREP_DISCOVERY_CONTRACT[
                        "packaged_magic_hex"
                    ]:
                        continue
            except OSError:
                continue
        version = _ripgrep_version(candidate)
        if version is None:
            continue
        try:
            digest = sha256_file(candidate)
        except OSError:
            continue
        return candidate, {
            "backend": "ripgrep_path_executable",
            "source_path": str(candidate),
            "executable_path": str(candidate),
            "tool_sha256": digest,
            "version": version,
            "packaged_elf_copied": False,
        }

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for entry in path_entries:
        if not entry:
            continue
        source = Path(entry.strip('"')) / str(
            RIPGREP_DISCOVERY_CONTRACT["packaged_path_filename"]
        )
        try:
            if not source.is_file():
                continue
            with source.open("rb") as stream:
                magic = stream.read(4)
        except OSError:
            continue
        if magic.hex() != RIPGREP_DISCOVERY_CONTRACT["packaged_magic_hex"]:
            continue
        try:
            digest = sha256_file(source)
            executable = _copy_packaged_ripgrep(source, digest)
        except OSError:
            continue
        version = _ripgrep_version(executable)
        if version is None:
            continue
        return executable, {
            "backend": "ripgrep_packaged_elf_copy",
            "source_path": str(source),
            "executable_path": str(executable),
            "tool_sha256": digest,
            "version": version,
            "packaged_elf_copied": True,
            "elf_magic_hex": magic.hex(),
            "copy_mode_octal": "0700",
        }

    raise RuntimeError(
        "no verified ripgrep is available; freshness audit fails closed before reset"
    )


def scan_freshness_artifacts(
    workspace: str | Path,
    output: str | Path,
    *,
    task_ids: set[str],
    solver_seeds: set[int],
    config_sha256: str,
) -> dict[str, Any]:
    """Freeze a machine-readable zero-artifact audit before the first reset."""

    root = Path(workspace).resolve()
    output_root = Path(output).resolve()
    inventory = hashlib.sha256()
    inventory_count = 0
    matches: list[dict[str, Any]] = []
    scan_errors: list[dict[str, str]] = []
    rg, search_tool = _discover_ripgrep()
    search_backend = str(search_tool["backend"])
    common_args = [
        "--hidden",
        "--no-ignore",
        "-g",
        "!**/.git/**",
    ]
    manifest_args = [
        *common_args,
        "-g",
        "**/*manifest*.jsonl",
    ]
    seed_pattern = (
        r'"solver_seed"\s*:\s*('
        + "|".join(map(str, sorted(solver_seeds)))
        + r")([^0-9]|$)"
    )
    searched = subprocess.run(
        [str(rg), "-l", seed_pattern, *manifest_args, "build"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=float(RIPGREP_DISCOVERY_CONTRACT["search_timeout_seconds"]),
    )
    if searched.returncode not in {0, 1}:
        raise RuntimeError(
            f"freshness ripgrep search failed: {searched.stderr.strip()}"
        )
    relevant_text = {
        root / line.strip()
        for line in searched.stdout.splitlines()
        if line.strip()
    }
    episode_args = list(common_args)
    for seed in sorted(solver_seeds):
        episode_args.extend(
            ["-g", f"**/episodes/**/*__seed_{seed:04d}__*"]
        )
    listed_episodes = subprocess.run(
        [str(rg), "--files", *episode_args, "build"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=float(RIPGREP_DISCOVERY_CONTRACT["search_timeout_seconds"]),
    )
    if listed_episodes.returncode not in {0, 1}:
        raise RuntimeError(
            "freshness ripgrep episode inventory failed: "
            f"{listed_episodes.stderr.strip()}"
        )
    raw_candidates = list(relevant_text) + [
        root / line.strip()
        for line in listed_episodes.stdout.splitlines()
        if line.strip()
    ]
    try:
        output_relative = output_root.relative_to(root)
    except ValueError:
        output_relative = None
    candidates = sorted(
        {
            path
            for path in raw_candidates
            if output_relative is None
            or not (
                path.relative_to(root) == output_relative
                or output_relative in path.relative_to(root).parents
            )
        },
        key=lambda value: value.relative_to(root).as_posix(),
    )
    task_tokens = {task.encode("utf-8"): task for task in task_ids}
    seed_tokens = {
        seed: (f"seed_{seed:04d}".encode("ascii"), f'"solver_seed":{seed}'.encode("ascii"))
        for seed in solver_seeds
    }
    seen_matches: set[tuple[str, str, str, int]] = set()
    for artifact in candidates:
        relative = artifact.relative_to(root).as_posix()
        inventory.update(relative.encode("utf-8") + b"\n")
        inventory_count += 1

        relative_bytes = relative.encode("utf-8")
        for task_token, task_id in task_tokens.items():
            if task_token not in relative_bytes:
                continue
            for seed, tokens in seed_tokens.items():
                if tokens[0] in relative_bytes:
                    marker = (relative, "path", task_id, seed)
                    if marker not in seen_matches:
                        seen_matches.add(marker)
                        matches.append(
                            {
                                "path": relative,
                                "location": "path",
                                "task_id": task_id,
                                "solver_seed": seed,
                            }
                        )

        if artifact.suffix.lower() not in {".json", ".jsonl"}:
            continue
        if artifact not in relevant_text:
            continue
        try:
            content = artifact.read_bytes()
        except OSError as error:
            scan_errors.append(
                {"path": relative, "error": f"{type(error).__name__}: {error}"}
            )
            continue
        if not any(token in content for token in task_tokens) or b"solver_seed" not in content:
            continue
        try:
            if artifact.suffix.lower() == ".jsonl":
                documents = [
                    (f"line:{line_number}", json.loads(line))
                    for line_number, line in enumerate(
                        content.decode("utf-8").splitlines(), start=1
                    )
                    if line.strip()
                ]
            else:
                documents = [("json", json.loads(content.decode("utf-8")))]
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            scan_errors.append(
                {"path": relative, "error": f"{type(error).__name__}: {error}"}
            )
            continue
        for document_location, document in documents:
            for object_location, value in _nested_objects(document):
                matched = _artifact_object_match(value, task_ids, solver_seeds)
                if matched is None:
                    continue
                task_id, seed = matched
                location = f"{document_location}:{object_location}"
                marker = (relative, location, task_id, seed)
                if marker in seen_matches:
                    continue
                seen_matches.add(marker)
                matches.append(
                    {
                        "path": relative,
                        "location": location,
                        "task_id": task_id,
                        "solver_seed": seed,
                    }
                )

    matches.sort(
        key=lambda row: (
            str(row["path"]),
            str(row["location"]),
            str(row["task_id"]),
            int(row["solver_seed"]),
        )
    )
    scan_errors.sort(key=lambda row: (row["path"], row["error"]))
    payload = {
        "schema": FRESHNESS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": config_sha256,
        "workspace_root": str(root),
        "excluded_output_root": str(output_root),
        "search_scope": list(FRESHNESS_SEARCH_SCOPE),
        "search_backend": search_backend,
        "search_tool": search_tool,
        "task_ids": sorted(task_ids),
        "solver_seeds": sorted(solver_seeds),
        "inventory_definition": (
            "sha256_of_sorted_exact_seed_artifact_candidate_paths"
        ),
        "inventory_file_count": inventory_count,
        "inventory_sha256": inventory.hexdigest(),
        "matches": matches,
        "match_count": len(matches),
        "scan_errors": scan_errors,
        "passed": not matches and not scan_errors,
        "resume_policy": "read_and_verify_frozen_audit_without_workspace_rescan",
        "pre_controller_identity_correction": EXPECTED_SEED_CORRECTION,
        "ripgrep_discovery": RIPGREP_DISCOVERY_CONTRACT,
    }
    payload["audit_payload_fingerprint"] = json_fingerprint(payload)
    return payload


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _valid_freshness_search_tool(audit: Mapping[str, Any]) -> bool:
    tool = audit.get("search_tool")
    if not isinstance(tool, dict):
        return False
    backend = str(tool.get("backend", ""))
    if (
        backend != audit.get("search_backend")
        or backend
        not in {"ripgrep_path_executable", "ripgrep_packaged_elf_copy"}
        or not str(tool.get("source_path", ""))
        or not str(tool.get("executable_path", ""))
        or not _valid_sha256(tool.get("tool_sha256"))
        or not str(tool.get("version", ""))
    ):
        return False
    if backend == "ripgrep_path_executable":
        return (
            tool.get("packaged_elf_copied") is False
            and tool.get("source_path") == tool.get("executable_path")
        )
    digest = str(tool["tool_sha256"])
    executable = Path(str(tool["executable_path"]))
    return (
        tool.get("packaged_elf_copied") is True
        and tool.get("elf_magic_hex")
        == RIPGREP_DISCOVERY_CONTRACT["packaged_magic_hex"]
        and tool.get("copy_mode_octal")
        == RIPGREP_DISCOVERY_CONTRACT["copy_mode_octal"]
        and executable.name == f"crossmap-profile-rg-{digest}"
        and executable.parent == Path(tempfile.gettempdir())
    )


def _registration_freshness_payload(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    registered = dict(
        config["freshness_audit"]["registration_time_full_workspace_audit"]
    )
    expected_output = (root / str(registered["excluded_output_root"])).resolve()
    if output.resolve() != expected_output:
        raise ValueError(
            "registration-time freshness audit is valid only for the frozen r2 output"
        )
    return {
        "schema": FRESHNESS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": config_sha256,
        "workspace_root": str(root.resolve()),
        "excluded_output_root": str(expected_output),
        "search_scope": list(FRESHNESS_SEARCH_SCOPE),
        "search_backend": str(registered["backend"]),
        "search_tool_sha256": str(registered["tool_sha256"]),
        "search_tool_version": str(registered["version"]),
        "task_ids": list(registered["task_ids"]),
        "solver_seeds": list(registered["solver_seeds"]),
        "inventory_definition": (
            "sha256_of_sorted_exact_seed_artifact_candidate_paths"
        ),
        "inventory_file_count": int(registered["inventory_file_count"]),
        "inventory_sha256": str(registered["inventory_sha256"]),
        "matches": list(registered["matches"]),
        "match_count": int(registered["match_count"]),
        "scan_errors": list(registered["scan_errors"]),
        "passed": True,
        "resume_policy": "current_r2_only_without_workspace_rescan",
        "pre_controller_identity_correction": EXPECTED_SEED_CORRECTION,
        "registration_time_full_workspace_audit": registered,
    }


def _validate_freshness_audit(
    audit: Mapping[str, Any],
    *,
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    payload = dict(audit)
    fingerprint = payload.pop("audit_payload_fingerprint", None)
    expected = _registration_freshness_payload(
        root, output, config, config_sha256
    )
    if payload != expected or fingerprint != json_fingerprint(payload):
        raise ValueError("cross-map profile frozen freshness audit is invalid")
    return dict(audit)


def _prepare_freshness_audit(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    *,
    config_sha256: str,
    resumed: bool,
    status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = output / FRESHNESS_FILENAME
    if path.is_file():
        if not resumed:
            raise ValueError("fresh output unexpectedly contains a freshness audit")
        audit = _validate_freshness_audit(
            read_json(path),
            root=root,
            output=output,
            config=config,
            config_sha256=config_sha256,
        )
        expected_file_sha = dict(status or {}).get("freshness_audit_sha256")
        if expected_file_sha is not None and expected_file_sha != sha256_file(path):
            raise ValueError("cross-map profile freshness audit file hash changed")
        return audit
    if resumed:
        raise ValueError(
            "resumed cross-map profile output is missing its frozen freshness audit"
        )
    audit = _registration_freshness_payload(root, output, config, config_sha256)
    audit["audit_payload_fingerprint"] = json_fingerprint(audit)
    write_json(path, audit)
    return _validate_freshness_audit(
        read_json(path),
        root=root,
        output=output,
        config=config,
        config_sha256=config_sha256,
    )


def _freshness_status_base(
    base: Mapping[str, Any], output: Path, audit: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **dict(base),
        "freshness_audit_sha256": sha256_file(output / FRESHNESS_FILENAME),
        "freshness_inventory_sha256": str(audit["inventory_sha256"]),
        "freshness_match_count": int(audit["match_count"]),
    }


def _group(config: Mapping[str, Any], group_id: str) -> dict[str, Any]:
    return next(
        dict(group)
        for group in config["cohort"]["groups"]
        if str(group["id"]) == group_id
    )


def _runtime_config_path(
    output: Path, group: Mapping[str, Any], solver_seed: int
) -> Path:
    source = Path(str(group["_runtime_path"])).resolve()
    payload = read_json(source)
    if not isinstance(payload, dict):
        raise ValueError(f"{group['id']} runtime config must be an object")
    payload["solver_seeds"] = [solver_seed]
    destination = (
        output
        / "runtime_configs"
        / f"{group['id']}__seed_{solver_seed:04d}.json"
    )
    if destination.is_file():
        if read_json(destination) != payload:
            raise ValueError(f"{group['id']} materialized runtime config changed")
    else:
        write_json(destination, payload)
    return destination


def _key_root(output: Path, item: Mapping[str, Any]) -> Path:
    return (
        output
        / "maps"
        / str(item["group_id"])
        / f"seed_{int(item['solver_seed']):04d}"
    )


def _manifest_path(output: Path, item: Mapping[str, Any]) -> Path:
    return (
        _key_root(output, item)
        / str(item["controller"])
        / "realized_dynamic_manifest.jsonl"
    )


def _manifest_row(output: Path, item: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _manifest_path(output, item)
    matches = [
        row
        for row in (read_jsonl(path) if path.is_file() else [])
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if len(matches) > 1:
        raise ValueError("cross-map profile manifest is ambiguous")
    return dict(matches[0]) if matches else None


def _manifest_prefix_length(output: Path, rows: list[dict[str, Any]]) -> int:
    present = [_manifest_row(output, item) is not None for item in rows]
    prefix = 0
    while prefix < len(present) and present[prefix]:
        prefix += 1
    if any(present[prefix:]):
        raise ValueError("falsification manifests are not a strict schedule prefix")
    return prefix


def _progress_status(
    output: Path,
    rows: list[dict[str, Any]],
    base: Mapping[str, Any],
    *,
    terminal_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    present = sum(_manifest_row(output, item) is not None for item in rows)
    result = {
        **dict(base),
        "completed_schedule_entries": present,
        "executed_schedule_entries": present,
        "cancelled_schedule_entries": 0,
        "complete": False,
    }
    if terminal_failure is not None:
        result["terminal_failure"] = dict(terminal_failure)
    return result


def _completed_status(
    output: Path,
    rows: list[dict[str, Any]],
    base: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    executed = sum(_manifest_row(output, item) is not None for item in rows)
    cancelled = len(rows) - executed
    result = {
        **dict(base),
        # A scientific stop resolves the untouched suffix as cancelled.  The
        # executed/cancelled fields preserve the distinction explicitly.
        "completed_schedule_entries": len(rows),
        "executed_schedule_entries": executed,
        "cancelled_schedule_entries": cancelled,
        "complete": True,
        "decision": str(report["decision"]),
        "scientific_stop": str(report["decision"])
        == "stop_profile_hypothesis_falsified",
        "report_sha256": sha256_file(output / REPORT_FILENAME),
    }
    return result


def _qualify_key(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    key: Mapping[str, Any],
    *,
    resume: bool,
) -> Path:
    group = _group(config, str(key["group_id"]))
    seed = int(key["solver_seed"])
    task_id = str(key["task_id"])
    job_key = {(task_id, seed)}
    runtime = _runtime_config_path(output, group, seed)
    qualification = _key_root(output, key) / "qualification"
    run_closed_loop_collection(
        root / str(group["dataset"]),
        runtime,
        qualification,
        phase="qualify",
        workers=1,
        resume=resume and qualification.joinpath("run_config.json").is_file(),
        task_ids=[task_id],
        cohort_job_keys=job_key,
        job_keys=job_key,
        qualification_process_timeout_seconds=PROCESS_FUSE_SECONDS,
        use_global_collection_lock=False,
        **controller_kwargs(root, config, "v2_only"),
    )
    report = read_json(qualification / "qualification_report.json")
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise RuntimeError(f"{key['key_id']} reset qualification failed")
    return qualification


def _run_episode(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    item: Mapping[str, Any],
    qualification: Path,
) -> dict[str, Any]:
    group = _group(config, str(item["group_id"]))
    seed = int(item["solver_seed"])
    task_id = str(item["task_id"])
    job_key = {(task_id, seed)}
    runtime = _runtime_config_path(output, group, seed)
    collection = _key_root(output, item) / str(item["controller"])
    kwargs = controller_kwargs(root, config, str(item["controller"]))
    run_closed_loop_collection(
        root / str(group["dataset"]),
        runtime,
        collection,
        phase="qualify",
        workers=1,
        resume=collection.joinpath("run_config.json").is_file(),
        task_ids=[task_id],
        cohort_job_keys=job_key,
        job_keys=job_key,
        qualification_source=qualification,
        qualification_process_timeout_seconds=PROCESS_FUSE_SECONDS,
        use_global_collection_lock=False,
        **kwargs,
    )
    run_closed_loop_collection(
        root / str(group["dataset"]),
        runtime,
        collection,
        phase="realized_dynamic",
        workers=1,
        resume=True,
        task_ids=[task_id],
        cohort_job_keys=job_key,
        job_keys=job_key,
        qualification_source=qualification,
        qualification_process_timeout_seconds=PROCESS_FUSE_SECONDS,
        use_global_collection_lock=False,
        **kwargs,
    )
    row = _manifest_row(output, item)
    if row is None:
        raise RuntimeError("cross-map profile episode produced no manifest")
    return row


def _audit_runtime_config_path(
    output: Path, group: Mapping[str, Any], solver_seed: int
) -> Path:
    source = Path(str(group["_runtime_path"])).resolve()
    expected = read_json(source)
    expected["solver_seeds"] = [solver_seed]
    path = (
        output
        / "runtime_configs"
        / f"{group['id']}__seed_{solver_seed:04d}.json"
    )
    if not path.is_file() or read_json(path) != expected:
        raise ValueError("cross-map profile materialized runtime identity changed")
    return path


def _expected_effective_configuration(
    root: Path,
    config: Mapping[str, Any],
    item: Mapping[str, Any],
    runtime_path: Path,
) -> dict[str, Any]:
    task_id = str(item["task_id"])
    seed = int(item["solver_seed"])
    kwargs = controller_kwargs(root, config, str(item["controller"]))
    expected = read_json(runtime_path)
    expected["deterministic_pp_replay"] = False
    expected["environment"] = dict(expected["environment"])
    expected["environment"]["time_limit"] = WALL_TIME_SECONDS
    expected["wall_time_budget_seconds"] = WALL_TIME_SECONDS
    expected["episode_process_timeout_seconds"] = PROCESS_FUSE_SECONDS
    expected["stopping_rule"] = "wall-clock"
    expected["max_decisions"] = 0
    expected["metric_iteration_budget"] = None
    expected["environment"]["max_repair_iterations"] = 0
    expected_proposal = dict(expected["proposal"])
    augmentation = kwargs.get("hybridstructpool_augmentation")
    if augmentation is not None:
        expected_proposal["hybridstructpool"] = augmentation
    expected["proposal"] = expected_proposal
    expected.update(
        {
            "task_ids_override": [task_id],
            "cohort_job_keys_override": [[task_id, seed]],
            "controller": "v2-full",
            "feature_backend": "native",
            "controller_runtime": "optimized",
            "verification_profile": "deployment",
            "diagnostic_shadow_controllers": [],
            "deterministic_pp_replay": False,
            "repair_seed_policy": "episode_stream",
            "controller_bundle": str(
                (root / str(config["controller_bundle"])).resolve()
            ),
            "diagnostic_shadow_bundles": {},
            "feature_shadow_validation": False,
            "v3_s3_bundle": None,
            "episode_override_fingerprints": {},
            "qualification_process_timeout_seconds": PROCESS_FUSE_SECONDS,
        }
    )
    return expected


def _expected_episode_run(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    item: Mapping[str, Any],
    runtime_path: Path,
) -> dict[str, Any]:
    group = _group(config, str(item["group_id"]))
    task_id = str(item["task_id"])
    seed = int(item["solver_seed"])
    job_key = {(task_id, seed)}
    return run_closed_loop_collection(
        root / str(group["dataset"]),
        runtime_path,
        _key_root(output, item) / str(item["controller"]),
        phase="realized_dynamic",
        workers=1,
        dry_run=True,
        task_ids=[task_id],
        cohort_job_keys=job_key,
        job_keys=job_key,
        qualification_source=_key_root(output, item) / "qualification",
        qualification_process_timeout_seconds=PROCESS_FUSE_SECONDS,
        use_global_collection_lock=False,
        **controller_kwargs(root, config, str(item["controller"])),
    )


def _validate_run_config_identity(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    item: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    runtime_path: Path,
) -> str:
    group = _group(config, str(item["group_id"]))
    expected_configuration = _expected_effective_configuration(
        root, config, item, runtime_path
    )
    observed_configuration = dict(payload.get("configuration") or {})
    configuration_fingerprint = json_fingerprint(observed_configuration)
    expected = _expected_episode_run(
        root, output, config, item, runtime_path
    )
    recomputed_run_fingerprint = json_fingerprint(
        {
            "dataset_fingerprint": payload.get("dataset_fingerprint"),
            "configuration_fingerprint": configuration_fingerprint,
            "freeze_manifest": payload.get("frozen_models"),
            "controller_bundle_manifest": payload.get("controller_bundle"),
            "diagnostic_shadow_bundle_manifests": payload.get(
                "diagnostic_shadow_bundles"
            ),
            "v3_s3_bundle_manifest": payload.get("v3_s3_bundle"),
            "controller_implementation": payload.get(
                "controller_implementation"
            ),
        }
    )
    run_fingerprint = str(payload.get("run_fingerprint", ""))
    if (
        payload.get("schema") != CLOSED_LOOP_SCHEMA
        or payload.get("formal") is not False
        or Path(str(payload.get("dataset", ""))).resolve()
        != (root / str(group["dataset"])).resolve()
        or observed_configuration != expected_configuration
        or str(payload.get("configuration_fingerprint", ""))
        != configuration_fingerprint
        or run_fingerprint != recomputed_run_fingerprint
        or run_fingerprint != str(expected.get("run_fingerprint", ""))
        or str(payload.get("trace_format")) != str(expected.get("trace_format"))
        or str(payload.get("storage_fingerprint"))
        != str(expected.get("storage_fingerprint"))
        or str(payload.get("controller")) != "v2-full"
        or str(payload.get("feature_backend")) != "native"
        or str(payload.get("controller_runtime")) != "optimized"
        or str(payload.get("verification_profile")) != "deployment"
    ):
        raise ValueError(
            f"cross-map profile inner run identity changed: {item['key_id']} "
            f"{item['controller']}"
        )
    return run_fingerprint


def _audit_episode(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    item: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    group = _group(config, str(item["group_id"]))
    task_id = str(item["task_id"])
    seed = int(item["solver_seed"])
    controller = str(item["controller"])
    collection = _key_root(output, item) / controller
    trace_value = row.get("trace_file")
    if not isinstance(trace_value, str) or not trace_value:
        raise ValueError("manifest trace path is missing")
    trace_path = (collection / trace_value).resolve()
    try:
        trace_path.relative_to(collection.resolve())
    except ValueError as error:
        raise ValueError("manifest trace path escapes its collection") from error
    if not trace_path.is_file():
        raise ValueError("manifest trace file is missing")

    run_path = collection / "run_config.json"
    if not run_path.is_file():
        raise ValueError("inner collection run_config.json is missing")
    runtime_path = _audit_runtime_config_path(output, group, seed)
    run_config = read_json(run_path)
    run_fingerprint = _validate_run_config_identity(
        root,
        output,
        config,
        item,
        run_config,
        runtime_path=runtime_path,
    )
    dataset_rows = _dataset_tasks(
        root / str(group["dataset"]), str(group["split"])
    )
    dataset_row = dict(dataset_rows[task_id])
    expected_episode = closed_loop_episode_id(
        dataset_row, seed, "realized_dynamic"
    )
    if (
        row.get("schema") != CLOSED_LOOP_SCHEMA
        or str(row.get("episode_id", "")) != expected_episode
        or str(row.get("task_id", "")) != task_id
        or int(row.get("solver_seed", -1)) != seed
        or str(row.get("policy", "")) != "realized_dynamic"
        or str(row.get("status", "")) not in {"ok", "resumed"}
        or row.get("error") is not None
        or str(row.get("split", "")) != str(group["split"])
        or str(row.get("map_id", "")) != str(group["map_id"])
        or str(row.get("map_id", "")) != str(dataset_row["map_id"])
        or int(row.get("agent_count", -1)) != int(dataset_row["agent_count"])
        or str(row.get("trace_format", ""))
        != str(run_config.get("trace_format", ""))
        or str(row.get("storage_fingerprint", ""))
        != str(run_config.get("storage_fingerprint", ""))
    ):
        raise ValueError("manifest policy/task/seed/status identity changed")

    metadata = trace_file_metadata(trace_path)
    if (
        str(row.get("trace_sha256", "")) != str(metadata["trace_sha256"])
        or int(row.get("trace_bytes", -1)) != int(metadata["trace_bytes"])
    ):
        raise ValueError("manifest trace hash or size changed")
    metric_budget = dict(run_config["configuration"]).get(
        "metric_iteration_budget"
    )
    validated = validate_closed_loop_trace(
        trace_path,
        run_fingerprint,
        expected_episode_id=expected_episode,
        expected_policy="realized_dynamic",
        expected_solver_seed=seed,
        metric_iteration_budget=(
            int(metric_budget) if metric_budget is not None else None
        ),
        collection_root=collection,
    )
    if validated.get("summary") != row.get("summary"):
        raise ValueError("manifest and trace summaries differ")
    if int(validated.get("event_count", -1)) != int(
        row.get("trace_event_count", -2)
    ):
        raise ValueError("manifest and trace event counts differ")
    if validated.get("initial_state_ref") != row.get("initial_state_ref"):
        raise ValueError("manifest and trace initial state references differ")
    events = list(validated.get("events") or ())
    if len(events) < 2:
        raise ValueError("validated trace has no initial/final boundary")
    summary = dict(validated["summary"])
    initial_fingerprint = str(events[0].get("state_fingerprint", ""))
    final_fingerprint = str(events[-1].get("final_fingerprint", ""))
    if (
        not initial_fingerprint
        or str(summary.get("initial_fingerprint", "")) != initial_fingerprint
        or not final_fingerprint
        or events[-1].get("summary") != summary
    ):
        raise ValueError("validated trace initial/final identity differs from summary")
    return {
        "key_id": str(item["key_id"]),
        "controller": controller,
        "episode_id": expected_episode,
        "run_fingerprint": run_fingerprint,
        "trace_file": trace_value,
        "trace_sha256": str(metadata["trace_sha256"]),
        "trace_bytes": int(metadata["trace_bytes"]),
        "trace_event_count": int(validated["event_count"]),
        "initial_fingerprint": initial_fingerprint,
        "final_fingerprint": final_fingerprint,
    }


def evaluate_key(
    expected_winner: str,
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if expected_winner not in {"component16", "hotspot16"}:
        raise ValueError(f"unsupported expected winner: {expected_winner}")
    if set(summaries) != set(CONTROLLERS):
        raise ValueError("falsification key requires all three controllers")
    restricted_ttf = {
        controller: float(summary["capped_wall_time_to_feasible"])
        for controller, summary in summaries.items()
    }
    success = {
        controller: bool(summary["success"])
        for controller, summary in summaries.items()
    }
    competitors = [
        controller for controller in CONTROLLERS if controller != expected_winner
    ]
    gates = {
        "expected_winner_strictly_fastest": all(
            restricted_ttf[expected_winner] < restricted_ttf[controller]
            for controller in competitors
        ),
        "expected_winner_success_noninferior": all(
            int(success[expected_winner]) >= int(success[controller])
            for controller in competitors
        ),
    }
    return {
        "expected_winner": expected_winner,
        "restricted_ttf": restricted_ttf,
        "success": success,
        "gates": gates,
        "passed": all(gates.values()),
        "tie_is_failure": True,
    }


def analyze(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output_path = Path(output).resolve()
    if producer is None and (output_path / STATUS_FILENAME).is_file():
        existing_status = read_json(output_path / STATUS_FILENAME)
        if isinstance(existing_status, dict) and isinstance(
            existing_status.get("producer_identity"), dict
        ):
            producer = dict(existing_status["producer_identity"])
    errors: list[str] = []
    freshness: dict[str, Any] | None = None
    freshness_path = output_path / FRESHNESS_FILENAME
    try:
        if not freshness_path.is_file():
            raise ValueError("frozen freshness audit is missing")
        freshness = _validate_freshness_audit(
            read_json(freshness_path),
            root=root,
            output=output_path,
            config=config,
            config_sha256=sha256_file(path),
        )
        if not bool(freshness["passed"]):
            raise ValueError("frozen freshness audit did not pass")
        status_path = output_path / STATUS_FILENAME
        if status_path.is_file():
            status = read_json(status_path)
            expected_sha = status.get("freshness_audit_sha256")
            if expected_sha is not None and expected_sha != sha256_file(freshness_path):
                raise ValueError("frozen freshness audit file hash changed")
    except (OSError, TypeError, ValueError) as error:
        errors.append(f"freshness integrity: {type(error).__name__}: {error}")

    rows = schedule(config)
    try:
        prefix = _manifest_prefix_length(output_path, rows)
    except ValueError as error:
        errors.append(str(error))
        prefix = sum(_manifest_row(output_path, item) is not None for item in rows)
    if prefix % len(CONTROLLERS):
        errors.append("executed schedule ends inside a paired three-controller key")
    evaluated_key_count = prefix // len(CONTROLLERS)

    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    trace_integrity: list[dict[str, Any]] = []
    for item in rows[:prefix]:
        row = _manifest_row(output_path, item)
        key = (str(item["key_id"]), str(item["controller"]))
        if row is None:
            errors.append(f"missing manifest inside executed prefix: {key}")
            continue
        try:
            trace_integrity.append(
                _audit_episode(root, output_path, config, item, row)
            )
        except (OSError, KeyError, TypeError, ValueError) as error:
            errors.append(
                f"episode integrity {key}: {type(error).__name__}: {error}"
            )
            continue
        if not isinstance(row.get("summary"), dict):
            errors.append(f"invalid episode summary: {key}")
            continue
        indexed[key] = row

    allowed_stops = {"success", "wall_timeout", "controller_stalled", "native_terminal"}
    for key, row in indexed.items():
        summary = dict(row["summary"])
        if (
            float(summary.get("wall_time_budget_seconds", -1.0))
            != WALL_TIME_SECONDS
            or str(summary.get("stop_reason")) not in allowed_stops
            or int(summary.get("invalid_action_count", -1)) != 0
            or int(summary.get("fingerprint_mismatch_count", -1)) != 0
            or summary.get("capped_wall_time_to_feasible") is None
        ):
            errors.append(f"invalid bounded episode: {key}")

    key_results: list[dict[str, Any]] = []
    keys = key_rows(config)
    for key in keys[:evaluated_key_count]:
        key_id = str(key["key_id"])
        summaries = {
            controller: dict(indexed[(key_id, controller)]["summary"])
            for controller in CONTROLLERS
            if (key_id, controller) in indexed
        }
        if len(summaries) != len(CONTROLLERS):
            errors.append(f"paired key lacks all controllers: {key_id}")
            continue
        fingerprints = {str(summary["initial_fingerprint"]) for summary in summaries.values()}
        conflicts = {int(summary["initial_conflicts"]) for summary in summaries.values()}
        if len(fingerprints) != 1 or len(conflicts) != 1:
            errors.append(f"paired reset mismatch: {key_id}")

        qualification_path = (
            _key_root(output_path, key)
            / "qualification"
            / "qualification_manifest.jsonl"
        )
        qualification_matches = [
            row
            for row in (
                read_jsonl(qualification_path) if qualification_path.is_file() else []
            )
            if str(row.get("task_id")) == str(key["task_id"])
            and int(row.get("solver_seed", -1)) == int(key["solver_seed"])
        ]
        if len(qualification_matches) != 1:
            errors.append(f"missing or ambiguous qualification anchor: {key_id}")
        else:
            qualification = qualification_matches[0]
            if (
                str(qualification.get("task_id", "")) != str(key["task_id"])
                or int(qualification.get("solver_seed", -1))
                != int(key["solver_seed"])
                or str(qualification.get("status", "")) not in {"ok", "resumed"}
                or qualification.get("error") is not None
                or any(
                    str(summary["initial_fingerprint"])
                    != str(qualification.get("state_fingerprint"))
                    or int(summary["initial_conflicts"])
                    != int(qualification.get("initial_conflicts", -1))
                    for summary in summaries.values()
                )
            ):
                errors.append(f"qualification anchor mismatch: {key_id}")

        result = evaluate_key(str(key["expected_winner"]), summaries)
        result.update(
            {
                "key_index": int(key["key_index"]),
                "key_id": key_id,
                "group_id": str(key["group_id"]),
                "map_id": str(key["map_id"]),
                "solver_seed": int(key["solver_seed"]),
                "initial_fingerprint": next(iter(fingerprints)),
                "initial_conflicts": next(iter(conflicts)),
            }
        )
        key_results.append(result)

    failed = next((row for row in key_results if not bool(row["passed"])), None)
    if failed is not None and prefix > (int(failed["key_index"]) + 1) * len(CONTROLLERS):
        errors.append("episodes were executed after the first failed key")

    if errors:
        decision = "invalid_or_incomplete_profile_falsification"
    elif failed is not None:
        decision = "stop_profile_hypothesis_falsified"
    elif evaluated_key_count == len(keys):
        decision = "profile_hypothesis_survived_minimal_falsification"
    else:
        decision = "continue_pending_falsification_keys"

    controller_summaries: dict[str, dict[str, Any]] = {}
    for controller in CONTROLLERS:
        selected = [
            dict(indexed[(str(key["key_id"]), controller)]["summary"])
            for key in keys[:evaluated_key_count]
            if (str(key["key_id"]), controller) in indexed
        ]
        if not selected:
            continue
        controller_summaries[controller] = {
            "episode_count": len(selected),
            "success_count": sum(bool(row["success"]) for row in selected),
            "mean_restricted_ttf": statistics.fmean(
                float(row["capped_wall_time_to_feasible"]) for row in selected
            ),
        }

    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "minimal_independent_falsification_only",
        "integrity_passed": not errors,
        "errors": errors,
        "decision": decision,
        "scheduled_paired_key_count": 4,
        "evaluated_paired_key_count": evaluated_key_count,
        "scheduled_episode_count": 12,
        "executed_episode_count": prefix,
        "cancelled_episode_count": (
            len(rows) - prefix if decision == "stop_profile_hypothesis_falsified" else 0
        ),
        "wall_time_budget_seconds": WALL_TIME_SECONDS,
        "freshness_audit": (
            {
                "path": FRESHNESS_FILENAME,
                "sha256": sha256_file(freshness_path),
                "inventory_sha256": str(freshness["inventory_sha256"]),
                "inventory_file_count": int(freshness["inventory_file_count"]),
                "match_count": int(freshness["match_count"]),
                "registration_time_full_workspace_audit": dict(
                    freshness["registration_time_full_workspace_audit"]
                ),
            }
            if freshness is not None and freshness_path.is_file()
            else None
        ),
        "trace_integrity": trace_integrity,
        "key_results": key_results,
        "first_failed_key": dict(failed) if failed is not None else None,
        "controller_summaries": controller_summaries,
        "all_four_key_gates_passed": (
            evaluated_key_count == 4
            and len(key_results) == 4
            and all(bool(row["passed"]) for row in key_results)
        ),
        "threshold_tuning_allowed": False,
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "larger_test_allowed": decision
        == "profile_hypothesis_survived_minimal_falsification",
        "omitted": list(config["explicitly_omitted"]),
        "inputs": {
            "config_sha256": sha256_file(path),
            "trace_sha256_by_episode": {
                str(row["episode_id"]): str(row["trace_sha256"])
                for row in trace_integrity
            },
        },
        "producer_identity": dict(producer) if producer is not None else None,
    }
    write_json(output_path / REPORT_FILENAME, report)
    return report


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        return plan(config_path)
    path, root, config = load_config(config_path)
    rows = schedule(config)
    keys = key_rows(config)
    output_path = Path(output).resolve()
    producer = closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/crossmap_profile_falsification_v1.py",
            "experiments/closed_loop_confirmation.py",
            "lns2_selector/runtime/structshell_single_family.py",
            "lns2_selector/runtime/hybridstructpool.py",
            "lns2_selector/runtime/topology_candidates.py",
        ),
    )
    prepared = prepare_resumable_output(
        output_path,
        status_filename=STATUS_FILENAME,
        status_schema=STATUS_SCHEMA,
        config_path=path,
        schedule=rows,
        producer=producer,
        resume=resume,
        report_filename=REPORT_FILENAME,
        report_schema=REPORT_SCHEMA,
        label="cross-map profile falsification",
    )
    try:
        freshness = _prepare_freshness_audit(
            root,
            output_path,
            config,
            config_sha256=sha256_file(path),
            resumed=prepared.resumed,
            status=prepared.status,
        )
        base_status = _freshness_status_base(
            prepared.base_status, output_path, freshness
        )
        if prepared.resumed:
            for field in (
                "freshness_audit_sha256",
                "freshness_inventory_sha256",
                "freshness_match_count",
            ):
                if prepared.status.get(field) != base_status[field]:
                    raise ValueError(f"falsification resume {field} mismatch")
        else:
            write_json(
                output_path / STATUS_FILENAME,
                {**dict(prepared.status), **base_status},
            )
    except Exception as error:
        status = {
            **dict(prepared.base_status),
            "completed_schedule_entries": 0,
            "executed_schedule_entries": 0,
            "cancelled_schedule_entries": 0,
            "complete": False,
            "terminal_failure": {
                "phase": "freshness_audit",
                "error": f"{type(error).__name__}: {error}",
            },
        }
        write_json(output_path / STATUS_FILENAME, status)
        return status
    if not bool(freshness["passed"]):
        status = {
            **base_status,
            "completed_schedule_entries": 0,
            "executed_schedule_entries": 0,
            "cancelled_schedule_entries": 0,
            "complete": False,
            "terminal_failure": {
                "phase": "freshness_audit",
                "error": "exact task/seed artifact inventory is not empty",
                "match_count": int(freshness["match_count"]),
                "scan_error_count": len(freshness["scan_errors"]),
            },
        }
        write_json(output_path / STATUS_FILENAME, status)
        return status
    if prepared.completed_report is not None:
        return prepared.completed_report

    try:
        prefix = _manifest_prefix_length(output_path, rows)
    except Exception as error:
        status = _progress_status(
            output_path,
            rows,
            base_status,
            terminal_failure={
                "phase": "resume_integrity",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        write_json(output_path / STATUS_FILENAME, status)
        return status
    recorded = int(prepared.status.get("completed_schedule_entries", -1))
    if recorded < 0 or recorded > prefix:
        raise ValueError("falsification status is ahead of the manifest prefix")
    if recorded < prefix:
        # A worker may finish and atomically publish its manifest immediately
        # before the outer progress write.  Exact-identity resume can safely
        # reconcile that one-way crash window from the manifest prefix.
        write_json(
            output_path / STATUS_FILENAME,
            _progress_status(output_path, rows, base_status),
        )

    for key in keys:
        block = rows[
            int(key["key_index"]) * len(CONTROLLERS) :
            (int(key["key_index"]) + 1) * len(CONTROLLERS)
        ]
        if all(_manifest_row(output_path, item) is not None for item in block):
            report = analyze(path, output_path, producer=producer)
            if report["decision"] == "stop_profile_hypothesis_falsified":
                status = _completed_status(
                    output_path, rows, base_status, report
                )
                write_json(output_path / STATUS_FILENAME, status)
                return report
            if report["decision"] == "invalid_or_incomplete_profile_falsification":
                status = _progress_status(
                    output_path,
                    rows,
                    base_status,
                    terminal_failure={
                        "phase": "analysis",
                        "completed_block": str(key["key_id"]),
                        "terminal": True,
                        "errors": list(report["errors"]),
                    },
                )
                write_json(output_path / STATUS_FILENAME, status)
                return status
            continue

        try:
            qualification = _qualify_key(
                root,
                output_path,
                config,
                key,
                resume=prepared.resumed,
            )
        except Exception as error:
            status = _progress_status(
                output_path,
                rows,
                base_status,
                terminal_failure={
                    "phase": "qualification",
                    "key": dict(key),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            write_json(output_path / STATUS_FILENAME, status)
            return status

        for item in block:
            if _manifest_row(output_path, item) is not None:
                continue
            try:
                manifest = _run_episode(
                    root,
                    output_path,
                    config,
                    item,
                    qualification,
                )
            except Exception as error:
                status = _progress_status(
                    output_path,
                    rows,
                    base_status,
                    terminal_failure={
                        "phase": "timed_episode",
                        "item": dict(item),
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
                write_json(output_path / STATUS_FILENAME, status)
                return status
            if manifest.get("status") in {"error", "timeout"}:
                status = _progress_status(
                    output_path,
                    rows,
                    base_status,
                    terminal_failure={
                        "phase": "timed_episode",
                        "item": dict(item),
                        "error": str(manifest.get("error") or manifest.get("status")),
                    },
                )
                write_json(output_path / STATUS_FILENAME, status)
                return status
            write_json(
                output_path / STATUS_FILENAME,
                _progress_status(output_path, rows, base_status),
            )

        report = analyze(path, output_path, producer=producer)
        if report["decision"] == "stop_profile_hypothesis_falsified":
            status = _completed_status(output_path, rows, base_status, report)
            write_json(output_path / STATUS_FILENAME, status)
            return report
        if report["decision"] == "invalid_or_incomplete_profile_falsification":
            status = _progress_status(
                output_path,
                rows,
                base_status,
                terminal_failure={
                    "phase": "analysis",
                    "completed_block": str(key["key_id"]),
                    "terminal": True,
                    "errors": list(report["errors"]),
                },
            )
            write_json(output_path / STATUS_FILENAME, status)
            return status

    report = analyze(path, output_path, producer=producer)
    if report["decision"] == "invalid_or_incomplete_profile_falsification":
        status = _progress_status(
            output_path,
            rows,
            base_status,
            terminal_failure={
                "phase": "analysis",
                "completed_block": str(keys[-1]["key_id"]),
                "terminal": True,
                "errors": list(report["errors"]),
            },
        )
        write_json(output_path / STATUS_FILENAME, status)
        return status
    status = _completed_status(output_path, rows, base_status, report)
    write_json(output_path / STATUS_FILENAME, status)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "CONTROLLERS",
    "EXPERIMENT_ID",
    "PROCESS_FUSE_SECONDS",
    "PROFILES",
    "REPORT_SCHEMA",
    "SOLVER_SEEDS",
    "STATUS_SCHEMA",
    "WALL_TIME_SECONDS",
    "analyze",
    "controller_kwargs",
    "evaluate_key",
    "key_rows",
    "load_config",
    "plan",
    "run",
    "schedule",
]
