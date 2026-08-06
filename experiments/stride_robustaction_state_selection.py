from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_collection import (
    STRIDE_SELECTION_SCHEMA,
    _agent_band,
    _conflict_band,
    _decision_stage,
)
from experiments.trace_replay import result_blind_decision_rows


CONFIG_SCHEMA = (
    "lns2.stride.robustaction_structpool_combined_state_selection_design.v1"
)
CONFIG_SCHEMA_V2 = (
    "lns2.stride.robustaction_structpool_combined_state_selection_design.v2"
)
REPORT_SCHEMA = "lns2.stride.robustaction_combined_state_selection_report.v1"
REPORT_SCHEMA_V2 = "lns2.stride.robustaction_combined_state_selection_report.v2"
SOURCE_POLICIES = ("official_adaptive", "realized_dynamic")
SELECTION_POLICIES = ("official_adaptive", "v2-full")


def _registered(project_root: Path, spec: dict[str, Any]) -> Path:
    path = (project_root / str(spec["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"registered selection input is missing: {path}")
    observed = sha256_file(path)
    expected = str(spec["sha256"])
    if observed != expected:
        raise ValueError(
            f"registered selection input hash differs: {path}: "
            f"expected {expected}, got {observed}"
        )
    return path


def _source_path(project_root: Path, source: dict[str, Any]) -> Path:
    root = (project_root / str(source["root"])).resolve()
    if not root.is_dir():
        raise ValueError(f"registered source root is missing: {root}")
    return root


def validate_robustaction_state_selection_design(
    config: dict[str, Any], *, project_root: Path | None = None
) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("robust-action state-selection schema changed")
    if config.get("selection_id") != (
        "stride-robustaction-structpool-combined-selection-v1"
    ):
        raise ValueError("robust-action state-selection identity changed")
    if config.get("planned_model_id") != "stride-robustaction-v1":
        raise ValueError("robust-action model identity changed")

    source_contract = dict(config.get("source_contract") or {})
    if (
        source_contract.get("registered_split") != "balanced_wall_clock"
        or tuple(source_contract.get("source_manifest_policies") or ())
        != SOURCE_POLICIES
        or tuple(source_contract.get("selection_policies") or ())
        != SELECTION_POLICIES
        or source_contract.get("stopping_rule") != "historical"
        or int(source_contract.get("max_decisions", -1)) != 12
        or int(source_contract.get("max_repair_iterations", -1)) != 12
        or int(source_contract.get("metric_iteration_budget", -1)) != 12
        or source_contract.get("deterministic_pp_replay") is not True
        or int(source_contract.get("expected_total_episode_rows", -1)) != 224
        or int(source_contract.get("expected_raw_positive_state_count", -1))
        != 1559
        or int(source_contract.get("expected_capped_state_capacity", -1)) != 415
        or int(source_contract.get("expected_eligible_episode_count", -1)) != 216
    ):
        raise ValueError("robust-action state-selection source contract changed")

    selection = dict(config.get("selection_contract") or {})
    allowed = (
        "source_namespace",
        "source_policy",
        "episode_id",
        "decision_index",
        "before_fingerprint",
        "before_conflicts",
    )
    if (
        int(selection.get("target_state_count", -1)) != 320
        or int(selection.get("target_states_per_policy", -1)) != 160
        or int(selection.get("maximum_states_per_episode", -1)) != 2
        or selection.get("require_every_eligible_episode") is not True
        or int(selection.get("expected_selected_episode_count", -1)) != 216
        or tuple(selection.get("state_rank_fields") or ()) != allowed
        or tuple(selection.get("permitted_selection_inputs") or ()) != allowed
        or tuple(selection.get("episode_rank_fields") or ())
        != allowed[:3]
        or selection.get("failure_action")
        != "preserve_complete_product_and_stop_before_candidate_labels_without_filtering"
    ):
        raise ValueError("robust-action state-selection sampling contract changed")

    boundary = dict(config.get("claim_boundary") or {})
    if (
        boundary.get("candidate_repair_outcomes_read") is not False
        or boundary.get("target_decision_outcomes_read") is not False
        or boundary.get("ttf_read") is not False
        or boundary.get("source_episode_outcomes_used_to_filter") is not False
        or boundary.get("formal_speed_claim") is not False
    ):
        raise ValueError("robust-action state-selection outcome boundary changed")

    sources = list(dict(config.get("inputs") or {}).get("sources") or [])
    if [str(row.get("namespace")) for row in sources] != [
        "source-v4",
        "da2-stability-v2",
    ]:
        raise ValueError("robust-action state-selection source namespaces changed")
    for source in sources:
        policies = dict(source.get("policy_manifests") or {})
        if tuple(policies) != SOURCE_POLICIES:
            raise ValueError("robust-action source policy manifests changed")
        if [str(policies[name].get("selection_policy")) for name in SOURCE_POLICIES] != list(
            SELECTION_POLICIES
        ):
            raise ValueError("robust-action source-policy mapping changed")

    if project_root is None:
        return
    project_root = project_root.resolve()
    inputs = dict(config["inputs"])
    _registered(project_root, dict(inputs["capacity_report"]))
    for source in sources:
        _source_path(project_root, source)
        _registered(project_root, dict(source["run_config"]))
        _registered(project_root, dict(source["dataset_manifest"]))
        _registered(project_root, dict(source["dataset_summary"]))
        for policy in SOURCE_POLICIES:
            _registered(
                project_root,
                dict(dict(source["policy_manifests"])[policy]),
            )


def validate_robustaction_state_selection_v2_design(
    config: dict[str, Any], *, project_root: Path | None = None
) -> None:
    if config.get("schema") != CONFIG_SCHEMA_V2:
        raise ValueError("robust-action state-selection-v2 schema changed")
    if (
        config.get("selection_id")
        != "stride-robustaction-structpool-combined-selection-v2"
        or config.get("planned_model_id") != "stride-robustaction-v1"
        or config.get("pre_registration_git_commit")
        != "16e4225e4e692ef135565e822721b98b9da032ca"
    ):
        raise ValueError("robust-action state-selection-v2 identity changed")

    observed = dict(config.get("observed_prelabel_feasibility") or {})
    if (
        int(observed.get("v1_selected_state_count", -1)) != 320
        or int(observed.get("v1_structpool_eligible_state_count", -1)) != 70
        or float(observed.get("v1_structpool_eligible_fraction", -1.0))
        != 0.21875
        or int(observed.get("required_state_count", -1)) != 80
        or observed.get("candidate_repair_outcomes_read") is not False
        or observed.get("ttf_read") is not False
    ):
        raise ValueError("robust-action state-selection-v2 diagnosis changed")

    source_contract = dict(config.get("source_contract") or {})
    if (
        source_contract.get("registered_split") != "balanced_wall_clock"
        or tuple(source_contract.get("source_manifest_policies") or ())
        != SOURCE_POLICIES
        or tuple(source_contract.get("selection_policies") or ())
        != SELECTION_POLICIES
        or source_contract.get("stopping_rule") != "historical"
        or int(source_contract.get("max_decisions", -1)) != 12
        or int(source_contract.get("max_repair_iterations", -1)) != 12
        or int(source_contract.get("metric_iteration_budget", -1)) != 12
        or source_contract.get("deterministic_pp_replay") is not True
        or int(source_contract.get("expected_total_episode_rows", -1)) != 224
        or int(source_contract.get("expected_raw_positive_state_count", -1))
        != 1559
        or int(source_contract.get("expected_capped_state_capacity", -1)) != 415
        or int(source_contract.get("expected_eligible_episode_count", -1)) != 216
    ):
        raise ValueError("robust-action state-selection-v2 source contract changed")

    selection = dict(config.get("selection_contract") or {})
    state_fields = (
        "source_namespace",
        "source_policy",
        "episode_id",
        "decision_index",
        "before_fingerprint",
        "before_conflicts",
        "agent_count",
    )
    if (
        int(selection.get("target_state_count", -1)) != 320
        or int(selection.get("target_states_per_policy", -1)) != 160
        or int(selection.get("maximum_states_per_episode", -1)) != 2
        or selection.get("require_every_eligible_episode") is not True
        or int(selection.get("expected_selected_episode_count", -1)) != 216
        or int(selection.get("minimum_structpool_eligible_states", -1)) != 80
        or int(selection.get("minimum_structpool_eligible_states_per_policy", -1))
        != 40
        or int(selection.get("structpool_minimum_conflicts", -1)) != 16
        or int(selection.get("structpool_minimum_agents", -1)) != 96
        or tuple(selection.get("permitted_selection_inputs") or ()) != state_fields
        or tuple(selection.get("episode_rank_fields") or ()) != state_fields[:3]
        or selection.get("failure_action")
        != "preserve_complete_product_and_stop_before_candidate_labels_without_filtering"
    ):
        raise ValueError("robust-action state-selection-v2 sampling contract changed")

    boundary = dict(config.get("claim_boundary") or {})
    if (
        boundary.get("candidate_repair_outcomes_read") is not False
        or boundary.get("target_decision_outcomes_used_for_ranking") is not False
        or boundary.get("ttf_read") is not False
        or boundary.get("source_episode_outcomes_used_to_filter") is not False
        or boundary.get("formal_speed_claim") is not False
    ):
        raise ValueError("robust-action state-selection-v2 outcome boundary changed")

    inputs = dict(config.get("inputs") or {})
    if inputs.get("reuse_predecessor_source_registry") is not True:
        raise ValueError("robust-action state-selection-v2 source registry changed")

    if project_root is None:
        return
    project_root = project_root.resolve()
    for name in (
        "capacity_report",
        "predecessor_design",
        "predecessor_selection",
        "predecessor_selection_report",
    ):
        _registered(project_root, dict(inputs[name]))
    predecessor = _read_json(
        _registered(project_root, dict(inputs["predecessor_design"]))
    )
    validate_robustaction_state_selection_design(
        predecessor, project_root=project_root
    )


def _state_rank(row: dict[str, Any]) -> int:
    identity = {
        "source_namespace": str(row["source_namespace"]),
        "source_policy": str(row["source_policy"]),
        "episode_id": str(row["episode_id"]),
        "decision_index": int(row["decision_index"]),
        "before_fingerprint": str(row["before_fingerprint"]),
        "before_conflicts": int(row["before_conflicts"]),
    }
    return int(_fingerprint(identity), 16)


def _episode_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["source_namespace"]),
        str(row["source_policy"]),
        str(row["episode_id"]),
    )


def _episode_rank(key: tuple[str, str, str]) -> int:
    namespace, policy, episode_id = key
    return int(
        _fingerprint(
            {
                "source_namespace": namespace,
                "source_policy": policy,
                "episode_id": episode_id,
            }
        ),
        16,
    )


def _select_episode_first_states(
    pool: list[dict[str, Any]],
    *,
    target_per_policy: int,
    prefer_structpool_eligible_primary: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Maximize independent episodes, then add one hash-ranked second state.

    Only the six preregistered current-state identity fields influence either
    ranking.  Metadata and the recorded source/controller outcomes are never
    accepted as ranking inputs.
    """

    grouped: dict[str, dict[tuple[str, str, str], list[dict[str, Any]]]] = {
        policy: defaultdict(list) for policy in SELECTION_POLICIES
    }
    for row in pool:
        policy = str(row["source_policy"])
        if policy not in grouped:
            raise ValueError(f"unexpected selection policy: {policy}")
        grouped[policy][_episode_key(row)].append(row)

    selected: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for policy in SELECTION_POLICIES:
        episodes = grouped[policy]
        if target_per_policy < len(episodes):
            raise ValueError(
                f"policy target would discard eligible episodes: {policy}"
            )
        hash_ordered_states = {
            key: sorted(rows, key=lambda row: (_state_rank(row), str(row["state_id"])))
            for key, rows in episodes.items()
        }
        primary_by_episode: dict[tuple[str, str, str], dict[str, Any]] = {}
        remaining_by_episode: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for key, rows in hash_ordered_states.items():
            preferred = (
                [
                    row
                    for row in rows
                    if int(row["before_conflicts"]) >= 16
                    and int(row["agent_count"]) >= 96
                ]
                if prefer_structpool_eligible_primary
                else []
            )
            primary = (
                preferred[0]
                if prefer_structpool_eligible_primary and preferred
                else rows[0]
            )
            primary_by_episode[key] = primary
            remaining_by_episode[key] = [row for row in rows if row is not primary]
        chosen = list(primary_by_episode.values())
        extra_required = target_per_policy - len(chosen)
        second_eligible = sorted(
            (key for key, rows in remaining_by_episode.items() if rows),
            key=lambda key: (_episode_rank(key), key),
        )
        if extra_required > len(second_eligible):
            raise ValueError(f"policy lacks capped state capacity: {policy}")
        second_keys = set(second_eligible[:extra_required])
        chosen.extend(remaining_by_episode[key][0] for key in second_keys)
        chosen.sort(key=lambda row: str(row["state_id"]))
        selected.extend(chosen)

        namespace_counts = Counter(str(row["source_namespace"]) for row in chosen)
        episode_counts = Counter(_episode_key(row) for row in chosen)
        policy_report = {
            "available_state_count": sum(len(rows) for rows in episodes.values()),
            "available_episode_count": len(episodes),
            "capped_state_capacity": sum(min(2, len(rows)) for rows in episodes.values()),
            "selected_state_count": len(chosen),
            "selected_episode_count": len(episode_counts),
            "single_state_episode_count": sum(value == 1 for value in episode_counts.values()),
            "two_state_episode_count": sum(value == 2 for value in episode_counts.values()),
            "maximum_states_per_episode": max(episode_counts.values(), default=0),
            "selected_source_namespace_counts": dict(sorted(namespace_counts.items())),
            "selected_decision_stage_counts": dict(
                sorted(Counter(str(row["decision_stage"]) for row in chosen).items())
            ),
            "selected_conflict_band_counts": dict(
                sorted(Counter(str(row["conflict_band"]) for row in chosen).items())
            ),
        }
        if prefer_structpool_eligible_primary:
            policy_report["selected_structpool_eligible_state_count"] = sum(
                int(row["before_conflicts"]) >= 16
                and int(row["agent_count"]) >= 96
                for row in chosen
            )
        reports[policy] = policy_report
    selected.sort(key=lambda row: (str(row["source_policy"]), str(row["state_id"])))
    return selected, reports


def select_episode_first_states(
    pool: list[dict[str, Any]], *, target_per_policy: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _select_episode_first_states(
        pool,
        target_per_policy=target_per_policy,
        prefer_structpool_eligible_primary=False,
    )


def select_episode_first_structpool_states(
    pool: list[dict[str, Any]], *, target_per_policy: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _select_episode_first_states(
        pool,
        target_per_policy=target_per_policy,
        prefer_structpool_eligible_primary=True,
    )


def _runtime_matches(run: dict[str, Any], contract: dict[str, Any]) -> bool:
    configuration = dict(run.get("configuration") or {})
    environment = dict(configuration.get("environment") or {})
    return (
        configuration.get("split") == contract["registered_split"]
        and configuration.get("stopping_rule") == contract["stopping_rule"]
        and int(configuration.get("max_decisions", -1)) == int(contract["max_decisions"])
        and int(environment.get("max_repair_iterations", -1))
        == int(contract["max_repair_iterations"])
        and int(configuration.get("metric_iteration_budget", -1))
        == int(contract["metric_iteration_budget"])
        and configuration.get("deterministic_pp_replay") is True
    )


def build_robustaction_combined_state_selection(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    output = Path(output).resolve()
    config = _read_json(config_path)
    is_v2 = config.get("schema") == CONFIG_SCHEMA_V2
    if is_v2:
        validate_robustaction_state_selection_v2_design(
            config, project_root=project_root
        )
    else:
        validate_robustaction_state_selection_design(
            config, project_root=project_root
        )

    inputs = dict(config["inputs"])
    if is_v2:
        predecessor_design = _read_json(
            _registered(project_root, dict(inputs["predecessor_design"]))
        )
        inputs["sources"] = list(
            dict(predecessor_design["inputs"])["sources"]
        )
    capacity_report = _read_json(_registered(project_root, dict(inputs["capacity_report"])))
    source_contract = dict(config["source_contract"])
    selection_contract = dict(config["selection_contract"])
    if not bool(capacity_report.get("passed")):
        raise ValueError("combined source capacity report did not pass")

    pool: list[dict[str, Any]] = []
    input_hashes: dict[str, Any] = {
        "capacity_report": sha256_file(
            _registered(project_root, dict(inputs["capacity_report"]))
        ),
        "sources": {},
    }
    if is_v2:
        input_hashes["predecessor"] = {
            name: sha256_file(_registered(project_root, dict(inputs[name])))
            for name in (
                "predecessor_design",
                "predecessor_selection",
                "predecessor_selection_report",
            )
        }
    source_reports: dict[str, Any] = {}
    episode_row_count = 0
    for source in list(inputs["sources"]):
        namespace = str(source["namespace"])
        source_root = _source_path(project_root, source)
        run_path = _registered(project_root, dict(source["run_config"]))
        dataset_path = _registered(project_root, dict(source["dataset_manifest"]))
        dataset_rows = _read_jsonl(dataset_path)
        dataset = {str(row["task_id"]): row for row in dataset_rows}
        if len(dataset) != len(dataset_rows):
            raise ValueError(f"source dataset contains duplicate task IDs: {namespace}")
        run = _read_json(run_path)
        if not _runtime_matches(run, source_contract):
            raise ValueError(f"source runtime differs from selection contract: {namespace}")

        source_hashes: dict[str, Any] = {
            "run_config": sha256_file(run_path),
            "dataset_manifest": sha256_file(dataset_path),
            "dataset_summary": sha256_file(
                _registered(project_root, dict(source["dataset_summary"]))
            ),
            "policy_manifests": {},
        }
        source_episode_counts: dict[str, int] = {}
        source_raw_start = len(pool)
        source_episode_ids: set[tuple[str, str]] = set()
        source_cap = 0
        for manifest_policy in SOURCE_POLICIES:
            spec = dict(dict(source["policy_manifests"])[manifest_policy])
            manifest_path = _registered(project_root, spec)
            source_hashes["policy_manifests"][manifest_policy] = sha256_file(
                manifest_path
            )
            manifests = _read_jsonl(manifest_path)
            expected_rows = int(spec["expected_episode_count"])
            if len(manifests) != expected_rows:
                raise ValueError(
                    f"source manifest row count differs: {namespace}/{manifest_policy}"
                )
            selection_policy = str(spec["selection_policy"])
            per_episode_counts: Counter[str] = Counter()
            for manifest in manifests:
                episode_row_count += 1
                if str(manifest.get("status")) != "ok" or manifest.get("error") is not None:
                    raise ValueError(
                        "complete source product contains a non-valid episode: "
                        f"{namespace}/{manifest_policy}/{manifest.get('episode_id')}"
                    )
                if str(manifest.get("policy")) != manifest_policy:
                    raise ValueError("source manifest policy identity differs")
                episode_id = str(manifest["episode_id"])
                episode_key = (selection_policy, episode_id)
                if episode_key in source_episode_ids:
                    raise ValueError("source contains a duplicate episode identity")
                source_episode_ids.add(episode_key)
                task_id = str(manifest["task_id"])
                dataset_row = dataset.get(task_id)
                if dataset_row is None:
                    raise ValueError(f"source task is absent from dataset: {task_id}")
                decisions, _ = result_blind_decision_rows(
                    source_root, {"trace_file": manifest["trace_file"]}
                )
                eligible = [
                    decision
                    for decision in decisions
                    if int(decision["before_conflicts"]) > 0
                    and 0
                    <= int(decision["decision_index"])
                    < int(source_contract["max_decisions"])
                ]
                indices = [int(row["decision_index"]) for row in eligible]
                if len(indices) != len(set(indices)):
                    raise ValueError("source episode contains duplicate decision indices")
                per_episode_counts[episode_id] = len(eligible)
                for decision in eligible:
                    identity = {
                        "source_namespace": namespace,
                        "source_policy": selection_policy,
                        "episode_id": episode_id,
                        "decision_index": int(decision["decision_index"]),
                        "before_fingerprint": str(decision["before_fingerprint"]),
                        "before_conflicts": int(decision["before_conflicts"]),
                    }
                    pool.append(
                        {
                            "schema": STRIDE_SELECTION_SCHEMA,
                            "state_id": "stride-robustaction-"
                            + _fingerprint(identity)[:24],
                            "source_namespace": namespace,
                            "map_id": str(manifest["map_id"]),
                            "task_id": task_id,
                            "split": str(manifest["split"]),
                            "source_policy": selection_policy,
                            "source_manifest_policy": manifest_policy,
                            "decision_stage": _decision_stage(
                                int(decision["decision_index"])
                            ),
                            "conflict_band": _conflict_band(
                                int(decision["before_conflicts"])
                            ),
                            "source_group": str(
                                dataset_row.get("source_group", "unknown")
                            ),
                            "layout_mode": str(
                                manifest.get(
                                    "layout_mode",
                                    dataset_row.get("layout_mode", "unknown"),
                                )
                            ),
                            "source_root": str(source_root),
                            "episode_id": episode_id,
                            "before_fingerprint": identity["before_fingerprint"],
                            "before_conflicts": identity["before_conflicts"],
                            "solver_seed": int(manifest["solver_seed"]),
                            "decision_index": identity["decision_index"],
                            "agent_count": int(manifest["agent_count"]),
                            "agent_band": _agent_band(int(manifest["agent_count"])),
                            "prefix_actions": list(decision["prefix_actions"]),
                        }
                    )
            source_episode_counts[selection_policy] = len(per_episode_counts)
            source_cap += sum(
                min(int(selection_contract["maximum_states_per_episode"]), value)
                for value in per_episode_counts.values()
            )
        source_raw = len(pool) - source_raw_start
        # Count eligible episodes from the materialized rows, not controller summaries.
        source_eligible = len(
            {
                _episode_key(row)
                for row in pool[source_raw_start:]
            }
        )
        if (
            source_raw != int(source["expected_raw_positive_state_count"])
            or source_cap != int(source["expected_capped_state_count"])
            or source_eligible != int(source["expected_eligible_episode_count"])
        ):
            raise ValueError(f"source capacity differs from registration: {namespace}")
        input_hashes["sources"][namespace] = source_hashes
        source_reports[namespace] = {
            "episode_row_count": sum(source_episode_counts.values()),
            "episode_rows_by_policy": source_episode_counts,
            "raw_positive_state_count": source_raw,
            "capped_state_capacity": source_cap,
            "eligible_episode_count": source_eligible,
        }

    state_ids = [str(row["state_id"]) for row in pool]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("combined source contains duplicate state identities")
    raw_count = len(pool)
    eligible_episodes = len({_episode_key(row) for row in pool})
    capped_capacity = sum(
        min(int(selection_contract["maximum_states_per_episode"]), count)
        for count in Counter(_episode_key(row) for row in pool).values()
    )
    if (
        episode_row_count != int(source_contract["expected_total_episode_rows"])
        or raw_count != int(source_contract["expected_raw_positive_state_count"])
        or capped_capacity != int(source_contract["expected_capped_state_capacity"])
        or eligible_episodes != int(source_contract["expected_eligible_episode_count"])
    ):
        raise ValueError("combined source capacity differs from registration")

    selector = (
        select_episode_first_structpool_states
        if is_v2
        else select_episode_first_states
    )
    selected, policy_reports = selector(
        pool, target_per_policy=int(selection_contract["target_states_per_policy"])
    )
    selected_episode_counts = Counter(_episode_key(row) for row in selected)
    selected_policy_counts = Counter(str(row["source_policy"]) for row in selected)
    structpool_eligible_by_policy = Counter(
        str(row["source_policy"])
        for row in selected
        if int(row["before_conflicts"]) >= 16 and int(row["agent_count"]) >= 96
    )
    gates = {
        "capacity_report_passed": bool(capacity_report.get("passed")),
        "source_episode_rows_exact": episode_row_count
        == int(source_contract["expected_total_episode_rows"]),
        "raw_state_identity_exact": raw_count
        == int(source_contract["expected_raw_positive_state_count"]),
        "capped_capacity_exact": capped_capacity
        == int(source_contract["expected_capped_state_capacity"]),
        "selection_target_exact": len(selected)
        == int(selection_contract["target_state_count"]),
        "policy_balance_exact": all(
            selected_policy_counts[policy]
            == int(selection_contract["target_states_per_policy"])
            for policy in SELECTION_POLICIES
        ),
        "all_eligible_episodes_covered": len(selected_episode_counts)
        == int(selection_contract["expected_selected_episode_count"])
        == eligible_episodes,
        "episode_cap": max(selected_episode_counts.values(), default=0)
        <= int(selection_contract["maximum_states_per_episode"]),
        "state_ids_unique": len({str(row["state_id"]) for row in selected})
        == len(selected),
    }
    if is_v2:
        gates["minimum_structpool_eligible_states"] = sum(
            structpool_eligible_by_policy.values()
        ) >= int(selection_contract["minimum_structpool_eligible_states"])
        gates["minimum_structpool_eligible_states_per_policy"] = all(
            structpool_eligible_by_policy[policy]
            >= int(selection_contract["minimum_structpool_eligible_states_per_policy"])
            for policy in SELECTION_POLICIES
        )
    passed = all(gates.values())
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "state_selection.jsonl"
    _write_jsonl(selection_path, selected)
    report = {
        "schema": REPORT_SCHEMA_V2 if is_v2 else REPORT_SCHEMA,
        "scientific_status": (
            "result_blind_structpool_eligible_training_state_selection"
            if is_v2
            else "result_blind_training_state_selection"
        ),
        "data_line_id": str(config["data_line_id"]),
        "selection_id": str(config["selection_id"]),
        "planned_model_id": str(config["planned_model_id"]),
        "config_sha256": sha256_file(config_path),
        "input_sha256": input_hashes,
        "source_reports": source_reports,
        "episode_row_count": episode_row_count,
        "raw_positive_state_count": raw_count,
        "capped_state_capacity": capped_capacity,
        "eligible_episode_count": eligible_episodes,
        "selected_state_count": len(selected),
        "selected_episode_count": len(selected_episode_counts),
        "selected_policy_counts": dict(sorted(selected_policy_counts.items())),
        "selected_source_namespace_counts": dict(
            sorted(Counter(str(row["source_namespace"]) for row in selected).items())
        ),
        "selected_map_count": len({str(row["map_id"]) for row in selected}),
        "selected_task_count": len({str(row["task_id"]) for row in selected}),
        "selected_episode_state_count_bands": {
            "one": sum(value == 1 for value in selected_episode_counts.values()),
            "two": sum(value == 2 for value in selected_episode_counts.values()),
        },
        "policies": policy_reports,
        "selection_rule": str(selection_contract["selection_rule"]),
        "candidate_repair_outcomes_read": False,
        "target_decision_outcomes_read": False,
        "ttf_read": False,
        "source_episode_outcomes_used_to_filter": False,
        "formal_speed_claim": False,
        "selection_sha256": sha256_file(selection_path),
        "gates": gates,
        "passed": passed,
        "next_decision": config[
            "next_decision_on_pass" if passed else "next_decision_on_failure"
        ],
    }
    if is_v2:
        report["selected_structpool_eligible_state_count"] = sum(
            structpool_eligible_by_policy.values()
        )
        report["selected_structpool_eligible_fraction"] = (
            sum(structpool_eligible_by_policy.values()) / len(selected)
            if selected
            else 0.0
        )
        report["selected_structpool_eligible_by_policy"] = dict(
            sorted(structpool_eligible_by_policy.items())
        )
    _write_json(output / "state_selection_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "CONFIG_SCHEMA_V2",
    "REPORT_SCHEMA",
    "REPORT_SCHEMA_V2",
    "build_robustaction_combined_state_selection",
    "select_episode_first_states",
    "select_episode_first_structpool_states",
    "validate_robustaction_state_selection_design",
    "validate_robustaction_state_selection_v2_design",
]
