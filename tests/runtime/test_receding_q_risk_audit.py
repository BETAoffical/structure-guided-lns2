from __future__ import annotations

from pathlib import Path

import pytest

from experiments import receding_q_risk_audit as module
from experiments import receding_q_pilot as pilot
from experiments import receding_q_stability as stability_module


def _row(
    candidate: str,
    trial: int,
    auc: float,
    *,
    map_id: str = "map",
    v2: bool = False,
) -> dict:
    return {
        "state_id": f"state-{map_id}",
        "candidate_id": candidate,
        "trial_index": trial,
        "feasible": True,
        "final_conflict_ratio": auc,
        "normalized_step_auc": auc,
        "normalized_wall_auc_seconds": auc,
        "observed_total_seconds": auc,
        "is_v2_candidate": v2,
        "map_id": map_id,
        "layout_mode": "layout",
        "agent_count": 600,
    }


def test_standard_error_penalty_prefers_stable_candidate() -> None:
    rows = [
        _row("unstable", 0, 0.0),
        _row("unstable", 1, 0.3),
        _row("unstable", 2, 0.0),
        _row("stable", 0, 0.12),
        _row("stable", 1, 0.12),
        _row("stable", 2, 0.12),
    ]
    mean = module.select_risk_candidate(
        rows,
        policy={
            "policy_id": "mean",
            "risk_mode": "mean",
            "risk_lambda": 0.0,
        },
    )
    conservative = module.select_risk_candidate(
        rows,
        policy={
            "policy_id": "risk",
            "risk_mode": "quality",
            "risk_lambda": 1.0,
        },
    )
    assert mean["candidate_id"] == "unstable"
    assert conservative["candidate_id"] == "stable"


def test_risk_loo_uses_only_other_three_seeds() -> None:
    rows = []
    for trial in range(4):
        rows.append(_row("a", trial, 0.10, v2=True))
        rows.append(_row("b", trial, 0.20))
    policies = [
        {
            "policy_id": "mean",
            "risk_mode": "mean",
            "risk_lambda": 0.0,
            "complexity": 0,
        }
    ]
    result = module.build_risk_loo_rows(
        rows,
        target_state_ids=["state-map"],
        policies=policies,
    )
    assert len(result) == 4
    assert {row["selected_candidate_id"] for row in result} == {"a"}
    assert all(row["selected_vs_v2_outcome"] == 0 for row in result)


def test_map_group_selection_never_uses_heldout_map_for_policy_choice() -> None:
    rows = []
    for map_id, preferred in (("map-a", "mean"), ("map-b", "risk")):
        for policy_id in ("mean", "risk"):
            for trial in range(2):
                win = policy_id == preferred
                rows.append(
                    {
                        "policy_id": policy_id,
                        "risk_mode": policy_id,
                        "risk_lambda": 0.0 if policy_id == "mean" else 1.0,
                        "policy_complexity": 0 if policy_id == "mean" else 1,
                        "state_id": f"state-{map_id}",
                        "map_id": map_id,
                        "agent_count": 600,
                        "selected_vs_v2_outcome": 1 if win else -1,
                        "selected_feasible": True,
                        "v2_feasible": True,
                        "selected_minus_v2_final_conflict_ratio": (
                            -0.1 if win else 0.1
                        ),
                        "selected_minus_v2_normalized_step_auc": (
                            -0.1 if win else 0.1
                        ),
                        "selected_minus_v2_total_seconds": (
                            -0.1 if win else 0.1
                        ),
                        "oracle_normalized_step_auc_regret": 0.0,
                    }
                )
    selected, selections = module.map_group_policy_selection(rows)
    by_map = {
        row["heldout_map_id"]: row["selected_policy_id"]
        for row in selections
    }
    assert by_map["map-a"] == "risk"
    assert by_map["map-b"] == "mean"
    assert len(selected) == 4


def test_persisted_wsl_source_falls_back_to_sibling(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-pilot"
    source.mkdir()
    resolved = module.resolve_persisted_source_path(
        "/mnt/c/old/location/source-pilot",
        sibling_root=tmp_path,
    )
    assert resolved == source


def test_risk_audit_rejects_corrupt_stability_configuration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-pilot"
    stability = tmp_path / "stability"
    source.mkdir()
    stability.mkdir()
    module._write_json(
        source / "status.json",
        {
            "schema": pilot.RECEDING_Q_PILOT_SCHEMA,
            "status": "complete",
            "error_count": 0,
        },
    )
    module._write_json(
        stability / "status.json",
        {
            "schema": stability_module.RECEDING_Q_STABILITY_SCHEMA,
            "status": "complete",
            "error_count": 0,
        },
    )
    module._write_json(
        stability / "run_config.json",
        {
            "schema": "corrupt",
            "source": str(source),
        },
    )
    with pytest.raises(ValueError, match="configuration schema mismatch"):
        module.audit_receding_q_risk(
            stability=stability,
            output=tmp_path / "output",
        )
