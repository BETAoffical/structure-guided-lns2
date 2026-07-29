from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import atomic_write_csv, read_json, sha256_file, write_json
from experiments.run_output_guard import prepare_run_output
from experiments.stall_oracle import STALL_ORACLE_SCHEMA
from experiments.stall_risk_audit import STALL_RISK_AUDIT_SCHEMA


STALL_ESCAPE_ORACLE_COHORT_SCHEMA = "lns2.stall_escape_oracle_cohort.v1"
STALL_ESCAPE_ORACLE_COHORT_VERSION = 1


def _explicit_branches(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in report.get("branches", ())
        if str(row.get("branch_mode")) == "explicit_neighborhood"
    ]


def _load_pair(
    oracle_path: Path,
    risk_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    oracle = dict(read_json(oracle_path))
    risk_path = risk_root / "stall_risk_audit_report.json"
    runner_path = risk_root / "runner_config.json"
    risk = dict(read_json(risk_path))
    runner = dict(read_json(runner_path))
    if str(oracle.get("schema")) != STALL_ORACLE_SCHEMA:
        raise ValueError("stall cohort Oracle schema is unsupported")
    if str(risk.get("schema")) != STALL_RISK_AUDIT_SCHEMA:
        raise ValueError("stall cohort risk schema is unsupported")
    identity = runner.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("stall cohort risk runner identity is missing")
    if str(identity.get("oracle_report_sha256")) != sha256_file(oracle_path):
        raise ValueError("stall cohort risk audit does not bind its Oracle")
    if str(identity.get("v3_manifest_sha256")) != str(
        risk.get("v3_manifest_sha256")
    ):
        raise ValueError("stall cohort v3 identity differs across artifacts")
    if str(oracle.get("before_fingerprint")) != str(
        risk.get("before_fingerprint")
    ) or str(oracle.get("before_repair_fingerprint")) != str(
        risk.get("before_repair_fingerprint")
    ):
        raise ValueError("stall cohort state fingerprints differ")
    explicit = _explicit_branches(oracle)
    oracle_ids = {str(row.get("candidate_id") or "") for row in explicit}
    risk_ids = {
        str(row.get("candidate_id") or "") for row in risk.get("candidates", ())
    }
    if not all(oracle_ids) or oracle_ids != risk_ids:
        raise ValueError("stall cohort risk audit does not cover the Oracle pool")
    expected_trials = sum(int(row["trial_count"]) for row in explicit)
    if int(risk.get("candidate_count", -1)) != len(explicit) or int(
        risk.get("trial_count", -1)
    ) != expected_trials:
        raise ValueError("stall cohort candidate or trial coverage differs")
    return oracle, risk, {
        "oracle": str(oracle_path),
        "oracle_sha256": sha256_file(oracle_path),
        "risk": str(risk_path),
        "risk_sha256": sha256_file(risk_path),
        "risk_runner": str(runner_path),
        "risk_runner_sha256": sha256_file(runner_path),
    }


def build_stall_escape_oracle_cohort(
    oracle_paths: Iterable[str | Path],
    risk_roots: Iterable[str | Path],
    output: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    oracles = [Path(value).resolve() for value in oracle_paths]
    risks = [Path(value).resolve() for value in risk_roots]
    if not oracles or len(oracles) != len(risks):
        raise ValueError("stall cohort requires equal non-empty Oracle/risk sources")
    loaded = [_load_pair(oracle, risk) for oracle, risk in zip(oracles, risks)]
    manifests = [item[2] for item in loaded]
    v3_hashes = {str(item[1]["v3_manifest_sha256"]) for item in loaded}
    if len(v3_hashes) != 1:
        raise ValueError("stall cohort mixes different frozen v3 bundles")
    state_keys: set[tuple[str, int, int]] = set()
    rows: list[dict[str, Any]] = []
    for oracle, risk, source in loaded:
        key = (
            str(risk.get("task_id") or ""),
            int(risk.get("solver_seed", -1)),
            int(risk.get("decision_index", -1)),
        )
        if not key[0] or min(key[1:]) < 0 or key in state_keys:
            raise ValueError("stall cohort has invalid or duplicate state identity")
        state_keys.add(key)
        explicit = _explicit_branches(oracle)
        selected = dict(oracle["selected_v2_branch"])
        stable_alternatives = list(oracle.get("stable_alternatives", ()))
        rows.append(
            {
                "task_id": key[0],
                "solver_seed": key[1],
                "decision_index": key[2],
                "classification": str(oracle["classification"]),
                "candidate_count": len(explicit),
                "paired_trial_count": sum(
                    int(row["trial_count"]) for row in explicit
                ),
                "selected_rank": int(selected["candidate_rank"]),
                "selected_size": int(selected["candidate_size"]),
                "selected_escape_fraction": float(selected["escape_fraction"]),
                "selected_stable_failure": bool(selected["stable_failure"]),
                "stable_alternative_count": len(stable_alternatives),
                "best_stable_rank": (
                    min(int(row["candidate_rank"]) for row in stable_alternatives)
                    if stable_alternatives
                    else None
                ),
                "v3_risk_supported": bool(risk["local_signal_supported"]),
                "v3_trial_auc": float(risk["trial_no_progress_auc"]),
                "v3_candidate_spearman": float(
                    risk["candidate_no_progress_spearman"]
                ),
                "v3_selected_risk_rank": int(
                    risk["selected_v2"]["predicted_risk_rank"]
                ),
                "v3_best_stable_risk_rank": int(
                    risk["best_stable_alternative"]["predicted_risk_rank"]
                ),
                "portable_native_maximum_delta": float(
                    risk["portable_native_maximum_delta"]
                ),
                "oracle_sha256": source["oracle_sha256"],
                "risk_sha256": source["risk_sha256"],
            }
        )
    output_root = Path(output).resolve()
    identity = {
        "schema": STALL_ESCAPE_ORACLE_COHORT_SCHEMA,
        "schema_version": STALL_ESCAPE_ORACLE_COHORT_VERSION,
        "sources": manifests,
        "frozen_v3_manifest_sha256": next(iter(v3_hashes)),
    }
    runner = prepare_run_output(output_root, resume=resume, identity=identity)
    selector_failures = sum(row["classification"] == "selector_failure" for row in rows)
    supported = sum(bool(row["v3_risk_supported"]) for row in rows)
    report = {
        "schema": STALL_ESCAPE_ORACLE_COHORT_SCHEMA,
        "schema_version": STALL_ESCAPE_ORACLE_COHORT_VERSION,
        "run_fingerprint": str(runner["identity_fingerprint"]),
        "state_count": len(rows),
        "selector_failure_count": selector_failures,
        "candidate_pool_failure_count": sum(
            row["classification"] == "candidate_pool_failure" for row in rows
        ),
        "candidate_count": sum(int(row["candidate_count"]) for row in rows),
        "paired_trial_count": sum(int(row["paired_trial_count"]) for row in rows),
        "stable_alternative_count": sum(
            int(row["stable_alternative_count"]) for row in rows
        ),
        "frozen_v3_supported_state_count": supported,
        "mean_v3_trial_auc": statistics.fmean(
            float(row["v3_trial_auc"]) for row in rows
        ),
        "mean_v3_candidate_spearman": statistics.fmean(
            float(row["v3_candidate_spearman"]) for row in rows
        ),
        "portable_native_maximum_delta": max(
            float(row["portable_native_maximum_delta"]) for row in rows
        ),
        "decision": (
            "stall_specific_shadow_labels_required_frozen_v3_rejected"
            if selector_failures > 0 and supported == 0
            else "additional_exact_shadow_evidence_required"
        ),
        "deployment_promoted": False,
        "training_started": False,
        "evidence_level": "multi-state exact paired Oracle diagnostic",
        "next_step": (
            "collect independent stall-specific history labels in action-preserving "
            "shadow; do not reuse the generic frozen v3 risk head"
        ),
        "states": rows,
        "sources": manifests,
    }
    atomic_write_csv(output_root / "stall_escape_oracle_states.csv", rows)
    write_json(output_root / "stall_escape_oracle_cohort_report.json", report)
    lines = [
        "# v2 stall-escape Oracle cohort",
        "",
        f"- Decision: `{report['decision']}`.",
        f"- Exact states: {report['state_count']}.",
        f"- Selector failures: {report['selector_failure_count']}.",
        f"- Explicit candidates / paired trials: {report['candidate_count']} / {report['paired_trial_count']}.",
        f"- Stable alternatives: {report['stable_alternative_count']}.",
        f"- Frozen v3 risk supported states: {report['frozen_v3_supported_state_count']}.",
        f"- Mean trial AUC: {report['mean_v3_trial_auc']:.6f}.",
        f"- Mean candidate Spearman: {report['mean_v3_candidate_spearman']:.6f}.",
        "",
        "This is exact same-state diagnostic evidence, not an end-to-end controller promotion.",
        "",
    ]
    (output_root / "stall_escape_oracle_cohort_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


__all__ = [
    "STALL_ESCAPE_ORACLE_COHORT_SCHEMA",
    "build_stall_escape_oracle_cohort",
]
