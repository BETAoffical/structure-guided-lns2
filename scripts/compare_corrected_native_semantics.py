from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "lns2.corrected_native_semantics_comparison.v1"
SOURCE_SCHEMA = "lns2.corrected_native_semantics_audit.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"unexpected semantics audit schema: {path}")
    return value


def _episode_index(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = list(report.get("episodes", ()))
    index = {int(row["seed"]): dict(row) for row in rows}
    if len(index) != len(rows):
        raise ValueError("semantics audit contains duplicate seeds")
    return index


def _proposal_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["heuristic"]),
        int(row["requested_size"]),
        int(row["random_seed"]),
    )


def compare(baseline: dict[str, Any], corrected: dict[str, Any]) -> dict[str, Any]:
    baseline_input = dict(baseline.get("input") or {})
    corrected_input = dict(corrected.get("input") or {})
    comparable_input = {
        "map_sha256",
        "scenario_sha256",
        "agent_count",
        "repair_steps",
        "seeds",
    }
    for field in comparable_input:
        if baseline_input.get(field) != corrected_input.get(field):
            raise ValueError(
                f"baseline and corrected audits use different {field}"
            )

    baseline_index = _episode_index(baseline)
    corrected_index = _episode_index(corrected)
    if set(baseline_index) != set(corrected_index):
        raise ValueError("baseline and corrected audits cover different seeds")

    episodes = []
    proposal_total = 0
    proposal_changed = 0
    proposal_changed_by_family: dict[str, int] = {}
    for seed in sorted(baseline_index):
        left = baseline_index[seed]
        right = corrected_index[seed]
        left_proposals = {
            _proposal_key(row): dict(row) for row in left["proposals"]
        }
        right_proposals = {
            _proposal_key(row): dict(row) for row in right["proposals"]
        }
        if set(left_proposals) != set(right_proposals):
            raise ValueError(f"proposal coverage differs for seed {seed}")
        changed_keys = []
        for key in sorted(left_proposals):
            proposal_total += 1
            if left_proposals[key] != right_proposals[key]:
                proposal_changed += 1
                proposal_changed_by_family[key[0]] = (
                    proposal_changed_by_family.get(key[0], 0) + 1
                )
                changed_keys.append(
                    {
                        "heuristic": key[0],
                        "requested_size": key[1],
                        "random_seed": key[2],
                        "baseline_seed_agent": int(
                            left_proposals[key]["seed_agent"]
                        ),
                        "corrected_seed_agent": int(
                            right_proposals[key]["seed_agent"]
                        ),
                    }
                )
        episodes.append(
            {
                "seed": seed,
                "initial_state_changed": (
                    left["initial"]["state_fingerprint"]
                    != right["initial"]["state_fingerprint"]
                ),
                "initial_conflict_delta": (
                    int(right["initial"]["conflicts"])
                    - int(left["initial"]["conflicts"])
                ),
                "proposal_count": len(left_proposals),
                "changed_proposal_count": len(changed_keys),
                "changed_proposals": changed_keys,
                "repair_sequence_changed": (
                    str(left["repair_fingerprint"])
                    != str(right["repair_fingerprint"])
                ),
                "baseline_repair_trajectory": list(left["repair_trajectory"]),
                "corrected_repair_trajectory": list(right["repair_trajectory"]),
                "final_state_changed": (
                    left["final"]["state_fingerprint"]
                    != right["final"]["state_fingerprint"]
                ),
            }
        )
    return {
        "schema": SCHEMA,
        "baseline_native": baseline["native"],
        "corrected_native": corrected["native"],
        "seed_count": len(episodes),
        "initial_state_changed_count": sum(
            bool(row["initial_state_changed"]) for row in episodes
        ),
        "proposal_count": proposal_total,
        "changed_proposal_count": proposal_changed,
        "changed_proposal_fraction": (
            proposal_changed / proposal_total if proposal_total else 0.0
        ),
        "changed_proposal_count_by_family": dict(
            sorted(proposal_changed_by_family.items())
        ),
        "repair_sequence_changed_count": sum(
            bool(row["repair_sequence_changed"]) for row in episodes
        ),
        "final_state_changed_count": sum(
            bool(row["final_state_changed"]) for row in episodes
        ),
        "episodes": episodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare legacy and corrected fixed-seed native semantics."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--corrected", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    baseline_path = Path(arguments.baseline).resolve()
    corrected_path = Path(arguments.corrected).resolve()
    report = compare(_read(baseline_path), _read(corrected_path))
    report["input"] = {
        "baseline": str(baseline_path),
        "baseline_sha256": _sha256(baseline_path),
        "corrected": str(corrected_path),
        "corrected_sha256": _sha256(corrected_path),
    }
    output = Path(arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
