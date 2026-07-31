from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import lns2_env  # noqa: E402


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logical_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "conflicts": int(state["num_of_colliding_pairs"]),
        "sum_of_costs": int(state["sum_of_costs"]),
        "done": bool(state["done"]),
        "feasible": bool(state["feasible"]),
        "paths": [list(map(int, agent["path"])) for agent in state["agents"]],
        "conflict_edges": [
            [int(left), int(right)] for left, right in state["conflict_edges"]
        ],
    }


def _episode(
    *,
    map_path: Path,
    scenario_path: Path,
    agent_count: int,
    seed: int,
    repair_steps: int,
) -> dict[str, Any]:
    environment = lns2_env.LNS2RepairEnv(
        str(map_path),
        str(scenario_path),
        agent_count=agent_count,
        time_limit=60.0,
        neighborhood_size=8,
        max_repair_iterations=0,
        context={"audit": "corrected-native-v1"},
    )
    state = dict(environment.reset(seed=seed))
    initial = _logical_state(state)
    proposals: list[dict[str, Any]] = []
    active_agents = sorted(
        {
            int(agent)
            for edge in state["conflict_edges"]
            for agent in edge
        }
    )
    for seed_index, seed_agent in enumerate(active_agents[:4]):
        for family_index, heuristic in enumerate(("target", "collision", "random")):
            for size in (4, 8, 16):
                random_seed = (
                    1_000_000
                    + seed * 10_000
                    + seed_index * 1_000
                    + family_index * 100
                    + size
                )
                proposal = dict(
                    environment.propose(
                        {
                            "mode": "seed",
                            "heuristic": heuristic,
                            "seed_agent": seed_agent,
                            "neighborhood_size": size,
                            "random_seed": random_seed,
                        }
                    )
                )
                proposals.append(
                    {
                        "seed_agent": seed_agent,
                        "heuristic": heuristic,
                        "requested_size": size,
                        "random_seed": random_seed,
                        "action_valid": bool(proposal["action_valid"]),
                        "generated": bool(proposal["generated"]),
                        "neighborhood": list(map(int, proposal["neighborhood"])),
                    }
                )
    if _logical_state(dict(environment.get_state())) != initial:
        raise RuntimeError("proposal audit changed the logical repair state")

    trajectory = [int(state["num_of_colliding_pairs"])]
    applied_repairs = []
    for step_index in range(repair_steps):
        if bool(state["done"]):
            break
        transition = dict(
            environment.step(
                {
                    "mode": "official",
                    "random_seed": 2_000_000 + seed * 1_000 + step_index,
                }
            )
        )
        state = dict(transition["observation"])
        metrics = dict(transition["metrics"])
        trajectory.append(int(state["num_of_colliding_pairs"]))
        applied_repairs.append(
            {
                "step": step_index + 1,
                "neighborhood": list(map(int, metrics.get("neighborhood", ()))),
                "repair_order": list(map(int, metrics.get("repair_order", ()))),
                "replan_success": bool(metrics.get("replan_success")),
                "conflicts_after": int(state["num_of_colliding_pairs"]),
                "sum_of_costs_after": int(state["sum_of_costs"]),
            }
        )
    final = _logical_state(state)
    return {
        "seed": seed,
        "initial": {
            "conflicts": initial["conflicts"],
            "sum_of_costs": initial["sum_of_costs"],
            "state_fingerprint": _fingerprint(initial),
        },
        "proposal_fingerprint": _fingerprint(proposals),
        "proposals": proposals,
        "repair_trajectory": trajectory,
        "repair_fingerprint": _fingerprint(applied_repairs),
        "repairs": applied_repairs,
        "final": {
            "conflicts": final["conflicts"],
            "sum_of_costs": final["sum_of_costs"],
            "state_fingerprint": _fingerprint(final),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record fixed-seed native reset, proposal, and repair semantics for "
            "legacy-versus-corrected binary comparison."
        )
    )
    parser.add_argument("--map", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--agent-count", type=int, default=80)
    parser.add_argument("--seeds", default="17,29,31")
    parser.add_argument("--repair-steps", type=int, default=10)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    seeds = [int(value) for value in arguments.seeds.split(",") if value.strip()]
    if not seeds or any(value < 0 for value in seeds):
        parser.error("seeds must contain non-negative integers")
    if arguments.agent_count <= 0 or arguments.repair_steps < 0:
        parser.error("agent-count must be positive and repair-steps non-negative")

    map_path = Path(arguments.map).resolve()
    scenario_path = Path(arguments.scenario).resolve()
    native_path = Path(str(lns2_env.__file__)).resolve()
    report = {
        "schema": "lns2.corrected_native_semantics_audit.v1",
        "native": {
            "path": str(native_path),
            "sha256": _file_sha256(native_path),
            "repair_timing_schema": str(
                getattr(lns2_env, "repair_timing_schema", "")
            ),
            "semantics_schema": str(
                getattr(lns2_env, "native_semantics_schema", "legacy-unversioned")
            ),
        },
        "input": {
            "map": str(map_path),
            "map_sha256": _file_sha256(map_path),
            "scenario": str(scenario_path),
            "scenario_sha256": _file_sha256(scenario_path),
            "agent_count": int(arguments.agent_count),
            "repair_steps": int(arguments.repair_steps),
            "seeds": seeds,
        },
        "episodes": [
            _episode(
                map_path=map_path,
                scenario_path=scenario_path,
                agent_count=int(arguments.agent_count),
                seed=seed,
                repair_steps=int(arguments.repair_steps),
            )
            for seed in seeds
        ],
    }
    output = Path(arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "output": str(output),
                "native": report["native"],
                "episodes": [
                    {
                        "seed": row["seed"],
                        "initial": row["initial"],
                        "proposal_fingerprint": row["proposal_fingerprint"],
                        "repair_fingerprint": row["repair_fingerprint"],
                        "repair_trajectory": row["repair_trajectory"],
                        "final": row["final"],
                    }
                    for row in report["episodes"]
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
