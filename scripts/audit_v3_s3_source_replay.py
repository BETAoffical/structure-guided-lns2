from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
sys.path.insert(0, str(PROJECT_ROOT))
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.repair_collection import _plain, state_fingerprint  # noqa: E402
from experiments.trace_replay import replay_prefix  # noqa: E402
from experiments.v3_s3_collection import (  # noqa: E402
    _source_replay_job,
    source_decisions,
)
from experiments.v3_s3_pipeline import source_roots  # noqa: E402


def audit_source_replay(
    source_output: Path,
    *,
    roots: dict[str, list[Path]] | None = None,
) -> dict[str, object]:
    rows = source_decisions(roots if roots is not None else source_roots(source_output))
    grouped: dict[tuple[str, str], list[dict[str, object]]] = collections.defaultdict(list)
    for row in rows:
        grouped[(str(row["source_root"]), str(row["episode_id"]))].append(row)

    matched = 0
    rejected = []
    mismatches = []
    terminal_after_mismatches = []
    by_policy: collections.Counter[str] = collections.Counter()
    for (_root, episode_id), episode_rows in sorted(grouped.items()):
        episode_rows.sort(key=lambda row: int(row["decision_index"]))
        replay, _configuration = _source_replay_job(episode_rows[0])
        environment, state = replay_prefix(replay, ())
        episode_failed = False
        applied_actions: list[dict[str, object]] = []
        for row in episode_rows:
            prefix = [dict(action) for action in row["prefix_actions"]]
            if prefix[: len(applied_actions)] != applied_actions:
                mismatches.append(
                    {
                        "episode_id": episode_id,
                        "decision_index": int(row["decision_index"]),
                        "reason": "canonical prefixes are not nested",
                    }
                )
                episode_failed = True
                break
            for action in prefix[len(applied_actions) :]:
                if bool(state["done"]):
                    rejected.append(
                        {
                            "episode_id": episode_id,
                            "decision_index": int(row["decision_index"]),
                            "reason": "prefix terminated before target state",
                        }
                    )
                    episode_failed = True
                    break
                state = dict(_plain(environment.step(action))["observation"])
                applied_actions.append(action)
            if episode_failed:
                break
            observed = state_fingerprint(state)
            expected = str(row["before_fingerprint"])
            if observed != expected:
                mismatches.append(
                    {
                        "episode_id": episode_id,
                        "decision_index": int(row["decision_index"]),
                        "expected": expected,
                        "observed": observed,
                    }
                )
                episode_failed = True
                break
            matched += 1
            by_policy[str(row["source_policy"])] += 1
        if (
            not episode_failed
            and episode_rows
        ):
            final_action = dict(episode_rows[-1]["replay_action"])
            if not bool(state["done"]):
                state = dict(_plain(environment.step(final_action))["observation"])
            final_matches = state_fingerprint(state) == str(
                episode_rows[-1]["after_fingerprint"]
            )
        else:
            final_matches = True
        if not final_matches:
            # The final post-step state is never used as a sampled decision
            # state. Keep the diagnostic visible without weakening any prefix
            # fingerprint requirement.
            terminal_after_mismatches.append(
                {
                    "episode_id": episode_id,
                    "decision_index": int(episode_rows[-1]["decision_index"]),
                }
            )

    return {
        "schema": "lns2.v3_s3_source_replay_audit.v2",
        "source_output": str(source_output),
        "source_state_count": len(rows),
        "episode_count": len(grouped),
        "matched_decision_state_count": matched,
        "rejected_episode_count": len(rejected),
        "rejections": rejected,
        "prefix_mismatch_count": len(mismatches),
        "prefix_mismatches": mismatches,
        "terminal_after_mismatch_count": len(terminal_after_mismatches),
        "terminal_after_mismatches": terminal_after_mismatches,
        "matched_by_source_policy": dict(sorted(by_policy.items())),
        "passed": (
            not rejected
            and not mismatches
            and not terminal_after_mismatches
            and matched == len(rows)
            and matched > 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit canonical v3-S3 source prefixes against recorded fingerprints."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-output")
    source_group.add_argument(
        "--source-root",
        help="Audit one closed-loop source collection instead of a six-source pipeline output.",
    )
    parser.add_argument(
        "--split",
        choices=("policy_train", "policy_validation"),
        default="policy_train",
        help="Split associated with --source-root.",
    )
    parser.add_argument("--report")
    arguments = parser.parse_args()
    source = Path(arguments.source_output or arguments.source_root)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    resolved = source.resolve()
    roots = (
        {str(arguments.split): [resolved]}
        if arguments.source_root is not None
        else None
    )
    report = audit_source_replay(resolved, roots=roots)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if arguments.report:
        destination = Path(arguments.report)
        if not destination.is_absolute():
            destination = PROJECT_ROOT / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(destination)
    print(text, end="")
    return 0 if bool(report["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
