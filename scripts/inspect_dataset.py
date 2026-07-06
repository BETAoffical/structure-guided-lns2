from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a generated stage 1 dataset."
    )
    parser.add_argument("--dataset", required=True)
    arguments = parser.parse_args()
    root = Path(arguments.dataset)
    summary_path = root / "dataset_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"master seed: {summary['master_seed']}")
    print(f"dataset: {root}")
    all_seeds: list[int] = []
    layout_counts: collections.Counter[str] = collections.Counter()
    scenario_counts: collections.Counter[str] = collections.Counter()
    layout_variant_counts: collections.Counter[str] = collections.Counter()
    task_variant_counts: collections.Counter[str] = collections.Counter()
    for name, split in summary["splits"].items():
        all_seeds.extend(split["map_seeds"])
        print(
            f"{name}: maps={split['map_count']}, "
            f"instances={split['instance_count']}, "
            f"agents={split['agent_count_min']}.."
            f"{split['agent_count_max']}"
        )
        manifest_path = root / name / "manifest.jsonl"
        if manifest_path.exists():
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                layout_counts[row.get("layout_mode", "unknown")] += 1
                scenario_counts[row.get("scenario_type", "unknown")] += 1
                if row.get("layout_variant"):
                    layout_variant_counts[row["layout_variant"]] += 1
                if row.get("task_variant"):
                    task_variant_counts[row["task_variant"]] += 1
    print(f"map seeds unique: {len(all_seeds) == len(set(all_seeds))}")
    print(f"layout instances: {dict(sorted(layout_counts.items()))}")
    print(f"task scenarios: {dict(sorted(scenario_counts.items()))}")
    print(
        "layout variants: "
        f"{dict(sorted(layout_variant_counts.items()))}"
    )
    print(
        f"task variants: {dict(sorted(task_variant_counts.items()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
