from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_lns import build_stride_labels, run_stage1_audit  # noqa: E402
from experiments.stride_collection import collect_stride_repairs  # noqa: E402


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the STRIDE-LNS six-stage pipeline.")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    audit = subparsers.add_parser(
        "audit", help="Run Stage 1 frozen-baseline and historical-data audit."
    )
    audit.add_argument(
        "--config", default="configs/stride_stage1_audit.json"
    )
    audit.add_argument("--output", default="build/stride-stage1-audit-v1")
    label = subparsers.add_parser(
        "label", help="Build Stage 2 quality labels from paired repair trials."
    )
    label.add_argument("--trials", required=True)
    label.add_argument("--output", default="build/stride-stage2-labels-v1")
    collect = subparsers.add_parser(
        "collect", help="Collect Stage 2 full-pool paired native repair trials."
    )
    collect.add_argument("--selection", required=True)
    collect.add_argument("--output", default="build/stride-stage2-pilot-v1")
    collect.add_argument("--workers", type=int, default=1)
    collect.add_argument("--max-states", type=int)
    collect.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.stage == "audit":
        report = run_stage1_audit(
            config_path=_resolve(arguments.config),
            output=_resolve(arguments.output),
            project_root=PROJECT_ROOT,
        )
        print(report["decision"])
        return 0 if report["stage1_passed"] else 2
    if arguments.stage == "label":
        summary = build_stride_labels(
            trials_path=_resolve(arguments.trials), output=_resolve(arguments.output)
        )
        print(
            "stride_label_coverage_passed"
            if summary["coverage_gate_passed"]
            else "stride_label_coverage_failed"
        )
        return 0 if summary["coverage_gate_passed"] else 2
    if arguments.stage == "collect":
        report = collect_stride_repairs(
            selection_path=_resolve(arguments.selection),
            output=_resolve(arguments.output),
            workers=arguments.workers,
            resume=arguments.resume,
            max_states=arguments.max_states,
        )
        print("stride_collection_complete" if report["complete"] else "stride_collection_failed")
        return 0 if report["complete"] else 2
    raise AssertionError(f"unhandled stage: {arguments.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
