from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_lns import build_stride_labels, run_stage1_audit  # noqa: E402
from experiments.stride_collection import (  # noqa: E402
    build_stride_state_selection,
    collect_stride_repairs,
    prepare_stride_pilot_dataset,
)
from experiments.stride_selection_v2 import (  # noqa: E402
    build_stride_state_selection_v2,
    preflight_stride_selection,
)
from experiments.stride_reuse import reuse_stride_collection  # noqa: E402
from experiments.stride_stability import (  # noqa: E402
    analyze_stride_stability,
    collect_stride_stability_trials,
    select_stride_stability_states,
)
from experiments.stride_quality_v2 import (  # noqa: E402
    analyze_stride_quality_v2_stability,
    build_stride_quality_v2_labels,
    collect_stride_quality_v2_confirmation_trials,
    collect_stride_quality_v2_completion_trials,
    collect_stride_quality_v2_design_trials,
    select_stride_quality_v2_completion_states,
    select_stride_quality_v2_confirmation_states,
)
from experiments.stride_stage3 import run_stride_stage3_label_audit  # noqa: E402


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
    prepare = subparsers.add_parser(
        "prepare-dataset", help="Merge the registered STRIDE Stage 2 Pilot dataset."
    )
    prepare.add_argument("--generated", required=True)
    prepare.add_argument("--movingai", required=True)
    prepare.add_argument("--output", default="build/stride-stage2-dataset-v1")
    selection = subparsers.add_parser(
        "select-states", help="Build the result-blind STRIDE Stage 2 Pilot cohort."
    )
    selection.add_argument("--source", action="append", required=True)
    selection.add_argument("--output", default="build/stride-stage2-pilot-selection-v1")
    selection.add_argument("--target-per-policy", type=int, default=120)
    selection.add_argument("--max-per-episode", type=int, default=2)
    selection_v2 = subparsers.add_parser(
        "select-states-v2", help="Build the decision-capped STRIDE Stage 2 cohort."
    )
    selection_v2.add_argument("--source", action="append", required=True)
    selection_v2.add_argument("--output", default="build/stride-stage2-pilot-selection-v2")
    selection_v2.add_argument("--exclude-report")
    selection_v2.add_argument("--target-per-policy", type=int, default=120)
    selection_v2.add_argument("--max-per-episode", type=int, default=2)
    selection_v2.add_argument(
        "--split",
        default="stride_pilot",
        help=(
            "Required source split, or 'auto' to accept each source's own "
            "registered split when combining Stage 3 extensions."
        ),
    )
    preflight = subparsers.add_parser(
        "preflight-selection", help="Repeat replay and candidate generation before PP trials."
    )
    preflight.add_argument("--selection", required=True)
    preflight.add_argument("--output", default="build/stride-stage2-pilot-preflight-v1")
    preflight.add_argument("--workers", type=int, default=4)
    preflight.add_argument("--repetitions", type=int, default=3)
    reuse = subparsers.add_parser(
        "reuse-collection", help="Audit and reuse complete state artifacts for a new cohort."
    )
    reuse.add_argument("--selection", required=True)
    reuse.add_argument("--source", required=True)
    reuse.add_argument("--preflight-report", required=True)
    reuse.add_argument("--output", required=True)
    stability_select = subparsers.add_parser(
        "select-stability", help="Select the result-blind 20 percent stability cohort."
    )
    stability_select.add_argument("--selection", required=True)
    stability_select.add_argument("--output", default="build/stride-stage2-stability-selection-v1")
    stability_select.add_argument("--count-per-policy", type=int, default=24)
    stability_collect = subparsers.add_parser(
        "collect-stability", help="Collect PP trial indices 4 through 7."
    )
    stability_collect.add_argument("--selection", required=True)
    stability_collect.add_argument("--collection", required=True)
    stability_collect.add_argument("--output", default="build/stride-stage2-stability-v1")
    stability_collect.add_argument("--workers", type=int, default=4)
    stability_analyze = subparsers.add_parser(
        "analyze-stability", help="Compare the first and second four-seed halves."
    )
    stability_analyze.add_argument("--base-trials", required=True)
    stability_analyze.add_argument("--extra-trials", required=True)
    stability_analyze.add_argument("--output", default="build/stride-stage2-stability-analysis-v1")
    quality_v2_design = subparsers.add_parser(
        "collect-quality-v2-design",
        help="Collect trial indices 8 through 15 on the consumed V2 design cohort.",
    )
    quality_v2_design.add_argument("--selection", required=True)
    quality_v2_design.add_argument("--collection", required=True)
    quality_v2_design.add_argument(
        "--output", default="build/stride-quality-v2-design-extension-v1"
    )
    quality_v2_design.add_argument("--workers", type=int, default=4)
    quality_v2_select = subparsers.add_parser(
        "select-quality-v2-confirmation",
        help="Select a result-blind V2 confirmation cohort disjoint from design episodes.",
    )
    quality_v2_select.add_argument("--selection", required=True)
    quality_v2_select.add_argument("--design-selection", required=True)
    quality_v2_select.add_argument(
        "--output", default="build/stride-quality-v2-confirmation-selection-v1"
    )
    quality_v2_select.add_argument("--count-per-policy", type=int, default=24)
    quality_v2_confirm = subparsers.add_parser(
        "collect-quality-v2-confirmation",
        help="Collect trial indices 4 through 15 on the untouched confirmation cohort.",
    )
    quality_v2_confirm.add_argument("--selection", required=True)
    quality_v2_confirm.add_argument("--collection", required=True)
    quality_v2_confirm.add_argument(
        "--output", default="build/stride-quality-v2-confirmation-v1"
    )
    quality_v2_confirm.add_argument("--workers", type=int, default=4)
    quality_v2_completion_select = subparsers.add_parser(
        "select-quality-v2-completion",
        help="Select registered states that still lack trial indices 4 through 7.",
    )
    quality_v2_completion_select.add_argument("--selection", required=True)
    quality_v2_completion_select.add_argument("--base-trials", required=True)
    quality_v2_completion_select.add_argument(
        "--extension-trials", action="append", required=True
    )
    quality_v2_completion_select.add_argument(
        "--output", default="build/stride-quality-v2-completion-selection-v1"
    )
    quality_v2_completion = subparsers.add_parser(
        "collect-quality-v2-completion",
        help="Collect trial indices 4 through 7 for an eight-seed label.",
    )
    quality_v2_completion.add_argument("--selection", required=True)
    quality_v2_completion.add_argument("--collection", required=True)
    quality_v2_completion.add_argument(
        "--output", default="build/stride-quality-v2-completion-v1"
    )
    quality_v2_completion.add_argument("--workers", type=int, default=4)
    quality_v2_analyze = subparsers.add_parser(
        "analyze-quality-v2",
        help="Compare independent eight-seed halves under the V2 quality score.",
    )
    quality_v2_analyze.add_argument("--trials", action="append", required=True)
    quality_v2_analyze.add_argument(
        "--output", default="build/stride-quality-v2-stability-analysis-v1"
    )
    quality_v2_analyze.add_argument("--expected-state-count", type=int, default=48)
    quality_v2_label = subparsers.add_parser(
        "label-quality-v2", help="Build state-balanced eight-seed quality V2 labels."
    )
    quality_v2_label.add_argument("--trials", action="append", required=True)
    quality_v2_label.add_argument("--output", default="build/stride-quality-v2-labels-v1")
    stage3_audit = subparsers.add_parser(
        "audit-stage3-labels",
        help="Audit the registered 600-state Stage 3 quality-label cohort.",
    )
    stage3_audit.add_argument(
        "--config", default="configs/stride_stage3_label_audit.json"
    )
    stage3_audit.add_argument(
        "--output", default="build/stride-stage3-label-audit-v1"
    )
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
    if arguments.stage == "prepare-dataset":
        summary = prepare_stride_pilot_dataset(
            generated=_resolve(arguments.generated),
            movingai=_resolve(arguments.movingai),
            output=_resolve(arguments.output),
        )
        print(f"stride_pilot_dataset_ready:{summary['task_count']}")
        return 0
    if arguments.stage == "select-states":
        report = build_stride_state_selection(
            source_roots=[_resolve(value) for value in arguments.source],
            output=_resolve(arguments.output),
            target_per_policy=arguments.target_per_policy,
            max_per_episode=arguments.max_per_episode,
        )
        print("stride_selection_passed" if report["passed"] else "stride_selection_failed")
        return 0 if report["passed"] else 2
    if arguments.stage == "select-states-v2":
        report = build_stride_state_selection_v2(
            source_roots=[_resolve(value) for value in arguments.source],
            output=_resolve(arguments.output),
            exclusion_report=(
                _resolve(arguments.exclude_report) if arguments.exclude_report else None
            ),
            target_per_policy=arguments.target_per_policy,
            max_per_episode=arguments.max_per_episode,
            split=arguments.split,
        )
        print("stride_selection_v2_passed" if report["passed"] else "stride_selection_v2_failed")
        return 0 if report["passed"] else 2
    if arguments.stage == "preflight-selection":
        report = preflight_stride_selection(
            selection_path=_resolve(arguments.selection),
            output=_resolve(arguments.output),
            workers=arguments.workers,
            repetitions=arguments.repetitions,
        )
        print("stride_preflight_passed" if report["passed"] else "stride_preflight_failed")
        return 0 if report["passed"] else 2
    if arguments.stage == "reuse-collection":
        report = reuse_stride_collection(
            selection_path=_resolve(arguments.selection),
            source=_resolve(arguments.source),
            output=_resolve(arguments.output),
            preflight_report=_resolve(arguments.preflight_report),
        )
        print(
            f"stride_reuse_ready:{report['reused_state_count']}:"
            f"{report['pending_state_count']}"
        )
        return 0
    if arguments.stage == "select-stability":
        report = select_stride_stability_states(
            selection_path=_resolve(arguments.selection),
            output=_resolve(arguments.output),
            count_per_policy=arguments.count_per_policy,
        )
        print("stride_stability_selection_passed" if report["passed"] else "stride_stability_selection_failed")
        return 0 if report["passed"] else 2
    if arguments.stage == "collect-stability":
        report = collect_stride_stability_trials(
            selection_path=_resolve(arguments.selection),
            collection=_resolve(arguments.collection),
            output=_resolve(arguments.output),
            workers=arguments.workers,
        )
        print("stride_stability_collection_complete" if report["complete"] else "stride_stability_collection_failed")
        return 0 if report["complete"] else 2
    if arguments.stage == "analyze-stability":
        report = analyze_stride_stability(
            base_trials=_resolve(arguments.base_trials),
            extra_trials=_resolve(arguments.extra_trials),
            output=_resolve(arguments.output),
        )
        print("stride_stability_passed" if report["passed"] else "stride_stability_failed")
        return 0 if report["passed"] else 2
    if arguments.stage == "collect-quality-v2-design":
        report = collect_stride_quality_v2_design_trials(
            selection_path=_resolve(arguments.selection),
            collection=_resolve(arguments.collection),
            output=_resolve(arguments.output),
            workers=arguments.workers,
        )
        print("stride_quality_v2_design_complete" if report["complete"] else "stride_quality_v2_design_failed")
        return 0 if report["complete"] else 2
    if arguments.stage == "select-quality-v2-confirmation":
        report = select_stride_quality_v2_confirmation_states(
            selection_path=_resolve(arguments.selection),
            design_selection_path=_resolve(arguments.design_selection),
            output=_resolve(arguments.output),
            count_per_policy=arguments.count_per_policy,
        )
        print("stride_quality_v2_selection_passed" if report["passed"] else "stride_quality_v2_selection_failed")
        return 0 if report["passed"] else 2
    if arguments.stage == "collect-quality-v2-confirmation":
        report = collect_stride_quality_v2_confirmation_trials(
            selection_path=_resolve(arguments.selection),
            collection=_resolve(arguments.collection),
            output=_resolve(arguments.output),
            workers=arguments.workers,
        )
        print("stride_quality_v2_confirmation_complete" if report["complete"] else "stride_quality_v2_confirmation_failed")
        return 0 if report["complete"] else 2
    if arguments.stage == "select-quality-v2-completion":
        report = select_stride_quality_v2_completion_states(
            selection_path=_resolve(arguments.selection),
            base_trials=_resolve(arguments.base_trials),
            extension_trial_paths=[
                _resolve(value) for value in arguments.extension_trials
            ],
            output=_resolve(arguments.output),
        )
        print(f"stride_quality_v2_completion_pending:{report['pending_state_count']}")
        return 0 if report["passed"] else 2
    if arguments.stage == "collect-quality-v2-completion":
        report = collect_stride_quality_v2_completion_trials(
            selection_path=_resolve(arguments.selection),
            collection=_resolve(arguments.collection),
            output=_resolve(arguments.output),
            workers=arguments.workers,
        )
        print("stride_quality_v2_completion_complete" if report["complete"] else "stride_quality_v2_completion_failed")
        return 0 if report["complete"] else 2
    if arguments.stage == "analyze-quality-v2":
        report = analyze_stride_quality_v2_stability(
            trial_paths=[_resolve(value) for value in arguments.trials],
            output=_resolve(arguments.output),
            expected_state_count=arguments.expected_state_count,
        )
        print("stride_quality_v2_stability_passed" if report["passed"] else "stride_quality_v2_stability_failed")
        return 0 if report["passed"] else 2
    if arguments.stage == "label-quality-v2":
        report = build_stride_quality_v2_labels(
            trial_paths=[_resolve(value) for value in arguments.trials],
            output=_resolve(arguments.output),
        )
        print(f"stride_quality_v2_labels_ready:{report['state_count']}")
        return 0
    if arguments.stage == "audit-stage3-labels":
        report = run_stride_stage3_label_audit(
            config_path=_resolve(arguments.config),
            output=_resolve(arguments.output),
            project_root=PROJECT_ROOT,
        )
        print(
            "stride_stage3_label_audit_passed"
            if report["passed"]
            else "stride_stage3_label_audit_failed"
        )
        return 0 if report["passed"] else 2
    raise AssertionError(f"unhandled stage: {arguments.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
