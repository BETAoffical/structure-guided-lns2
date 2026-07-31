from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
sys.path.insert(0, str(PROJECT_ROOT))
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments._common import resolve_cli_path  # noqa: E402
from experiments.v3_s3_pipeline import (  # noqa: E402
    run_v3_s3_collection_stage,
    run_v3_s3_native_audit_stage,
    run_v3_s3_training_stage,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the retained V3-S3 collection, training, or native-audit "
            "stage. Each stage is invoked separately across Windows/WSL."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("source", "collect", "train", "native-audit"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument(
        "--dataset-config", default="configs/v3_s3_pilot_dataset.json"
    )
    parser.add_argument("--workers", default="4")
    parser.add_argument("--training-jobs", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parallelism-audit", action="store_true")
    parser.add_argument(
        "--reuse-sources",
        help="Reuse a completed V3-S3 source output for collection.",
    )
    parser.add_argument(
        "--reuse-collection",
        help="Reuse a completed V3-S3 collection for training.",
    )
    return parser


def _summary(stage: str, output: Path, report: dict[str, object]) -> dict[str, object]:
    training = report if stage == "train" else dict(report.get("training", {}))
    collection = dict(report.get("collection", {}))
    completed = collection.get("completed_state_count", report.get("completed_state_count"))
    if completed is None and training:
        completed = int(training.get("training_state_count", 0)) + int(
            training.get("diagnostic_state_count", 0)
        )
    return {
        "schema": "lns2.v3_s3_cli_summary.v2",
        "controller": "v3-s3",
        "stage": stage,
        "output": str(output),
        "complete": bool(report.get("complete", True)),
        "completed_states": completed,
        "error_states": collection.get(
            "error_state_count", report.get("error_state_count", 0)
        ),
        "decision": training.get("decision", report.get("decision")),
        "next_stage": report.get("next_stage"),
        "automatic_followup": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        output = resolve_cli_path(PROJECT_ROOT, arguments.output)
        controller_bundle = resolve_cli_path(
            PROJECT_ROOT, arguments.controller_bundle
        )
        dataset_config = resolve_cli_path(PROJECT_ROOT, arguments.dataset_config)
        reuse_sources = (
            resolve_cli_path(PROJECT_ROOT, arguments.reuse_sources)
            if arguments.reuse_sources
            else None
        )
        reuse_collection = (
            resolve_cli_path(PROJECT_ROOT, arguments.reuse_collection)
            if arguments.reuse_collection
            else None
        )
        if reuse_sources is not None and arguments.stage not in {"source", "collect"}:
            raise ValueError("--reuse-sources is valid only for source/collect")
        if reuse_collection is not None and arguments.stage != "train":
            raise ValueError("--reuse-collection is valid only for train")
        if arguments.stage in {"source", "collect"}:
            report = run_v3_s3_collection_stage(
                project_root=PROJECT_ROOT,
                output=output,
                controller_bundle=controller_bundle,
                dataset_config=dataset_config,
                workers=arguments.workers,
                resume=arguments.resume,
                parallelism_audit=arguments.parallelism_audit,
                stop_after_sources=arguments.stage == "source",
                reuse_source_output=reuse_sources,
            )
        elif arguments.stage == "train":
            report = run_v3_s3_training_stage(
                project_root=PROJECT_ROOT,
                output=output,
                training_jobs=arguments.training_jobs,
                resume=arguments.resume,
                collection_source=reuse_collection,
            )
        else:
            report = run_v3_s3_native_audit_stage(
                output=output,
                resume=arguments.resume,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            _summary(arguments.stage, output, dict(report)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
