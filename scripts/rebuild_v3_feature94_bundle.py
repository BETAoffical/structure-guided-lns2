from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
sys.path.insert(0, str(PROJECT_ROOT))
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments._common import resolve_cli_path, sha256_file, write_json  # noqa: E402
from experiments.feature_schema_v3 import (  # noqa: E402
    V3_FEATURE_SCHEMA_SHA256,
    v3_feature_schema_manifest,
)
from experiments.repair_collection import _read_json, _utc_now  # noqa: E402
from experiments.run_output_guard import prepare_run_output  # noqa: E402
from experiments.v3_bundle_equivalence import (  # noqa: E402
    audit_v3_bundle_equivalence,
)
from experiments.v3_training import (  # noqa: E402
    finalize_v3_native_audit,
    train_v3_controller,
)


def _portable_identity_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _identity(
    *, source: Path, reference_bundle: Path, main_controller_bundle: Path
) -> dict[str, object]:
    implementation_files = (
        "scripts/rebuild_v3_feature94_bundle.py",
        "experiments/feature_schema_v3.py",
        "experiments/v3_training.py",
        "experiments/v3_controller.py",
        "experiments/v3_bundle_equivalence.py",
    )
    return {
        "runner": "rebuild_v3_feature94_bundle.v1",
        "source": _portable_identity_path(source),
        "feature_index_sha256": sha256_file(
            source / "collection" / "feature_index.jsonl"
        ),
        "trial_manifest_sha256": sha256_file(
            source / "collection" / "trial_manifest.jsonl"
        ),
        "reference_bundle": _portable_identity_path(reference_bundle),
        "reference_manifest_sha256": sha256_file(
            reference_bundle / "v3_manifest.json"
        ),
        "main_controller_bundle": _portable_identity_path(main_controller_bundle),
        "main_controller_manifest_sha256": sha256_file(
            main_controller_bundle / "controller_manifest.json"
        ),
        "feature_schema_sha256": V3_FEATURE_SCHEMA_SHA256,
        "implementation": {
            name: sha256_file(PROJECT_ROOT / name) for name in implementation_files
        },
        "automatic_followup": False,
    }


def _status(output: Path, **values: object) -> None:
    write_json(
        output / "status.json",
        {
            "schema": "lns2.v3_feature94_rebuild_status.v1",
            "updated_at": _utc_now(),
            **values,
        },
    )


def _coverage(source: Path) -> dict[str, object]:
    report = dict(_read_json(source / "collection" / "collection_report.json"))
    coverage = dict(report.get("coverage") or {})
    if not bool(report.get("complete")) or not bool(coverage.get("passed")):
        raise ValueError("source v3 pilot collection is incomplete")
    if int(report.get("completed_state_count", 0)) != 180:
        raise ValueError("source v3 pilot must contain 180 completed states")
    if int(coverage.get("candidate_count", 0)) != 3412:
        raise ValueError("source v3 pilot candidate coverage differs from 3412")
    if int(coverage.get("trial_count", 0)) != 7262:
        raise ValueError("source v3 pilot trial coverage differs from 7262")
    return report


def run_stage(
    *,
    stage: str,
    source: Path,
    output: Path,
    reference_bundle: Path,
    main_controller_bundle: Path,
    resume: bool,
    benchmark_repeats: int,
) -> dict[str, object]:
    identity = _identity(
        source=source,
        reference_bundle=reference_bundle,
        main_controller_bundle=main_controller_bundle,
    )
    prepare_run_output(output, resume=resume, identity=identity)
    collection = _coverage(source)
    feature_index = source / "collection" / "feature_index.jsonl"
    trial_manifest = source / "collection" / "trial_manifest.jsonl"
    controller_output = output / "controller"
    started = time.perf_counter()
    try:
        if stage == "train":
            _status(output, status="running", phase="training", stage=stage)
            selection_overhead = float(
                dict(collection["selection"])["selection_overhead_seconds"]
            )
            report = train_v3_controller(
                feature_index=feature_index,
                trial_manifest=trial_manifest,
                controller_bundle=main_controller_bundle,
                output=controller_output,
                selection_overhead_seconds=selection_overhead,
            )
            result: dict[str, object] = {
                "stage": stage,
                "complete": True,
                "elapsed_seconds": time.perf_counter() - started,
                "feature_schema": v3_feature_schema_manifest(),
                "training": report,
                "next_stage": "native-audit",
            }
            write_json(output / "training_stage_report.json", result)
            _status(
                output,
                status="running",
                phase="awaiting-native-audit",
                stage=stage,
                elapsed_seconds=result["elapsed_seconds"],
            )
            return result
        if not controller_output.is_dir():
            raise FileNotFoundError("94-feature v3 controller output is missing")
        if stage == "native-audit":
            _status(output, status="running", phase="native-audit", stage=stage)
            report = finalize_v3_native_audit(
                feature_index=feature_index,
                trial_manifest=trial_manifest,
                controller_output=controller_output,
            )
            result = {
                "stage": stage,
                "complete": True,
                "elapsed_seconds": time.perf_counter() - started,
                "training": report,
                "next_stage": "verify",
            }
            write_json(output / "native_audit_stage_report.json", result)
            _status(
                output,
                status="running",
                phase="awaiting-equivalence",
                stage=stage,
                elapsed_seconds=result["elapsed_seconds"],
            )
            return result
        if stage == "verify":
            _status(output, status="running", phase="equivalence", stage=stage)
            equivalence = audit_v3_bundle_equivalence(
                feature_index=feature_index,
                reference_bundle=reference_bundle,
                candidate_bundle=controller_output,
                output=output / "equivalence",
                main_controller_bundle=main_controller_bundle,
                require_native=True,
                benchmark_repeats=benchmark_repeats,
            )
            result = {
                "schema": "lns2.v3_feature94_rebuild.v1",
                "stage": stage,
                "complete": True,
                "passed": bool(equivalence["passed"]),
                "elapsed_seconds": time.perf_counter() - started,
                "feature_schema": v3_feature_schema_manifest(),
                "source_coverage": dict(collection["coverage"]),
                "equivalence": equivalence,
                "seed2_or_seed3_started": False,
                "quick_started": False,
                "formal_started": False,
            }
            write_json(output / "v3_feature94_rebuild_report.json", result)
            _status(
                output,
                status="complete" if result["passed"] else "failed",
                phase="complete",
                stage=stage,
                passed=result["passed"],
                elapsed_seconds=result["elapsed_seconds"],
                automatic_followup=False,
            )
            return result
        raise ValueError(f"unknown rebuild stage: {stage}")
    except BaseException as error:
        _status(
            output,
            status="error",
            phase=f"{stage}-failed",
            stage=stage,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the v3 pilot with the frozen 94-feature training schema. "
            "This command reuses existing PP trials and never starts quick/formal."
        )
    )
    parser.add_argument("--stage", choices=("train", "native-audit", "verify"), required=True)
    parser.add_argument("--source", default="build/initlns-v3-pilot-v1")
    parser.add_argument("--output", default="build/initlns-v3-pilot-feature94-v1")
    parser.add_argument(
        "--reference-bundle",
        default="build/initlns-v3-pilot-v1/controller",
    )
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument("--benchmark-repeats", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    try:
        report = run_stage(
            stage=arguments.stage,
            source=resolve_cli_path(PROJECT_ROOT, arguments.source),
            output=resolve_cli_path(PROJECT_ROOT, arguments.output),
            reference_bundle=resolve_cli_path(PROJECT_ROOT, arguments.reference_bundle),
            main_controller_bundle=resolve_cli_path(
                PROJECT_ROOT, arguments.controller_bundle
            ),
            resume=arguments.resume,
            benchmark_repeats=arguments.benchmark_repeats,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
