from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_VERSIONS = {
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "scikit-learn": "1.5.0",
    "joblib": "1.4.2",
    "threadpoolctl": "3.5.0",
}


def _check(
    rows: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    required: bool = True,
    expected: str | None = None,
    observed: str | None = None,
    detail: str | None = None,
) -> None:
    rows.append(
        {
            "name": name,
            "required": required,
            "passed": bool(passed),
            "expected": expected,
            "observed": observed,
            "detail": detail,
        }
    )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_wsl_checks(rows: list[dict[str, Any]]) -> None:
    release = platform.release().lower()
    is_wsl = platform.system() == "Linux" and (
        "microsoft" in release or bool(os.environ.get("WSL_DISTRO_NAME"))
    )
    _check(
        rows,
        "platform",
        is_wsl,
        expected="WSL2 Linux",
        observed=f"{platform.system()} {platform.release()}",
    )
    for package in ("numpy", "scipy"):
        version = _package_version(package)
        _check(
            rows,
            f"python-package:{package}",
            version is not None,
            expected="installed",
            observed=version,
        )

    native_build = PROJECT_ROOT / "build" / "linux" / "project"
    if native_build.is_dir() and str(native_build) not in sys.path:
        sys.path.insert(0, str(native_build))
    try:
        module = importlib.import_module("lns2_env")
    except (ImportError, OSError) as error:
        module = None
        native_error = f"{type(error).__name__}: {error}"
    else:
        native_error = None
    _check(
        rows,
        "native-module:lns2_env",
        module is not None,
        expected="importable from build/linux/project",
        observed=getattr(module, "__file__", None),
        detail=native_error,
    )
    for attribute in ("LNS2RepairEnv", "PortableTreeEnsemble", "batch_online_features"):
        _check(
            rows,
            f"lns2_env:{attribute}",
            module is not None and callable(getattr(module, attribute, None)),
            expected="callable",
            observed=(
                type(getattr(module, attribute, None)).__name__
                if module is not None and hasattr(module, attribute)
                else None
            ),
        )
    required_paths = {
        "controller-v2": PROJECT_ROOT
        / "artifacts"
        / "initlns-closed-loop-controller-v2"
        / "controller_manifest.json",
        "movingai-dataset": PROJECT_ROOT
        / "build"
        / "initlns-movingai-ood-dataset-v1"
        / "dataset_summary.json",
        "balanced-controller": PROJECT_ROOT
        / "build"
        / "initlns-lns2-speed-quality-calibration"
        / "balanced_controller.json",
    }
    for name, path in required_paths.items():
        _check(
            rows,
            f"runtime-file:{name}",
            path.is_file(),
            expected=str(path),
            observed="present" if path.is_file() else "missing",
        )


def _training_windows_checks(rows: list[dict[str, Any]]) -> None:
    _check(
        rows,
        "platform",
        platform.system() == "Windows",
        expected="Windows",
        observed=platform.system(),
    )
    for package, expected in TRAINING_VERSIONS.items():
        observed = _package_version(package)
        _check(
            rows,
            f"python-package:{package}",
            observed == expected,
            expected=expected,
            observed=observed,
            detail=(
                "Use an isolated environment; do not install into WSL runtime."
                if observed != expected
                else None
            ),
        )


def environment_report(profile: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    _check(
        rows,
        "python-version",
        sys.version_info >= (3, 10),
        expected=">=3.10",
        observed=platform.python_version(),
    )
    if profile == "runtime-wsl":
        _runtime_wsl_checks(rows)
    elif profile == "training-windows":
        _training_windows_checks(rows)
    else:
        raise ValueError(f"unsupported environment profile: {profile}")
    failures = [row for row in rows if row["required"] and not row["passed"]]
    return {
        "schema": "lns2.environment_check.v1",
        "schema_version": 1,
        "profile": profile,
        "project_root": str(PROJECT_ROOT),
        "passed": not failures,
        "required_failure_count": len(failures),
        "checks": rows,
        "installation_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the isolated LNS2 runtime or training environment."
    )
    parser.add_argument(
        "--profile",
        choices=("runtime-wsl", "training-windows"),
        required=True,
    )
    arguments = parser.parse_args()
    report = environment_report(arguments.profile)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
