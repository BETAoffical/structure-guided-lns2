"""Shared implementation details for experiment and audit modules.

This module intentionally contains only semantics-free helpers. Study-specific
Pareto definitions, bootstrap procedures, labels, and acceptance gates remain
in their owning experiment modules so historical results stay reproducible.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


PRODUCER_IDENTITY_SCHEMA = "lns2.producer_identity.v2"
NATIVE_SEMANTICS_SCHEMA = "lns2.native_semantics.official_step_timed_extension.v3"
CLOSED_LOOP_IMPLEMENTATION_FILES = (
    "CMakeLists.txt",
    "experiments/_common.py",
    "experiments/closed_loop_confirmation.py",
    "experiments/closed_loop_confirmation_analysis.py",
    "experiments/closed_loop_trace_storage.py",
    "experiments/compact_controller_model.py",
    "experiments/context_audit.py",
    "experiments/feature_schema_v2.py",
    "experiments/neighborhood_candidates.py",
    "experiments/neighborhood_features.py",
    "experiments/online_feature_engine.py",
    "experiments/repair_collection.py",
    "experiments/run_output_guard.py",
    "experiments/state_analysis.py",
    "experiments/v3_s3.py",
    "lns2_selector/compatibility/controller_diagnostics.py",
    "lns2_selector/compatibility/metrics.py",
    "lns2_selector/controllers/__init__.py",
    "lns2_selector/controllers/official.py",
    "lns2_selector/controllers/v2.py",
    "lns2_selector/controllers/v3_s3.py",
    "lns2_selector/evaluation/trace_validation.py",
    "lns2_selector/runtime/artifact_validation.py",
    "lns2_selector/runtime/causalclosurepool.py",
    "lns2_selector/runtime/contracts.py",
    "lns2_selector/runtime/fingerprints.py",
    "lns2_selector/runtime/hybridstructpool.py",
    "lns2_selector/runtime/hybridstructpool_routed.py",
    "lns2_selector/runtime/metrics.py",
    "lns2_selector/runtime/online_selection.py",
    "lns2_selector/runtime/portable_scalar.py",
    "lns2_selector/runtime/repair_outcomes.py",
    "lns2_selector/runtime/structshell_dual16.py",
    "lns2_selector/runtime/topology_candidates.py",
    "lns2_selector/solver/native.py",
    "lns2_selector/training/policy_bundle.py",
    "src/jsonl_observer.cpp",
    "src/online_features.cpp",
    "src/online_features.h",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/BasicLNS.h",
    "third_party/mapf_lns2/inc/InitLNS.h",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/inc/SIPP.h",
    "third_party/mapf_lns2/inc/SingleAgentSolver.h",
    "third_party/mapf_lns2/inc/SpaceTimeAStar.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
    "third_party/mapf_lns2/src/SIPP.cpp",
    "third_party/mapf_lns2/src/SpaceTimeAStar.cpp",
)


def _native_filesystem_path(path: Path) -> Path:
    """Return a Windows extended path when the absolute path is long.

    The experiment outputs deliberately use descriptive episode names.  A
    valid contained trace can therefore exceed the legacy Win32 MAX_PATH even
    though the same file is readable from WSL.  Keep all containment checks on
    the ordinary resolved path, then use the extended prefix only for native
    filesystem operations.
    """

    if platform.system() != "Windows" or not path.is_absolute():
        return path
    text = str(path)
    if text.startswith("\\\\?\\") or len(text) < 248:
        return path
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text.lstrip("\\"))
    return Path("\\\\?\\" + text)


def strict_nonnegative_int(value: Any) -> bool:
    """Return whether *value* is a non-boolean, non-negative integer."""

    return type(value) is int and value >= 0


def strict_bool(value: Any, *, field: str) -> bool:
    """Return a JSON boolean while rejecting truthy substitutes."""

    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean")
    return value


def strict_int(
    value: Any, *, field: str, minimum: int | None = 0
) -> int:
    """Return a JSON integer while rejecting booleans and numeric coercion."""

    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def mean(values: Iterable[float | int | bool]) -> float:
    numbers = [float(value) for value in values]
    return statistics.fmean(numbers) if numbers else 0.0


def population_std(values: Iterable[float | int | bool]) -> float:
    numbers = [float(value) for value in values]
    return statistics.pstdev(numbers) if len(numbers) > 1 else 0.0


def ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def quantile(values: Iterable[float | int], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def relative_improvement(baseline: float, challenger: float) -> float:
    if baseline == 0.0:
        return 0.0 if challenger == 0.0 else -float("inf")
    return (baseline - challenger) / baseline


def standard_error(values: Iterable[float | int]) -> float:
    numbers = [float(value) for value in values]
    if len(numbers) < 2:
        return 0.0
    return statistics.stdev(numbers) / math.sqrt(len(numbers))


def state_groups(rows: list[dict[str, Any]]) -> list[list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["state_id"]), []).append(index)
    return [grouped[key] for key in sorted(grouped)]


def contained_file(root: Path, value: Any, *, field: str) -> Path:
    """Resolve a required relative file without following child symlinks."""

    root = root.resolve()
    text = str(value or "")
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must be a contained relative path")
    path = root
    for part in relative.parts:
        path /= part
        if _native_filesystem_path(path).is_symlink():
            raise ValueError(f"{field} must not traverse a symbolic link")
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes its collection root: {value}") from error
    filesystem_path = _native_filesystem_path(resolved)
    if not filesystem_path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {value}")
    return filesystem_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_input(
    project_root: Path,
    specification: dict[str, Any],
    *,
    label: str,
) -> Path:
    """Resolve a checksum-pinned input while enforcing repository containment."""

    if not isinstance(specification, dict):
        raise ValueError(f"registered {label} input specification is not an object")
    expected = specification.get("sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"registered {label} input has an invalid SHA-256")
    path = contained_file(
        Path(project_root).resolve(),
        specification.get("path"),
        field=f"registered {label} input",
    )
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"registered {label} input changed: {path}: "
            f"expected {expected}, got {observed}"
        )
    return path


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def canonical_json(value: Any) -> str:
    """Serialize strict, deterministic JSON suitable for fingerprints."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def producer_identity(
    *,
    project_root: Path,
    source_files: Iterable[str | Path],
    native_required: bool,
    package_names: Iterable[str] = (),
    optional_package_names: Iterable[str] = (),
) -> dict[str, Any]:
    """Describe every executable dependency that can change an artifact.

    Source keys are project-relative so identities remain readable and stable.
    When native execution is required, an unavailable or unversioned extension
    is a hard error rather than an identity with missing evidence.
    """

    root = Path(project_root).resolve()
    source_sha256: dict[str, str] = {}
    for value in source_files:
        candidate = Path(value)
        path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"producer source escapes project root: {value}") from error
        if relative in source_sha256:
            raise ValueError(f"duplicate producer source: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"producer source is missing: {relative}")
        source_sha256[relative] = sha256_file(path)
    if not source_sha256:
        raise ValueError("producer identity requires at least one source file")

    required_packages = tuple(sorted(set(map(str, package_names))))
    optional_packages = tuple(sorted(set(map(str, optional_package_names))))
    overlap = set(required_packages) & set(optional_packages)
    if overlap:
        raise ValueError(
            "producer packages cannot be both required and optional: "
            + ", ".join(sorted(overlap))
        )
    requested_packages = tuple(sorted((*required_packages, *optional_packages)))
    packages: dict[str, str | None] = {}
    for name in requested_packages:
        if not name:
            raise ValueError("producer package name must not be empty")
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            import_name = {
                "scikit-learn": "sklearn",
            }.get(name, name.replace("-", "_"))
            try:
                package = importlib.import_module(import_name)
            except ImportError:
                packages[name] = None
            else:
                raw_version = getattr(package, "__version__", None)
                packages[name] = (
                    str(raw_version) if raw_version is not None else None
                )

    native: dict[str, str] | None = None
    try:
        module = importlib.import_module("lns2_env")
    except ImportError as error:
        if native_required:
            raise RuntimeError("required native module lns2_env is unavailable") from error
    else:
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            raise RuntimeError("loaded lns2_env has no binary path")
        native_path = Path(str(raw_path)).resolve()
        if not native_path.is_file():
            raise RuntimeError(f"loaded lns2_env binary is missing: {native_path}")
        timing_schema = str(getattr(module, "repair_timing_schema", ""))
        if not timing_schema:
            raise RuntimeError("loaded lns2_env has no repair_timing_schema")
        semantics_schema = str(getattr(module, "native_semantics_schema", ""))
        if not semantics_schema:
            raise RuntimeError("loaded lns2_env has no native_semantics_schema")
        if semantics_schema != NATIVE_SEMANTICS_SCHEMA:
            raise RuntimeError(
                "loaded lns2_env has unsupported native semantics schema: "
                f"{semantics_schema}"
            )
        native = {
            "path": str(native_path),
            "sha256": sha256_file(native_path),
            "repair_timing_schema": timing_schema,
            "native_semantics_schema": semantics_schema,
        }

    result = {
        "schema": PRODUCER_IDENTITY_SCHEMA,
        "source_sha256": dict(sorted(source_sha256.items())),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": packages,
        "native_required": bool(native_required),
        "native": native,
    }
    validate_producer_identity(
        result,
        native_required=bool(native_required),
        package_names=required_packages,
        optional_package_names=optional_packages,
    )
    return result


def closed_loop_producer_identity(
    *,
    project_root: Path,
    source_files: Iterable[str | Path],
    native_required: bool = True,
) -> dict[str, Any]:
    """Build the shared identity for a native closed-loop experiment runner."""

    return producer_identity(
        project_root=project_root,
        source_files=tuple(
            dict.fromkeys(
                (*map(str, source_files), *CLOSED_LOOP_IMPLEMENTATION_FILES)
            )
        ),
        native_required=native_required,
        optional_package_names=("numpy", "scikit-learn"),
    )


def validate_producer_identity(
    value: Any,
    *,
    native_required: bool,
    package_names: Iterable[str] = (),
    optional_package_names: Iterable[str] = (),
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("producer identity is not an object")
    identity = dict(value)
    if str(identity.get("schema")) != PRODUCER_IDENTITY_SCHEMA:
        raise ValueError("producer identity schema mismatch")
    sources = identity.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("producer identity has no source hashes")
    hexadecimal = set("0123456789abcdef")
    for name, digest in sources.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or len(digest) != 64
            or not set(digest) <= hexadecimal
        ):
            raise ValueError("producer identity contains an invalid source hash")
    python = identity.get("python")
    if (
        not isinstance(python, dict)
        or not isinstance(python.get("implementation"), str)
        or not python["implementation"]
        or not isinstance(python.get("version"), str)
        or not python["version"]
    ):
        raise ValueError("producer identity has invalid Python runtime evidence")
    packages = identity.get("packages")
    if not isinstance(packages, dict):
        raise ValueError("producer identity package versions are missing")
    if any(
        not isinstance(name, str)
        or not name
        or (version is not None and not isinstance(version, str))
        or version == ""
        for name, version in packages.items()
    ):
        raise ValueError("producer identity contains invalid package evidence")
    required_packages = set(map(str, package_names))
    optional_packages = set(map(str, optional_package_names))
    if required_packages & optional_packages:
        raise ValueError("producer package requirement sets overlap")
    for name in required_packages:
        if name not in packages or not str(packages[name] or ""):
            raise ValueError(f"producer identity lacks package version: {name}")
    for name in optional_packages:
        if name not in packages or (
            packages[name] is not None and not str(packages[name])
        ):
            raise ValueError(f"producer identity lacks optional package state: {name}")
    if identity.get("native_required") is not bool(native_required):
        raise ValueError("producer identity native-required marker mismatch")
    native = identity.get("native")
    if native_required and not isinstance(native, dict):
        raise ValueError("producer identity lacks required native evidence")
    if native is not None:
        if not isinstance(native, dict):
            raise ValueError("producer identity native evidence is not an object")
        if not isinstance(native.get("path"), str) or not native["path"]:
            raise ValueError("producer identity native path is missing")
        digest = native.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not set(digest) <= hexadecimal
        ):
            raise ValueError("producer identity native hash is invalid")
        if (
            not isinstance(native.get("repair_timing_schema"), str)
            or not native["repair_timing_schema"]
        ):
            raise ValueError("producer identity native timing schema is missing")
        if native.get("native_semantics_schema") != NATIVE_SEMANTICS_SCHEMA:
            raise ValueError(
                "producer identity native semantics schema is missing or unsupported"
            )
    return identity


def config_producer_fingerprint(
    config: dict[str, Any],
    *,
    label: str,
    native_required: bool = True,
    package_names: Iterable[str] = (),
    optional_package_names: Iterable[str] = ("numpy", "scikit-learn"),
) -> str:
    identity = validate_producer_identity(
        config.get("producer_identity"),
        native_required=native_required,
        package_names=package_names,
        optional_package_names=optional_package_names,
    )
    expected = hashlib.sha256(
        canonical_json(identity).encode("utf-8")
    ).hexdigest()
    fingerprint = str(config.get("producer_identity_fingerprint", ""))
    if not fingerprint or fingerprint != expected:
        raise ValueError(f"{label} producer identity fingerprint mismatch")
    return fingerprint


def read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )


def read_collection_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing collection file: {path}")
    return [
        json.loads(line, parse_constant=_reject_json_constant)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line, parse_constant=_reject_json_constant)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [
            json.loads(line, parse_constant=_reject_json_constant)
            for line in stream
            if line.strip()
        ]


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a text artifact atomically, tolerating short DrvFS read locks."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        for attempt in range(8):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(0.025 * (2**attempt), 0.5))
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
    )


def atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write an empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({name for row in materialized for name in row})
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(materialized)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resolve_within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"collection path escapes its root: {relative}") from error
    return path


def resolve_cli_path(project_root: Path, value: str | Path) -> Path:
    """Resolve a CLI path without requiring the target to remain inside the repo."""

    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def add_categorical_feature(
    features: dict[str, float], prefix: str, value: Any
) -> None:
    normalized = str(value or "unknown").strip().lower().replace(" ", "_")
    features[f"{prefix}={normalized}"] = 1.0


def feature_names(rows: Iterable[dict[str, Any]], profile: str) -> list[str]:
    return sorted(
        {
            name
            for row in rows
            for name in row["features"][profile]
        }
    )


def state_storage_id(state_id: str) -> str:
    payload = json.dumps(
        {"state_id": state_id},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"state-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def trial_job_id(state_id: str, candidate_id: str, trial_index: int) -> str:
    return (
        f"{state_storage_id(state_id)}__{candidate_id}"
        f"__trial_{trial_index:04d}"
    )


def episode_id(row: dict[str, Any], solver_seed: int, policy: str) -> str:
    return f"{row['task_id']}__seed_{solver_seed:04d}__{policy}"


def select_rows_by_task_id(
    rows: list[dict[str, Any]], task_ids: list[str] | None
) -> list[dict[str, Any]]:
    if task_ids is None:
        return rows
    requested = list(dict.fromkeys(map(str, task_ids)))
    indexed = {str(row["task_id"]): row for row in rows}
    missing = sorted(set(requested) - set(indexed))
    if missing:
        raise ValueError(f"unknown task ids: {missing}")
    return [indexed[task_id] for task_id in requested]


__all__ = [
    "CLOSED_LOOP_IMPLEMENTATION_FILES",
    "NATIVE_SEMANTICS_SCHEMA",
    "PRODUCER_IDENTITY_SCHEMA",
    "add_categorical_feature",
    "atomic_write_text",
    "atomic_write_csv",
    "canonical_json",
    "closed_loop_producer_identity",
    "contained_file",
    "config_producer_fingerprint",
    "episode_id",
    "feature_names",
    "json_fingerprint",
    "mean",
    "population_std",
    "producer_identity",
    "quantile",
    "ratio",
    "read_collection_jsonl",
    "read_json",
    "read_jsonl",
    "read_optional_jsonl",
    "registered_input",
    "relative_improvement",
    "resolve_within",
    "sha256_file",
    "select_rows_by_task_id",
    "state_storage_id",
    "state_groups",
    "standard_error",
    "strict_bool",
    "strict_int",
    "strict_nonnegative_int",
    "trial_job_id",
    "validate_producer_identity",
    "write_json",
    "write_jsonl",
]
