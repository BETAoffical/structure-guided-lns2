from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import re
import stat
import subprocess
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "repository_hygiene.json"
SCHEMA = "lns2.repository_hygiene_report.v1"


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _repository_path(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"repository path must be relative: {value}")
    path = (root / relative).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError(f"repository path escapes root: {value}")
    return path


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if value.get("schema") != "lns2.repository_hygiene.v1":
        raise ValueError("unsupported repository hygiene config")
    if int(value.get("schema_version", -1)) != 1:
        raise ValueError("unsupported repository hygiene schema version")
    return value


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def tracked_files(root: Path) -> list[str]:
    return sorted(
        value
        for value in _git(
            root,
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ).split("\0")
        if value
    )


def duplicate_blob_groups(root: Path, files: Iterable[str]) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for relative in files:
        path = root / relative
        if path.is_file():
            groups[_sha256(path)].append(relative)
    return sorted(
        (sorted(paths) for paths in groups.values() if len(paths) > 1),
        key=lambda paths: paths[0],
    )


def _top_level_functions(path: Path) -> list[tuple[str, str, int, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rows = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        shape = ast.dump(
            ast.Module(body=[node], type_ignores=[]),
            annotate_fields=True,
            include_attributes=False,
        )
        digest = hashlib.sha256(shape.encode("utf-8")).hexdigest()
        rows.append(
            (
                digest,
                node.name,
                node.lineno,
                int(getattr(node, "end_lineno", node.lineno)),
            )
        )
    return rows


def duplicate_function_groups(
    root: Path, files: Iterable[str], production_roots: set[str]
) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relative in files:
        path = Path(relative)
        if path.suffix != ".py" or not path.parts or path.parts[0] not in production_roots:
            continue
        for digest, name, start, end in _top_level_functions(root / path):
            groups[digest].append(
                {"path": path.as_posix(), "name": name, "start": start, "end": end}
            )
    return sorted(
        (
            sorted(values, key=lambda value: (value["path"], value["start"]))
            for values in groups.values()
            if len(values) > 1
        ),
        key=lambda values: (values[0]["path"], values[0]["start"]),
    )


def _unused_imports(path: Path) -> list[dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[tuple[str, int]] = []
    exported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(
                (alias.asname or alias.name.split(".", 1)[0], node.lineno)
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            imported.extend(
                (alias.asname or alias.name, node.lineno)
                for alias in node.names
                if alias.name != "*"
            )
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            if isinstance(node.value, (ast.List, ast.Tuple)):
                exported.update(
                    str(item.value)
                    for item in node.value.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                )
    used = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Store)
    }
    return [
        {"name": name, "line": line}
        for name, line in imported
        if name != "annotations" and name not in used and name not in exported
    ]


def _module_references(root: Path, files: list[str]) -> dict[str, int]:
    text: dict[str, str] = {}
    for relative in files:
        path = root / relative
        if path.suffix.lower() not in {".py", ".md", ".json", ".txt"}:
            continue
        try:
            text[relative] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    references: dict[str, int] = {}
    for relative in files:
        path = Path(relative)
        is_active_experiment = path.parts[:1] == ("experiments",)
        is_research_module = (
            len(path.parts) >= 3
            and path.parts[0] == "research"
            and path.parts[1] in {"studies", "engineering"}
        )
        if not (is_active_experiment or is_research_module) or path.suffix != ".py":
            continue
        if path.name in {"__init__.py"}:
            references[relative] = 1
            continue
        module_name = ".".join(path.with_suffix("").parts)
        needles = (module_name, path.stem)
        references[relative] = sum(
            any(needle in contents for needle in needles)
            for other, contents in text.items()
            if other != relative
        )
    return references


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_status(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    evidence_config = _repository_path(
        root, str(config["result_consolidation_config"])
    )
    evidence = json.loads(evidence_config.read_text(encoding="utf-8"))
    entries = list(evidence.get("experiments", []))
    rows = []
    for entry in entries:
        source = dict(entry.get("source", {}))
        relative = str(source.get("path", ""))
        path = _repository_path(root, relative)
        expected = str(source.get("sha256", ""))
        exists = path.is_file()
        actual = _sha256(path) if exists else None
        rows.append(
            {
                "id": str(entry.get("id")),
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "exists": exists,
                "matches": exists and actual == expected,
            }
        )
    return {
        "entry_count": len(rows),
        "verified_count": sum(row["matches"] for row in rows),
        "missing_count": sum(not row["exists"] for row in rows),
        "mismatch_count": sum(row["exists"] and not row["matches"] for row in rows),
        "entries": rows,
    }


def run_check(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    files = tracked_files(root)
    roles = dict(config["tracked_roles"])
    root_files = set(map(str, config["tracked_root_files"]))
    production_roots = set(map(str, config["production_python_roots"]))
    forbidden_parts = set(map(str, config["forbidden_tracked_parts"]))
    forbidden_suffixes = set(map(str, config["forbidden_tracked_suffixes"]))
    maximum_size = int(config["maximum_tracked_file_bytes"])
    unowned = []
    forbidden = []
    oversized = []
    for relative in files:
        path = Path(relative)
        if len(path.parts) == 1:
            if relative not in root_files:
                unowned.append(relative)
        elif path.parts[0] not in roles:
            unowned.append(relative)
        if forbidden_parts.intersection(path.parts) or path.suffix in forbidden_suffixes:
            forbidden.append(relative)
        size = (root / path).stat().st_size
        if size > maximum_size:
            oversized.append({"path": path.as_posix(), "bytes": size})

    duplicate_blobs = duplicate_blob_groups(root, files)
    duplicate_functions = duplicate_function_groups(root, files, production_roots)
    unused_imports = []
    for relative in files:
        path = Path(relative)
        if path.suffix != ".py" or not path.parts or path.parts[0] not in production_roots:
            continue
        for item in _unused_imports(root / path):
            unused_imports.append({"path": path.as_posix(), **item})

    module_references = _module_references(root, files)
    orphan_modules = sorted(
        relative for relative, count in module_references.items() if count == 0
    )

    absolute_pattern = re.compile(r"(?<![A-Za-z])(?:[A-Za-z]:[\\/]|/(?:home|mnt)/)")
    absolute_paths = []
    scan_roots = set(map(str, config["absolute_path_scan_roots"]))
    for relative in files:
        path = Path(relative)
        if not path.parts or path.parts[0] not in scan_roots:
            continue
        if path.suffix.lower() not in {".json", ".md", ".txt", ".csv"}:
            continue
        contents = (root / path).read_text(encoding="utf-8")
        for line_number, line in enumerate(contents.splitlines(), start=1):
            if absolute_pattern.search(line):
                absolute_paths.append({"path": path.as_posix(), "line": line_number})

    evidence = evidence_status(root, config)
    errors = {
        "unowned_tracked_files": unowned,
        "forbidden_tracked_files": forbidden,
        "oversized_tracked_files": oversized,
        "duplicate_blob_groups": duplicate_blobs,
        "duplicate_function_groups": duplicate_functions,
        "unused_imports": unused_imports,
        "orphan_experiment_modules": orphan_modules,
        "absolute_machine_paths": absolute_paths,
        "evidence_hash_mismatches": [
            row["path"]
            for row in evidence["entries"]
            if row["exists"] and not row["matches"]
        ],
    }
    error_count = sum(len(values) for values in errors.values())
    return {
        "schema": SCHEMA,
        "repository": ".",
        "tracked_file_count": len(files),
        "tracked_role_counts": {
            role: sum(
                Path(relative).parts
                and Path(relative).parts[0] == top
                for relative in files
            )
            for top, role in roles.items()
        },
        "experiment_module_reference_counts": module_references,
        "evidence": evidence,
        "errors": errors,
        "error_count": error_count,
        "passed": error_count == 0,
    }


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _existing_build_reference(
    root: Path, source: Path, value: str
) -> Path | None:
    normalized = value.replace("\\", "/")
    candidates = []
    if normalized.startswith("build/"):
        candidates.append(root / normalized)
    raw = Path(value)
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend((root / raw, source.parent / raw))
    build = (root / "build").resolve()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(build)
        except (OSError, ValueError):
            continue
        if resolved.exists():
            return resolved
    return None


def _metadata_files(root: Path, build_root: Path, config: dict[str, Any]) -> list[Path]:
    names = set(map(str, config["build"]["dependency_metadata_names"]))
    limit = int(config["build"]["maximum_dependency_json_bytes"])
    files = []
    for path in build_root.rglob("*.json"):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        name = path.name.lower()
        report_like = any(
            marker in name
            for marker in ("report", "audit", "confirmation", "manifest", "registration")
        )
        if size <= limit and (path.name in names or report_like or path.parent == build_root):
            files.append(path)
    return sorted(set(files))


def protected_build_roots(
    root: Path, config: dict[str, Any], output_root: Path
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    build = (root / "build").resolve()
    reasons: dict[str, set[str]] = defaultdict(set)
    for name in map(str, config["build"]["fixed_protected_roots"]):
        reasons[name].add("fixed protection")
    output_relative = output_root.resolve().relative_to(build)
    if output_relative.parts:
        reasons[output_relative.parts[0]].add("repository hygiene records")

    evidence = evidence_status(root, config)
    queue: deque[Path] = deque()
    for row in evidence["entries"]:
        if not row["exists"]:
            continue
        path = _repository_path(root, row["path"])
        relative = path.resolve().relative_to(build)
        reasons[relative.parts[0]].add(f"formal evidence: {row['id']}")
        queue.append(path)

    scanned_files: set[Path] = set()
    scanned_roots: set[str] = set()
    while queue:
        source = queue.popleft().resolve()
        if source in scanned_files or not source.is_file() or source.suffix.lower() != ".json":
            continue
        scanned_files.add(source)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for text in _iter_strings(value):
            referenced = _existing_build_reference(root, source, text)
            if referenced is None:
                continue
            relative = referenced.relative_to(build)
            if not relative.parts:
                continue
            name = relative.parts[0]
            reasons[name].add(f"referenced by {_relative(root, source)}")
            if referenced.is_file() and referenced.suffix.lower() == ".json":
                queue.append(referenced)
            if name not in scanned_roots:
                scanned_roots.add(name)
                candidate_root = build / name
                if candidate_root.is_dir():
                    queue.extend(_metadata_files(root, candidate_root, config))
    return reasons, evidence


def _directory_inventory(root: Path, path: Path) -> dict[str, Any]:
    file_count = 0
    byte_count = 0
    root_stat = path.stat()
    latest = root_stat.st_mtime
    reparse_point = path.is_symlink() or bool(
        getattr(root_stat, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )
    iterator = () if reparse_point else os.walk(path, followlinks=False)
    for current, directories, files in iterator:
        directories[:] = [
            name for name in directories if not (Path(current) / name).is_symlink()
        ]
        for name in files:
            candidate = Path(current) / name
            try:
                file_stat = candidate.stat()
            except OSError:
                continue
            file_count += 1
            byte_count += file_stat.st_size
            latest = max(latest, file_stat.st_mtime)
    return {
        "path": _relative(root, path),
        "file_count": file_count,
        "bytes": byte_count,
        "modified_utc": dt.datetime.fromtimestamp(
            latest, tz=dt.timezone.utc
        ).isoformat(),
        "reparse_point": reparse_point,
    }


def _matches_temporary(name: str, config: dict[str, Any]) -> str | None:
    for entry in config["build"].get("safe_delete_roots", []):
        if str(entry.get("name")) == name:
            return str(entry.get("reason") or "explicit safe-delete root")
    if name in set(map(str, config["build"]["temporary_exact_roots"])):
        return "explicit temporary or superseded root"
    for pattern in map(str, config["build"]["temporary_name_patterns"]):
        if re.search(pattern, name, flags=re.IGNORECASE):
            return f"temporary name pattern: {pattern}"
    return None


def _cleanup_path_inventory(
    root: Path,
    config: dict[str, Any],
    entry: dict[str, Any],
    *,
    conditional: bool,
) -> dict[str, Any] | None:
    relative = str(entry.get("path", ""))
    path = _repository_path(root, relative)
    build = (root / "build").resolve()
    try:
        path.relative_to(build)
    except ValueError as error:
        raise ValueError(f"cleanup path must be below build/: {relative}") from error
    if path == build:
        raise ValueError("cleanup path cannot be the build root")
    if not path.exists():
        return None
    if not path.is_dir():
        raise ValueError(f"cleanup path must be a directory: {relative}")
    row = {
        **_directory_inventory(root, path),
        "reason": str(entry.get("reason") or "explicit cleanup path"),
    }
    if row["reparse_point"]:
        return {**row, "blocked": True, "evidence_preconditions_passed": False}
    if not conditional:
        return row

    expected_bytes = entry.get("expected_bytes")
    checks: list[dict[str, Any]] = []
    if expected_bytes is not None:
        checks.append(
            {
                "check": "expected_bytes",
                "expected": int(expected_bytes),
                "actual": int(row["bytes"]),
                "passed": int(row["bytes"]) == int(expected_bytes),
            }
        )
    verification_specs = list(entry.get("verification_json_checks", []))
    if not verification_specs:
        verification_specs = [
            {
                "path": entry.get("verification_json", ""),
                "required_values": entry.get("required_values", {}),
            }
        ]
    verification_paths = []
    for spec in verification_specs:
        verification_relative = str(spec.get("path", ""))
        verification_paths.append(verification_relative)
        verification_path = _repository_path(root, verification_relative)
        verification: dict[str, Any] = {}
        if verification_path.is_file():
            try:
                verification = json.loads(
                    verification_path.read_text(encoding="utf-8")
                )
                checks.append(
                    {
                        "check": "verification_json",
                        "path": verification_relative,
                        "passed": True,
                    }
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                checks.append(
                    {
                        "check": "verification_json",
                        "path": verification_relative,
                        "passed": False,
                        "error": str(error),
                    }
                )
        else:
            checks.append(
                {
                    "check": "verification_json",
                    "path": verification_relative,
                    "passed": False,
                    "error": "missing",
                }
            )
        for field, expected in dict(spec.get("required_values", {})).items():
            actual: Any = verification
            for part in str(field).split("."):
                actual = actual.get(part) if isinstance(actual, dict) else None
            checks.append(
                {
                    "check": f"required_value:{field}",
                    "path": verification_relative,
                    "expected": expected,
                    "actual": actual,
                    "passed": actual == expected,
                }
            )
    return {
        **row,
        "verification_json_checks": verification_paths,
        "checks": checks,
        "blocked": False,
        "evidence_preconditions_passed": bool(checks)
        and all(bool(check["passed"]) for check in checks),
        "requires_explicit_user_approval": True,
    }


def build_cleanup_plan(
    root: Path, config: dict[str, Any], output_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    build = (root / "build").resolve()
    output_root = output_root.resolve()
    try:
        output_root.relative_to(build)
    except ValueError as error:
        raise ValueError("hygiene output must be inside repository build/") from error
    output_root.mkdir(parents=True, exist_ok=True)

    protected, evidence = protected_build_roots(root, config, output_root)
    if evidence["missing_count"] or evidence["mismatch_count"]:
        raise RuntimeError("formal evidence is missing or has a SHA256 mismatch")

    inventories = [
        _directory_inventory(root, path)
        for path in sorted(build.iterdir(), key=lambda value: value.name.lower())
        if path.is_dir()
    ]
    delete_roots = []
    protected_roots = []
    retained_roots = []
    blocked_roots = []
    delete_names: set[str] = set()
    for row in inventories:
        name = Path(row["path"]).name
        if name in protected:
            protected_roots.append(
                {**row, "reasons": sorted(protected[name])}
            )
            continue
        reason = _matches_temporary(name, config)
        if reason is None:
            retained_roots.append(
                {**row, "reason": "not proven temporary; retained conservatively"}
            )
        elif row["reparse_point"]:
            blocked_roots.append(
                {**row, "reason": f"{reason}; reparse point requires manual review"}
            )
        else:
            delete_names.add(name)
            delete_roots.append({**row, "reason": reason})

    safe_delete_paths = []
    conditional_delete_paths = []
    blocked_paths = []
    for entry in config["build"].get("safe_delete_paths", []):
        row = _cleanup_path_inventory(root, config, dict(entry), conditional=False)
        if row is None:
            continue
        if row.get("blocked"):
            blocked_paths.append(row)
        else:
            safe_delete_paths.append(row)
    for entry in config["build"].get("conditional_delete_paths", []):
        row = _cleanup_path_inventory(root, config, dict(entry), conditional=True)
        if row is None:
            continue
        if row.get("blocked") or not row.get("evidence_preconditions_passed"):
            blocked_paths.append(row)
        else:
            conditional_delete_paths.append(row)

    cache_names = set(map(str, config["build"]["cache_directory_names"]))
    cache_directories = []
    for path in root.rglob("*"):
        if not path.is_dir() or path.name not in cache_names:
            continue
        relative = path.resolve().relative_to(root)
        if relative.parts[:2] == ("build", "venv-graph"):
            continue
        if relative.parts[:1] == ("build",) and len(relative.parts) > 1:
            if relative.parts[1] in delete_names:
                continue
        if path.is_symlink():
            continue
        cache_directories.append(_relative(root, path))

    suffixes = tuple(map(str, config["build"]["incomplete_file_suffixes"]))
    incomplete_files = []
    for path in build.rglob("*"):
        if not path.is_file() or not path.name.endswith(suffixes):
            continue
        relative = path.resolve().relative_to(build)
        if relative.parts and relative.parts[0] == "venv-graph":
            continue
        if relative.parts and relative.parts[0] in delete_names:
            continue
        incomplete_files.append(_relative(root, path))

    inventory = {
        "schema": SCHEMA,
        "repository": ".",
        "build_root": "build",
        "directory_count": len(inventories),
        "file_count": sum(row["file_count"] for row in inventories),
        "bytes": sum(row["bytes"] for row in inventories),
        "directories": inventories,
    }
    plan = {
        "schema": SCHEMA,
        "repository": ".",
        "build_root": "build",
        "evidence": evidence,
        "protected_roots": protected_roots,
        "delete_roots": delete_roots,
        "blocked_roots": blocked_roots,
        "retained_roots": retained_roots,
        "safe_delete_paths": safe_delete_paths,
        "conditional_delete_paths": conditional_delete_paths,
        "blocked_paths": blocked_paths,
        "cache_directories": sorted(cache_directories),
        "incomplete_files": sorted(incomplete_files),
        "summary": {
            "protected_root_count": len(protected_roots),
            "delete_root_count": len(delete_roots),
            "delete_root_bytes": sum(row["bytes"] for row in delete_roots),
            "blocked_root_count": len(blocked_roots),
            "safe_delete_path_count": len(safe_delete_paths),
            "safe_delete_path_bytes": sum(row["bytes"] for row in safe_delete_paths),
            "conditional_delete_path_count": len(conditional_delete_paths),
            "conditional_delete_path_bytes": sum(
                row["bytes"] for row in conditional_delete_paths
            ),
            "blocked_path_count": len(blocked_paths),
            "retained_root_count": len(retained_roots),
            "cache_directory_count": len(cache_directories),
            "incomplete_file_count": len(incomplete_files),
        },
        "deletion_supported": False,
    }
    return inventory, plan


def post_cleanup_report(
    root: Path, config: dict[str, Any], output_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    build = (root / "build").resolve()
    output_root = output_root.resolve()
    try:
        output_root.relative_to(build)
    except ValueError as error:
        raise ValueError("hygiene output must be inside repository build/") from error
    plan_path = output_root / "cleanup_plan.json"
    inventory_path = output_root / "pre_cleanup_inventory.json"
    if not plan_path.is_file() or not inventory_path.is_file():
        raise ValueError("post-cleanup verification requires the original plan and inventory")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    before = json.loads(inventory_path.read_text(encoding="utf-8"))
    current = [
        _directory_inventory(root, path)
        for path in sorted(build.iterdir(), key=lambda value: value.name.lower())
        if path.is_dir()
    ]
    current_paths = {row["path"] for row in current}
    expected_deleted = [row["path"] for row in plan["delete_roots"]]
    expected_deleted_paths = [
        row["path"]
        for key in ("safe_delete_paths", "conditional_delete_paths")
        for row in plan.get(key, [])
    ]
    expected_protected = [row["path"] for row in plan["protected_roots"]]
    remaining_delete_roots = sorted(set(expected_deleted) & current_paths)
    missing_protected_roots = sorted(set(expected_protected) - current_paths)
    remaining_delete_paths = sorted(
        relative
        for relative in expected_deleted_paths
        if _repository_path(root, relative).exists()
    )

    cache_names = set(map(str, config["build"]["cache_directory_names"]))
    remaining_caches = []
    for path in root.rglob("*"):
        if not path.is_dir() or path.name not in cache_names:
            continue
        relative = path.resolve().relative_to(root)
        if relative.parts[:2] != ("build", "venv-graph"):
            remaining_caches.append(relative.as_posix())
    evidence = evidence_status(root, config)
    after_bytes = sum(row["bytes"] for row in current)
    report = {
        "schema": SCHEMA,
        "repository": ".",
        "expected_deleted_roots": expected_deleted,
        "removed_roots": sorted(set(expected_deleted) - current_paths),
        "remaining_delete_roots": remaining_delete_roots,
        "expected_deleted_paths": expected_deleted_paths,
        "removed_paths": sorted(set(expected_deleted_paths) - set(remaining_delete_paths)),
        "remaining_delete_paths": remaining_delete_paths,
        "missing_protected_roots": missing_protected_roots,
        "remaining_cache_directories": sorted(remaining_caches),
        "before_bytes": int(before["bytes"]),
        "after_bytes": after_bytes,
        "freed_bytes": int(before["bytes"]) - after_bytes,
        "evidence": evidence,
        "passed": (
            not remaining_delete_roots
            and not remaining_delete_paths
            and not missing_protected_roots
            and not remaining_caches
            and evidence["missing_count"] == 0
            and evidence["mismatch_count"] == 0
        ),
    }
    inventory = {
        "schema": SCHEMA,
        "repository": ".",
        "build_root": "build",
        "directory_count": len(current),
        "file_count": sum(row["file_count"] for row in current),
        "bytes": after_bytes,
        "directories": current,
    }
    return inventory, report


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit tracked repository hygiene and emit a read-only build cleanup plan."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--emit-build-plan")
    parser.add_argument("--record-post-cleanup")
    arguments = parser.parse_args()
    modes = sum(
        bool(value)
        for value in (
            arguments.check,
            arguments.emit_build_plan,
            arguments.record_post_cleanup,
        )
    )
    if modes != 1:
        parser.error("choose exactly one audit mode")

    config = load_config(arguments.config)
    check = run_check(PROJECT_ROOT, config)
    if arguments.emit_build_plan:
        output = Path(arguments.emit_build_plan)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        inventory, plan = build_cleanup_plan(PROJECT_ROOT, config, output)
        _write_json(output / "repository_check.json", check)
        _write_json(output / "pre_cleanup_inventory.json", inventory)
        _write_json(output / "cleanup_plan.json", plan)
        print(
            json.dumps(
                {
                    "passed": check["passed"],
                    "output": _relative(PROJECT_ROOT, output),
                    **plan["summary"],
                },
                sort_keys=True,
            )
        )
    elif arguments.record_post_cleanup:
        output = Path(arguments.record_post_cleanup)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        inventory, report = post_cleanup_report(PROJECT_ROOT, config, output)
        _write_json(output / "post_cleanup_inventory.json", inventory)
        _write_json(output / "post_cleanup_verification.json", report)
        print(
            json.dumps(
                {
                    "passed": report["passed"],
                    "removed_root_count": len(report["removed_roots"]),
                    "freed_bytes": report["freed_bytes"],
                    "evidence_verified": report["evidence"]["verified_count"],
                },
                sort_keys=True,
            )
        )
        if not report["passed"]:
            return 1
    else:
        print(
            json.dumps(
                {
                    "passed": check["passed"],
                    "tracked_file_count": check["tracked_file_count"],
                    "evidence_verified": check["evidence"]["verified_count"],
                    "evidence_entries": check["evidence"]["entry_count"],
                    "error_count": check["error_count"],
                },
                sort_keys=True,
            )
        )
    return 0 if check["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
