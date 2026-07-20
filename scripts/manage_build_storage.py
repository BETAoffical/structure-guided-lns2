from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import zlib
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "build_storage_compaction.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "build" / "build-storage-management-20260720"
PLAN_SCHEMA = "lns2.build_storage_plan.v1"
MANIFEST_SCHEMA = "lns2.build_storage_manifest.v1"
TEXT_BLOCK_BYTES = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(TEXT_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _repository_path(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"repository path must be relative: {value}")
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ValueError(f"repository path escapes root: {value}")
    return path


def _repository_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _is_reparse_point(path: Path) -> bool:
    attributes = getattr(path.stat(), "st_file_attributes", 0)
    flag = getattr(os.stat_result, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if not flag:
        flag = getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return path.is_symlink() or bool(attributes & flag)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "lns2.build_storage_compaction.v1":
        raise ValueError("unsupported build storage config")
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("unsupported build storage config version")
    return config


def _allocated_bytes(path: Path) -> int:
    if os.name != "nt":
        blocks = getattr(path.stat(), "st_blocks", None)
        return int(blocks * 512) if blocks is not None else path.stat().st_size
    high = ctypes.c_ulong(0)
    low = ctypes.windll.kernel32.GetCompressedFileSizeW(str(path), ctypes.byref(high))
    if low == 0xFFFFFFFF and ctypes.windll.kernel32.GetLastError() != 0:
        raise ctypes.WinError()
    return int((high.value << 32) | low)


def _estimate_deflated_bytes(path: Path) -> int:
    compressor = zlib.compressobj(level=6, wbits=-15)
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(TEXT_BLOCK_BYTES), b""):
            size += len(compressor.compress(block))
    return size + len(compressor.flush())


def collect_target_files(root: Path, config: dict[str, Any]) -> list[Path]:
    resolved_root = root.resolve()
    build = (resolved_root / "build").resolve()
    extensions = {str(value).lower() for value in config["extensions"]}
    excluded = set(map(str, config.get("excluded_path_parts", [])))
    minimum_bytes = int(config["minimum_file_bytes"])
    files: set[Path] = set()
    for value in map(str, config["target_roots"]):
        target = _repository_path(resolved_root, value)
        try:
            target.relative_to(build)
        except ValueError as error:
            raise ValueError(f"storage target must be below build/: {value}") from error
        if not target.is_dir():
            raise FileNotFoundError(f"storage target does not exist: {value}")
        if _is_reparse_point(target):
            raise ValueError(f"storage target cannot be a reparse point: {value}")
        for path in target.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            relative = path.resolve().relative_to(resolved_root)
            if excluded.intersection(relative.parts):
                continue
            if _is_reparse_point(path):
                raise ValueError(f"storage file cannot be a reparse point: {relative}")
            if path.stat().st_size >= minimum_bytes:
                files.add(path.resolve())
    return sorted(files)


def build_plan(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    logical_total = 0
    allocated_total = 0
    estimated_total = 0
    for path in collect_target_files(root, config):
        logical = path.stat().st_size
        allocated = _allocated_bytes(path)
        estimated = _estimate_deflated_bytes(path)
        logical_total += logical
        allocated_total += allocated
        estimated_total += estimated
        rows.append(
            {
                "path": _repository_relative(root, path),
                "logical_bytes": logical,
                "allocated_bytes": allocated,
                "estimated_compressed_bytes": estimated,
            }
        )
    projected_savings = max(0, allocated_total - estimated_total)
    return {
        "schema": PLAN_SCHEMA,
        "config_sha256": _config_sha256(config),
        "file_count": len(rows),
        "logical_bytes": logical_total,
        "allocated_bytes": allocated_total,
        "estimated_compressed_bytes": estimated_total,
        "projected_savings_bytes": projected_savings,
        "minimum_projected_savings_bytes": int(
            config["minimum_projected_savings_bytes"]
        ),
        "compression_authorized": projected_savings
        >= int(config["minimum_projected_savings_bytes"]),
        "files": rows,
    }


def _config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def plan_is_current(root: Path, config: dict[str, Any], plan: dict[str, Any]) -> bool:
    if plan.get("schema") != PLAN_SCHEMA:
        return False
    if plan.get("config_sha256") != _config_sha256(config):
        return False
    for row in plan.get("files", []):
        path = _repository_path(root, str(row.get("path", "")))
        if not path.is_file():
            return False
        if path.stat().st_size != int(row.get("logical_bytes", -1)):
            return False
        if _allocated_bytes(path) != int(row.get("allocated_bytes", -1)):
            return False
    return True


def _chunks(values: list[Path], size: int = 40) -> Iterable[list[Path]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _compact(files: list[Path], *, compress: bool) -> None:
    if os.name != "nt":
        raise RuntimeError("NTFS transparent compression is only available on Windows")
    mode = "/C" if compress else "/U"
    extra = ["/EXE:LZX"] if compress else []
    for chunk in _chunks(files):
        result = subprocess.run(
            ["compact.exe", mode, "/I", "/Q", *extra, *(str(path) for path in chunk)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"compact.exe failed with {result.returncode}: "
                f"{result.stdout.strip()} {result.stderr.strip()}"
            )


def compress_plan(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported storage plan")
    if not plan.get("compression_authorized"):
        return {
            "schema": MANIFEST_SCHEMA,
            "status": "skipped_below_savings_threshold",
            "plan": plan,
            "files": [],
        }
    paths = [_repository_path(root, row["path"]) for row in plan["files"]]
    before = {path: _sha256(path) for path in paths}
    _compact(paths, compress=True)
    rows = []
    for path in paths:
        after_sha = _sha256(path)
        if after_sha != before[path]:
            raise RuntimeError(
                "content hash changed after compression: "
                f"{_repository_relative(root, path)}"
            )
        rows.append(
            {
                "path": _repository_relative(root, path),
                "sha256": after_sha,
                "logical_bytes": path.stat().st_size,
                "allocated_bytes": _allocated_bytes(path),
            }
        )
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "plan": plan,
        "file_count": len(rows),
        "logical_bytes": sum(row["logical_bytes"] for row in rows),
        "allocated_bytes": sum(row["allocated_bytes"] for row in rows),
        "files": rows,
    }


def verify_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported storage manifest")
    rows = []
    for item in manifest.get("files", []):
        path = _repository_path(root, str(item["path"]))
        exists = path.is_file()
        actual = _sha256(path) if exists else None
        rows.append(
            {
                "path": str(item["path"]),
                "exists": exists,
                "expected_sha256": str(item["sha256"]),
                "actual_sha256": actual,
                "matches": exists and actual == str(item["sha256"]),
            }
        )
    return {
        "schema": "lns2.build_storage_verification.v1",
        "file_count": len(rows),
        "verified_count": sum(row["matches"] for row in rows),
        "passed": bool(rows) and all(row["matches"] for row in rows),
        "files": rows,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _console_summary(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema",
        "status",
        "file_count",
        "verified_count",
        "logical_bytes",
        "allocated_bytes",
        "estimated_compressed_bytes",
        "projected_savings_bytes",
        "minimum_projected_savings_bytes",
        "compression_authorized",
        "passed",
    )
    return {key: value[key] for key in keys if key in value}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan, apply, or verify transparent compression of registered build text."
    )
    parser.add_argument("command", choices=("plan", "compress", "verify", "decompress"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    arguments = parser.parse_args()

    output = Path(arguments.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "compression_plan.json"
    manifest_path = output / "compression_manifest.json"
    if arguments.command == "plan":
        result = build_plan(PROJECT_ROOT, load_config(arguments.config))
        _atomic_json(plan_path, result)
    elif arguments.command == "compress":
        config = load_config(arguments.config)
        existing = _load_manifest(plan_path) if plan_path.is_file() else {}
        plan = existing if plan_is_current(PROJECT_ROOT, config, existing) else build_plan(
            PROJECT_ROOT, config
        )
        _atomic_json(plan_path, plan)
        result = compress_plan(PROJECT_ROOT, plan)
        _atomic_json(manifest_path, result)
    elif arguments.command == "verify":
        result = verify_manifest(PROJECT_ROOT, _load_manifest(manifest_path))
        _atomic_json(output / "verification.json", result)
        if not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 1
    else:
        manifest = _load_manifest(manifest_path)
        paths = [_repository_path(PROJECT_ROOT, row["path"]) for row in manifest["files"]]
        _compact(paths, compress=False)
        result = verify_manifest(PROJECT_ROOT, manifest)
        result["status"] = "decompressed" if result["passed"] else "hash_mismatch"
        _atomic_json(output / "decompression_report.json", result)
        if not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 1
    print(json.dumps(_console_summary(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
