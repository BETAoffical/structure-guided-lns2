from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from experiments._common import sha256_file


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    temporary.replace(destination)


def _archive_path(cache: Path, specification: dict[str, Any]) -> Path:
    name = str(specification["url"]).rsplit("/", 1)[-1]
    archive = cache / name
    expected = str(specification["sha256"]).lower()
    if archive.is_file() and sha256_file(archive) != expected:
        raise ValueError(f"cached archive checksum mismatch: {archive}")
    if not archive.is_file():
        _download(str(specification["url"]), archive)
        actual = sha256_file(archive)
        if actual != expected:
            archive.unlink(missing_ok=True)
            raise ValueError(
                f"download checksum mismatch for {name}: expected {expected}, got {actual}"
            )
    return archive


def _extract(archive: Path, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with zipfile.ZipFile(archive) as bundle:
        try:
            source = bundle.open(member)
        except KeyError as error:
            raise ValueError(f"archive {archive.name} has no member {member}") from error
        with source, temporary.open("wb") as stream:
            shutil.copyfileobj(source, stream)
    temporary.replace(destination)


def _scenario_member(member: str, scenario_index: int) -> str:
    replaced, count = re.subn(
        r"-random-\d+\.scen$", f"-random-{scenario_index}.scen", member
    )
    if count != 1:
        raise ValueError(
            "scenario archive member must end with '-random-N.scen' when "
            "multiple scenario indices are requested"
        )
    return replaced


def _scenario_index(member: str) -> int:
    match = re.search(r"-random-(\d+)\.scen$", member)
    if match is None:
        raise ValueError("scenario archive member must end with '-random-N.scen'")
    return int(match.group(1))


def fetch_devset(config: str | Path, output: str | Path) -> list[dict[str, Any]]:
    config_path = Path(config).resolve()
    output_root = Path(output).resolve()
    specification = json.loads(config_path.read_text(encoding="utf-8"))
    scenario_indices = [
        int(value) for value in specification.get("scenario_indices", [])
    ]
    if scenario_indices and (
        any(value <= 0 for value in scenario_indices)
        or len(scenario_indices) != len(set(scenario_indices))
    ):
        raise ValueError("scenario_indices must contain unique positive integers")
    cache = output_root / "_archives"
    manifest: list[dict[str, Any]] = []
    for benchmark in specification["benchmarks"]:
        map_archive = _archive_path(cache, benchmark["map_archive"])
        scenario_archive = _archive_path(cache, benchmark["scenario_archive"])
        map_member = str(benchmark["map_archive"]["member"])
        primary_scenario_member = str(benchmark["scenario_archive"]["member"])
        selected_indices = scenario_indices or [
            _scenario_index(primary_scenario_member)
        ]
        map_path = output_root / "maps" / Path(map_member).name
        _extract(map_archive, map_member, map_path)
        scenarios = []
        for scenario_index in selected_indices:
            scenario_member = _scenario_member(
                primary_scenario_member, scenario_index
            )
            scenario_path = output_root / "scenarios" / Path(scenario_member).name
            _extract(scenario_archive, scenario_member, scenario_path)
            scenarios.append(
                {
                    "index": scenario_index,
                    "file": scenario_path.relative_to(output_root).as_posix(),
                    "sha256": sha256_file(scenario_path),
                }
            )
        primary = scenarios[0]
        manifest.append(
            {
                "id": str(benchmark["id"]),
                "map_file": map_path.relative_to(output_root).as_posix(),
                "scenario_file": primary["file"],
                "agent_counts": [int(value) for value in benchmark["agent_counts"]],
                "map_sha256": sha256_file(map_path),
                "scenario_sha256": primary["sha256"],
                "scenarios": scenarios,
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in manifest:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    (output_root / "dataset_info.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": specification["source"],
                "config_sha256": config_digest,
                "benchmark_count": len(manifest),
                "scenario_indices": scenario_indices or None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest
