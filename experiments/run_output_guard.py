from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


RUNNER_CONFIG_SCHEMA = "lns2.runner_output_identity.v1"


def _canonical_identity_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def runner_identity_fingerprint(identity: dict[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_identity_json(identity).encode("utf-8")
    ).hexdigest()


def prepare_run_output(
    output: str | Path,
    *,
    resume: bool,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Validate a runner output before any log or status file is opened.

    A non-empty output is immutable unless the caller explicitly asks to resume
    and its runner identity is an exact match.  On a valid resume this function
    performs no writes.
    """

    root = Path(output).resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"runner output is not a directory: {root}")

    fingerprint = runner_identity_fingerprint(identity)
    config_path = root / "runner_config.json"
    has_content = root.is_dir() and next(root.iterdir(), None) is not None
    if has_content:
        if not resume:
            raise ValueError(
                f"runner output is non-empty; pass --resume to continue: {root}"
            )
        if not config_path.is_file():
            raise ValueError(
                "runner output predates output-identity protection or is incomplete; "
                f"choose a new --output directory: {root}"
            )
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"runner identity cannot be read: {config_path}") from error
        if str(existing.get("schema")) != RUNNER_CONFIG_SCHEMA:
            raise ValueError(f"runner identity schema mismatch: {config_path}")
        if str(existing.get("identity_fingerprint")) != fingerprint:
            raise ValueError(
                "runner output belongs to a different mode, configuration, or "
                f"implementation fingerprint: {root}"
            )
        if existing.get("identity") != identity:
            raise ValueError(f"runner identity payload mismatch: {config_path}")
        return existing

    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": RUNNER_CONFIG_SCHEMA,
        "schema_version": 1,
        "identity_fingerprint": fingerprint,
        "identity": identity,
    }
    partial = root / "runner_config.json.partial"
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(config_path)
    return payload


__all__ = [
    "RUNNER_CONFIG_SCHEMA",
    "prepare_run_output",
    "runner_identity_fingerprint",
]
