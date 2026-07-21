from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments._common import resolve_cli_path  # noqa: E402
from experiments.state_conditioned_rescue_audit import (  # noqa: E402
    audit_state_conditioned_rescue,
)


DEFAULT_SOURCES = (
    "locked_v1_balanced=build/initlns-rescue-lite-balanced-diagnostic-v1",
    "locked_v2=build/initlns-rescue-lite-locked-confirmation-v2",
)


def _sources(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError("--source must use a unique NAME=PATH value")
        result[name] = resolve_cli_path(PROJECT_ROOT, path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a shallow state-conditioned rescue selector across two "
            "completed confirmation sets without running the solver."
        )
    )
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument(
        "--output",
        default="build/initlns-state-conditioned-rescue-audit-v1",
    )
    arguments = parser.parse_args()
    report = audit_state_conditioned_rescue(
        project_root=PROJECT_ROOT,
        sources=_sources(arguments.source or list(DEFAULT_SOURCES)),
        output=resolve_cli_path(PROJECT_ROOT, arguments.output),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
