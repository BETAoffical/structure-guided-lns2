from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_hierarchical_ch_compact_flow_labels_readiness_v1 import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    build_compact_flow_labels_readiness,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build integrity-checked compact-flow H1 labels and sequential "
            "readiness diagnostics; never fit a model."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    config = (PROJECT_ROOT / arguments.config).resolve()
    output = (PROJECT_ROOT / arguments.output).resolve()
    result = build_compact_flow_labels_readiness(
        config,
        output=output,
        workers=arguments.workers,
        dry_run=arguments.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
