from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_hierarchical_ch_compact_flow_expanded_train_v1 import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    build_expanded_train,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the SHA-verified 16-map compact-flow expanded-train label "
            "product; no development, final, runtime, or promotion claim."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    report = build_expanded_train(
        (PROJECT_ROOT / arguments.config).resolve(),
        output=(PROJECT_ROOT / arguments.output).resolve(),
        workers=arguments.workers,
        dry_run=arguments.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
