from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.compact_controller_model import (  # noqa: E402
    update_controller_promotion_evidence,
)


def _resolve(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attach benchmark/run evidence and refresh controller-v2 promotion."
    )
    parser.add_argument(
        "--bundle", default="artifacts/initlns-closed-loop-controller-v2"
    )
    parser.add_argument("--performance-benchmark")
    parser.add_argument("--quick-status")
    parser.add_argument("--formal-status")
    arguments = parser.parse_args()
    result = update_controller_promotion_evidence(
        _resolve(arguments.bundle),
        performance_benchmark=_resolve(arguments.performance_benchmark),
        quick_status=_resolve(arguments.quick_status),
        formal_status=_resolve(arguments.formal_status),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
