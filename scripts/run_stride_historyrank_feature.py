from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_historyrank_feature import (  # noqa: E402
    run_historyrank_feature_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered HistoryRank feature audit."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_historyrank_feature_audit(config_path=args.config, output=args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
