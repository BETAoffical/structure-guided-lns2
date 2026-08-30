#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_hierarchical_ch_readiness_v1 import (  # noqa: E402
    LABEL_AUDIT_SHA256,
    LEGACY_FOLD_CONFIG_SHA256,
    check_hierarchical_ch_readiness,
)


DEFAULT_LABELS_ROOT = PROJECT_ROOT / "build" / "stride-hierarchical-ch-labels-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "build" / "stride-hierarchical-ch-readiness-v1"
DEFAULT_FOLD_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "stride_fresh_matched_unique_action_hierarchical_overlay_v1.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check frozen hierarchical C/H label support only. "
            "No model is fitted, exported, or enabled at runtime."
        )
    )
    parser.add_argument("--labels-root", default=str(DEFAULT_LABELS_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--legacy-fold-config", default=str(DEFAULT_FOLD_CONFIG))
    arguments = parser.parse_args()

    report = check_hierarchical_ch_readiness(
        project_root=PROJECT_ROOT,
        labels_root=arguments.labels_root,
        output=arguments.output,
        label_audit_sha256=LABEL_AUDIT_SHA256,
        legacy_fold_config=arguments.legacy_fold_config,
        legacy_fold_config_sha256=LEGACY_FOLD_CONFIG_SHA256,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
