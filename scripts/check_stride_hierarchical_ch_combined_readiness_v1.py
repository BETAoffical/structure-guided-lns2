#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_hierarchical_ch_combined_readiness_v1 import (  # noqa: E402
    LEGACY_CONSENSUS_LABEL_AUDIT_SHA256,
    ORIGINAL_LABEL_AUDIT_SHA256,
    check_hierarchical_ch_combined_readiness,
)
from experiments.stride_hierarchical_ch_readiness_v1 import (  # noqa: E402
    LEGACY_FOLD_CONFIG_SHA256,
)


DEFAULT_ORIGINAL = PROJECT_ROOT / "build" / "stride-hierarchical-ch-labels-v1"
DEFAULT_EXTENSION = (
    PROJECT_ROOT / "build" / "stride-hierarchical-ch-legacy-consensus-labels-v1"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "build" / "stride-hierarchical-ch-combined-readiness-v1"
)
DEFAULT_FOLD_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "stride_fresh_matched_unique_action_hierarchical_overlay_v1.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the two frozen sequential-development label audits and "
            "check support only. No model is fitted, exported, or enabled."
        )
    )
    parser.add_argument("--original-labels-root", default=str(DEFAULT_ORIGINAL))
    parser.add_argument(
        "--legacy-consensus-labels-root", default=str(DEFAULT_EXTENSION)
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--legacy-fold-config", default=str(DEFAULT_FOLD_CONFIG))
    arguments = parser.parse_args()

    report = check_hierarchical_ch_combined_readiness(
        project_root=PROJECT_ROOT,
        original_labels_root=arguments.original_labels_root,
        legacy_consensus_labels_root=arguments.legacy_consensus_labels_root,
        output=arguments.output,
        original_label_audit_sha256=ORIGINAL_LABEL_AUDIT_SHA256,
        legacy_consensus_label_audit_sha256=(
            LEGACY_CONSENSUS_LABEL_AUDIT_SHA256
        ),
        legacy_fold_config=arguments.legacy_fold_config,
        legacy_fold_config_sha256=LEGACY_FOLD_CONFIG_SHA256,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
