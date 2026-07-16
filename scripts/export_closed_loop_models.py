from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments._common import sha256_file as _sha256  # noqa: E402
from experiments.closed_loop_confirmation import (  # noqa: E402
    export_portable_policy_bundle,
    verify_portable_policy_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export frozen sklearn pairwise rankers for dependency-free inference."
    )
    parser.add_argument(
        "--frozen-models",
        default="build/initlns-natural-distribution-confirmation-v1-frozen-models",
    )
    parser.add_argument(
        "--config", default="configs/closed_loop_confirmation_collection.json"
    )
    parser.add_argument(
        "--output",
        default="artifacts/initlns-closed-loop-policy-v1",
    )
    arguments = parser.parse_args()
    config = json.loads(Path(arguments.config).read_text(encoding="utf-8"))
    manifest = export_portable_policy_bundle(
        arguments.frozen_models, config["model_registration"], arguments.output
    )
    output_root = Path(arguments.output).resolve()
    registration = dict(config["model_registration"])
    registration["deployment_bundle"] = str(output_root)
    registration["deployment_manifest_sha256"] = _sha256(
        output_root / "portable_manifest.json"
    )
    registration["portable_models"] = str(output_root)
    registration["portable_model_sha256"] = {
        row["profile"]: row["sha256"] for row in manifest["models"]
    }
    equivalence = verify_portable_policy_bundle(arguments.frozen_models, registration)
    (output_root / "equivalence_report.json").write_text(
        json.dumps(equivalence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "equivalence": equivalence}, indent=2, sort_keys=True))
    return 0 if equivalence["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
