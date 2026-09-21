"""Read-only design preflight. No collect/train/timing command is implemented."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import contained_file, json_fingerprint, read_json, sha256_file
from experiments.sa_paired_completion import require

CONFIG = "configs/sa_onpolicy_contract.json"


def inspect(root=ROOT):
    cfg = read_json(root / CONFIG)
    require(cfg["stage"] == "design_and_contract_tests_only", "design scope changed")
    require(all(cfg[k] is False for k in ("collection_authorized", "training_authorized",
                                         "formal_ttf", "automatic_promotion")), "execution not supported")
    hashes = dict(cfg["historical_documents"], **{cfg["source_plan"]: cfg["source_plan_sha256"]})
    for name, digest in hashes.items():
        require(sha256_file(contained_file(root, name, field="design evidence")) == digest,
                "evidence changed: " + name)
    source = read_json(root / cfg["source_plan"])
    require(source["binding"] == json_fingerprint({k: v for k, v in source.items() if k != "binding"}),
            "historical plan binding")
    split = source["split"]
    train, held = set(split["train_maps"]), set(split["validation_maps"])
    require(len(train) == 6 and len(held) == 2 and not train & held, "historical map split")
    cases = source["cases"]
    groups = {role: [] for role in ("train", "development_holdout")}
    for c in cases:
        require(c["map_id"] in train | held, "unknown map")
        role = "train" if c["map_id"] in train else "development_holdout"
        for seed in c["solver_seeds"]:
            groups[role].append((c["task_id"], seed))
    require(all(len(v) == len(set(v)) for v in groups.values()), "duplicate task/seed")
    require((len(groups["train"]), len(groups["development_holdout"])) == (24, 8), "case inventory")
    p = cfg["proposal"]
    train_jobs = len(groups["train"]) * p["train_replicas_per_condition"] * p["maximum_updates"]
    eval_jobs = len(groups["development_holdout"]) * p["evaluation_replicas_per_condition"] * len(p["evaluation_arms"])
    return dict(schema="lns2.sa.onpolicy_design_check.v1", status="design_checked_not_execution_ready",
                config_sha256=sha256_file(root / CONFIG), checked_evidence=hashes,
                train_conditions=len(groups["train"]), heldout_conditions=len(groups["development_holdout"]),
                proposed_training_episodes=train_jobs, proposed_evaluation_episodes=eval_jobs,
                maximum_proposed_repairs=(train_jobs + eval_jobs) * p["max_decisions"],
                actual_solver_calls=0, actual_training_updates=0,
                blockers=["No on-policy trajectory collector or optimizer implemented",
                          "Task files, full feature schema, model/native and implementation SHA need a new execution seal",
                          "New selector/PP/SA random streams and replica independence require native replay verification",
                          "Runtime safety, atomic resume and stop-at-episode-boundary still need integration tests",
                          "Collection/training require a separate bounded execution decision"],
                limitations=["Only the five listed design inputs were hashed; not a full historical-data audit",
                             "Previously viewed holdout maps cannot prove independent generalization",
                             "Work-budget completion is not wall-clock TTF"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(inspect(), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
