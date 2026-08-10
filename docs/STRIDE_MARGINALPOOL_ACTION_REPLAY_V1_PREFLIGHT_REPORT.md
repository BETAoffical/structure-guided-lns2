# STRIDE-MarginalPool Action Replay v1 Preflight Report

## Outcome

The preregistered execution-integrity preflight passed on 2026-08-10.
Candidate outcomes were not summarized or used to alter the cohort, thresholds
or full-run protocol.

| Check | Result |
|---|---:|
| Unique repair states | 2 / 2 |
| Existing candidates | 65 / 65 |
| Paired PP trials | 130 / 130 |
| Trial indices | 0, 1 |
| Errors | 0 |
| Timeouts | 0 |
| Resumable artifacts accepted without new PP work | 2 / 2 |

All registered integrity gates passed:

- Stage 1 state-blob SHA-256 and full state fingerprint;
- native `reset_paths` repair-structure identity;
- exact frozen candidate agent sets;
- exact selected-candidate 124-dimensional feature reproduction;
- one shared PP seed per state/trial index across all candidates, with two
  distinct trial seeds per state;
- native explicit-neighborhood, requested-seed and applied-seed semantics;
- no runtime, TTF, Cost-to-Go or future-trajectory fields in the product.

## Frozen identities

```text
run fingerprint:
460c650b4a2974c87235e71e8b356c30a35e26ee954c8eecedfbd06649c4cff1

run_config.json:
51bd53d289646be462f044a113d17a89aec38e1d64fa9921b0f0fafa97135dc5

collection_report.json:
d8a86226306327ef0214a37571d6f895d73db2c96ed52377e13cb3ae6f36d998

state_manifest.jsonl:
79e8fbea1071a6583303ed2d43b94d23e813da30252db522f1e6b749cd6c5a86

logical_checkpoint_manifest.jsonl:
c8f281635e806398215916edfb6b38d1e164c9901f928e19ccbbc33814bc4f6f
```

The full 78-state, 2,502-action, 40,032-trial collection is therefore allowed
to start.  This preflight is an execution qualification only and supplies no
candidate-quality, long-term, TTF or controller-promotion evidence.
