# STRIDE-LNS Stage 2 Pilot report

This report records the executed Pilot cohort contract. Generated artifacts stay
under `build/` and are not committed.

## Registered cohort

- 15 development maps: nine generated layouts and six MovingAI development maps.
- 96 tasks with 100, 200, 400, or 600 agents.
- solver seeds 101, 202, and 303.
- two source policies: official Adaptive and frozen `v2-full`.
- 120 states per policy, at most two states per source episode.
- only source decision indices 0 through 11 are eligible.
- all 14 maps that produced conflicts are represented; `den520d` produced no
  conflicting state under the registered tasks and seeds.
- 79 low/mid-agent states and 161 high-agent states.
- `v2-full` contributes 40 early, 40 middle, and 40 late states. Official
  Adaptive contributes replayable early states only because its historical PP
  transitions did not register deterministic PP seeds.

The first selection admitted arbitrarily late source decisions because the
wall-clock collector did not enforce the intended 12-decision configuration.
Observed decision indices reached 367, and 11 states showed replay or candidate
instability. `lns2.stride.state_selection.v2` fixes the contract by applying an
explicit index cap, excluding every state with observed instability, and
requiring three independent preflight replays with identical state and candidate
signatures.

## Executed result

- selection-v2: 240 states; SHA-256
  `c5432e43847ff1b756ce1fd5cff8baadfd8a8a16533b49800e1fa865f19734b5`.
- preflight: 240/240 states passed three repetitions; report SHA-256
  `28a241485198f93235efecac8aa8b8b2b09a9d5e8db9a3bfb1adf0c7b2610c5b`.
- collection: 240/240 states, zero errors, 17,084 paired repair trials.
- candidate counts: one state with 15, 11 with 16, 24 with 17, and 204 with
  the full 18 unique candidates. Lower counts are deterministic deduplication,
  not missing trials.
- repair-trial SHA-256:
  `9b82158312a5a67572e41013fae389d8b3ac6532c176bd457602304d72407a8d`.
- label coverage: 240/240 states, 4,271 candidate aggregates, 28,228 unordered
  dominance pairs, and 56,456 oriented training rows.
- every state has at least five dominance pairs; coverage is 100 percent.
- label schema: `lns2.stride.quality_label.v1`; runtime is not used in labels.

Of the final states, 188 were reused from complete first-run artifacts only after
state identity, pre-action fingerprint, producer identity, candidate/seed
cartesian completeness, and per-file hashes passed. The remaining 52 states were
computed anew. The final collection run fingerprint is
`3fc7bf9ee65c93cb038fb86969e7022c89db5acc24d1e2de22902ed0ebc83011`.

## Reproduction commands

```bash
PYTHONPATH=build/linux/project /usr/bin/python3 scripts/run_stride_pipeline.py select-states-v2 \
  --source build/stride-stage2-sources-v2 \
  --source build/stride-stage2-sources-seed303-v1 \
  --exclude-report configs/stride_stage2_instability_exclusions.json \
  --output build/stride-stage2-pilot-selection-v2

PYTHONPATH=build/linux/project /usr/bin/python3 scripts/run_stride_pipeline.py preflight-selection \
  --selection build/stride-stage2-pilot-selection-v2/state_selection.jsonl \
  --output build/stride-stage2-pilot-preflight-v2 --workers 4 --repetitions 3

PYTHONPATH=build/linux/project /usr/bin/python3 scripts/run_stride_pipeline.py collect \
  --selection build/stride-stage2-pilot-selection-v2/state_selection.jsonl \
  --output build/stride-stage2-pilot-v2 --workers 4 --resume

PYTHONPATH=build/linux/project /usr/bin/python3 scripts/run_stride_pipeline.py label \
  --trials build/stride-stage2-pilot-v2/repair_trials.jsonl \
  --output build/stride-stage2-labels-v2
```
