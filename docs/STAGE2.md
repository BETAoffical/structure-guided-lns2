# Stage 2: Raw LNS Experience

Stage 2 records what the baseline LNS actually attempted. It does not yet
compute map embeddings, retrieve similar experiences, or train a policy.

## Trace interface

Pass `--trace FILE` to `lns2_cli`. The output is JSON Lines with one
`iteration` event per attempted neighborhood repair and one final `summary`
event.

Trace schema version 2 iteration fields:

- `schema_version`, `event_type`, and `solver_seed`;
- `iteration`;
- `seed_conflict`, the conflict-graph edge used to start selection;
- `neighborhood`, the selected Agent IDs;
- conflicts and sum of costs before/after replanning;
- `candidate_valid` and `accepted`;
- `replan_runtime_ms`.
- the first conflict event for every conflicting Agent pair, including type,
  timestep, and cells;
- selected-neighborhood paths before and after candidate replanning.

If replanning cannot produce complete candidate paths, `candidate_valid` is
false and the after fields are `-1`. Omitting `--trace` does not change solver
behavior.

## Trace V4/V5 extension

Stage 5 v2 adds opt-in `--candidate-mode collect` tracing. The solver keeps
the legacy main trajectory but evaluates deterministic candidate
neighborhoods with one shared Agent priority. Use
`--candidate-generator-profile full8` to reproduce the original eight
candidate pool, or `--candidate-generator-profile core5` for the current
faster five-candidate pool. Each trial has an independent two-second limit,
consumes no main RNG state, and is excluded from the five-second main search
budget.

Trace V4 adds full current paths, candidate membership, explicit replanning
orders, validity, conflict/cost outcomes, and repaired candidate paths. The
ordinary Stage 2 command remains Trace V2 and is unchanged.

Stage 5 v2.2 adds `--candidate-replan-order-seeds A,B,C`. When more than one
order seed is supplied, candidate collection writes Trace V5. The top-level
candidate fields remain the `order_seed=0` trial for compatibility, while
`order_trials` records every deterministic replanning order and its outcome.
The Python collector aggregates these order trials into expected candidate
labels and also writes `candidate_order_cases.jsonl` for label-noise analysis.

Stage 5 v4 adds optional `--candidate-rollout-horizons A,B,C`. When supplied,
candidate collection writes Trace V6. Every valid candidate/order trial first
performs the one-step repair, then continues an isolated short LNS rollout to
the requested horizons. The rollout labels record horizon, solved flag,
iterations, accepted iterations, remaining conflicts, cost, and runtime.
Rollout simulation uses an isolated RNG and does not change the main solver
trajectory.

## Batch collection

```powershell
python scripts/collect_experience.py `
  --dataset build/feasibility-dataset `
  --solver build/windows/Release/lns2_cli.exe `
  --split train `
  --seeds 1,2,3 `
  --time-limit-ms 5000 `
  --output build/experience
```

The feasibility run uses solver seeds `1,2,3`, neighborhood size `6`, `500`
iterations, and a `5000 ms` per-run limit. The 72 training tasks therefore
produce 216 runs. Exit codes `0` and `1` are valid experiment results (solved
and unsolved); process or input errors are collection errors.

The collector writes one trace per task/seed pair,
`collection_manifest.jsonl`, and `collection_summary.json`. Candidate
collection also records candidate profile, replanning-order seeds, rollout
horizons, filter settings, worker count, and an estimated count of candidate
trials and rollout labels. A later stage may enrich these records with local
map structure and use them for retrieval or learned selection.
