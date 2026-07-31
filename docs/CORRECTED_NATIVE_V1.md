# Corrected native semantics v1

`lns2.corrected_native.v1` is a deliberate semantic revision of the bundled
MAPF-LNS2 native implementation. It is not byte-for-byte or action-for-action
equivalent to the frozen upstream behavior.

The revision:

- accumulates every target visited by the Target path search;
- makes native heap and conflict comparisons deterministic strict weak
  orderings;
- uses unbiased 0-based cumulative draws for Target and Random weighted
  sampling;
- rejects duplicate starts and duplicate goals at the validated Python and
  native CLI input boundaries;
- rejects negative seeds in both native CLIs; and
- excludes global RNG-mutex wait time from
  `agent_and_solver_setup_seconds`.

The Python extension exposes the revision as:

```text
native_semantics_schema = lns2.corrected_native.v1
```

## Frozen behavior evidence

For the existing 200-agent `random-32-32-20`, seed-29,
zero-repair-iteration path fixture:

- legacy frozen-official SHA-256:
  `915ee104f0168c463f05925541fef1c22ec1eb37e9bf8df7ab09807753013ecf`
- corrected-native.v1 SHA-256:
  `50a1e2f757778cf2b59b59a22172e1381304c271d17bef53d2ad73b128639d65`

The CTest fixture is consequently named
`lns2_corrected_native_determinism_*`, not official parity.

Existing frozen-official experiment artifacts remain historical evidence.
They must not be resumed or combined with corrected-native.v1 artifacts.
Comparative conclusions under corrected-native.v1 require fresh baselines,
new output directories, and the corrected native schema and binary SHA in the
producer identity.

## Fixed-seed semantic audit

The ignored `build/corrected-native-audit-v1` directory keeps the legacy
binary snapshot. The final corrected side is recorded separately under
`build/corrected-native-audit-v2`; its loaded native SHA-256 is
`8e47b42975521f3bbee31deaa87654d2e2aff8b0366b28d677af10f0c8401cc2`.
Record it from the canonical WSL build, then compare it with the legacy
snapshot:

```bash
PYTHONPATH=build/linux/project \
  /usr/bin/python3 scripts/audit_corrected_native_semantics.py \
  --map tests/data/proposal_required.map \
  --scenario tests/data/proposal_required.scen \
  --agent-count 80 --seeds 17,29,31 --repair-steps 10 \
  --output build/corrected-native-audit-v2/corrected/semantics.json

/usr/bin/python3 scripts/compare_corrected_native_semantics.py \
  --baseline build/corrected-native-audit-v1/baseline/semantics.json \
  --corrected build/corrected-native-audit-v2/corrected/semantics.json \
  --output build/corrected-native-audit-v2/comparison.json
```

This checks reset paths, candidate neighborhoods, repair orders, and conflict
trajectories. It demonstrates a native semantic change but does not replace a
paired end-to-end controller evaluation.

## Formal rerun contract

Use `configs/balanced_wall_clock_corrected_native_v1.json` and new
qualification, schedule, collection, and report directories.

- `environment.max_repair_iterations = 0`;
- `max_decisions = 0`;
- the 600-second wall-clock budget is the execution stop;
- `metric_iteration_budget = 100` is only the padded fixed-step AUC reporting
  window and never stops repair execution; and
- `WALL_CLOCK_SAFETY_MAX_DECISIONS = 100000` is an abnormal-run guard, not the
  experimental iteration budget.

The corrected reset can change fingerprints, conflicts, and initial PP load.
Requalifying only the frozen 36 task/seed keys did not preserve the registered
difficulty cells, so that path is not used for the formal comparison. The
replacement protocol qualifies the complete registered 688-task pool for all
three solver seeds and, where a registered cell lacks enough distinct maps, an
outcome-blind generated extension. It then fills the original 36 experimental
slots with an exact min-cost assignment while preserving:

- every conflict/load/source quota;
- 18 generated and 18 MovingAI tasks;
- one task per map; and
- the original schedule groups and controller orders.

The selector reads only reset-time conflict and PP-load measurements. It never
reads Adaptive, v2-full, or Mixed Full v2 repair outcomes. A diagnostic run
completed all 2064 base qualifications and 456 targeted-extension
qualifications with zero errors or timeouts, and demonstrated that all 36
registered slots are fillable. Because producer identity was subsequently
strengthened, those diagnostic qualifications are not resumable formal input:
the final chain must requalify into new `v3` output directories. Full-pool
schedule, report, and selector artifacts use schema v3; v1/v2 artifacts remain
read-only evidence and cannot resume or materialize the v3 chain.

The generated formal config is validated before any lane starts. It fixes
`0/0/100/600/600/660` for Python decisions, native repair iterations, metric
window, wall time, native time limit, and process timeout respectively. A
regression fixture executes 101 repairs, keeps the complete 102-state
trajectory, and computes the metric from only its first 100 repair intervals.
Adaptive, v2-full, and Mixed Full v2 then run on the same materialized
36-task schedule.
