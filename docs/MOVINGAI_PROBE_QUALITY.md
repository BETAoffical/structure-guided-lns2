# MovingAI Probe Quality Audit

## Purpose

The original mechanism probe produced 1,368 Horizon-1 outcome rows. This audit
checks how much independent information those rows contain before another model
or RL run is allowed. It does not reinterpret the original registered gates as a
positive result; its adequacy thresholds are diagnostic and were defined after
the original probe.

Run the audit with:

```powershell
python scripts/audit_movingai_probe_quality.py `
  --collection build/movingai-mechanism-probe-collection `
  --output build/movingai-probe-quality-audit
```

## Findings

- The 24 raw state rows collapse to 12 unique state fingerprints. Solver seeds
  0 and 1 reached the same states, so their outcomes provide four action trials,
  not independent state samples.
- Only seven task instances and four maps contribute repair labels. Three maps
  have labels at both tested densities. Every map uses only random-1 and every
  layout family has only one map.
- State-normalized action eta-squared is 0.493, with a task-bootstrap 95% interval
  of approximately `[0.386, 0.565]`. The previous pooled value was biased toward
  states with large absolute conflict counts.
- Candidate rankings from the two duplicate-solver halves have mean Spearman
  correlation 0.424. Their best-candidate sets have mean Jaccard 0.186. Four
  trials do not provide a stable oracle label.
- The same `(seed, rule, size)` action realizes different agent sets. Mean
  pairwise neighborhood Jaccard is 0.386 and exact stability is 13.7%. Target is
  comparatively stable (`0.68-0.76` Jaccard), while Random is highly stochastic
  (`0.09-0.15`).
- Horizon-1 conflict AUC is exactly `(conflicts_before + conflicts_after) / 2`.
  It is not an independent objective within a state.
- Effectiveness-only and generated-node-aware Pareto sets have mean Jaccard
  0.344. Compute cost materially changes the preferred candidates.
- Exact map-label enumeration gives `p=0.086` over all 630 assignments. A new
  directional density statistic gives `p=0.250` over all eight high/low swaps.
  These values are diagnostic only because the action labels are unstable.

The current decision is
`expand_independent_context_coverage_and_action_trials`. The 1,368 rows are
useful for diagnosing the experiment, but they are insufficient evidence for
contextual transfer or for starting RL.

## Corrected Data Design

The MovingAI source adapter now supports multiple pinned random scenarios while
remaining compatible with the original single-scenario manifest. The v2 probe
uses random scenarios 1, 2, and 3, one solver seed for state acquisition, and
eight independent random seeds per candidate action.

The v2 qualification contains 36 tasks across six maps. All 36 runs are valid,
24 enter repair, and there are no errors. Adaptive solves 32/36 baselines. The
four failures are three 200-agent maze tasks and one 200-agent room task at the
100-repair limit. `den520d` remains conflict-free at 400 and 600 agents; it is
retained in qualification statistics but cannot contribute repair labels.

Counterfactual sources are restricted to 1-200 initial conflicts. This excludes
the three extreme maze-200 episodes while preserving difficult source episodes
that Adaptive does not finish. The filter is part of the run fingerprint.

```powershell
python scripts/prepare_movingai_probe.py `
  --dataset build/movingai-dev `
  --config configs/movingai_mechanism_probe_v2_dataset.json `
  --output build/movingai-mechanism-probe-v2-dataset
```

The v2 probe is still a mechanism and label-stability experiment. Multiple
scenarios reduce task confounding, but one map per layout family cannot establish
map transfer. A later confirmation must include at least two independent maps per
layout family and must keep test/OOD layouts sealed.

## Next Gate

Before contextual ranking:

- obtain at least 24 independent states and 12 repairable task instances;
- use at least eight trials per candidate;
- confirm trial-split rank stability and bootstrap winner stability;
- keep effectiveness and generated-node-aware Pareto results separate;
- use at least three task scenarios per map;
- use independent layout replicates for any transfer claim;
- collect states from a learned policy only after an Adaptive-state policy passes
  its first closed-loop gate.

If Random and Collision remain highly unstable after eight trials, the next model
should rank several realized neighborhoods rather than treating the pre-generation
action tuple as a deterministic agent set.

## V2 Partial Confirmation

The corrected v2 run recovered 20 of 21 eligible source episodes before analysis:
35 independent states, 20 labeled task instances, 7,776 outcomes, and exactly eight
trials per candidate. One 600-agent warehouse episode with two initial conflicts
did not complete within a clean 20-minute execution window. The partial result is
therefore explicitly marked as compute-budget selected rather than presented as a
complete registered run.

Eight trials improved split-half candidate-rank Spearman from 0.424 to 0.684 and
Pareto-family Jaccard to 0.535. This confirms that extra trials recover part of the
expected-action ranking signal. It does not produce a reliable hard top-1 label:
best-candidate Jaccard is 0.376 and 82.9% of states have less than 80% bootstrap
confidence in their modal winning family.

Statewise action eta-squared is 0.468, with a task-bootstrap 95% interval of
approximately `[0.423, 0.500]`. Mean realized-neighborhood Jaccard remains 0.391.
Target neighborhoods are substantially more stable than Random neighborhoods, so
the action tuple should be interpreted as a stochastic generator distribution.

Map-specific oracle heterogeneity is detectable (`p=0.013` from 10,000 deterministic
Monte Carlo permutations), but directional density alignment is absent (`p=0.884`).
Because every layout family still has one map, the map result is not transfer
evidence. Effectiveness-only and generated-node-aware Pareto sets overlap by only
0.297, so compute cost must remain a separate objective. Host command timeouts once
left overlapping WSL collectors, so runtime-sensitive metrics from this run are
invalidated; conflict, neighborhood, trial-seed, fingerprint, and generated-node
metrics remain the registered analysis inputs.

The aggregate manifest and partial audit are reproduced with:

```powershell
python scripts/recover_counterfactual_manifest.py `
  --output build/movingai-mechanism-probe-v2-collection

python scripts/audit_movingai_probe_quality.py `
  --collection build/movingai-mechanism-probe-v2-collection `
  --probe-config configs/movingai_mechanism_probe_v2_dataset.json `
  --output build/movingai-mechanism-probe-v2-quality `
  --allow-partial --invalidate-runtime
```

The next data stage should not add more repetitions to these same states. It should
add independent maps within layout families, balance scenario/density coverage,
and use a bounded high-agent source budget. The bounded v2 configuration excludes
sources above 400 agents and records every exclusion in the source manifest.
