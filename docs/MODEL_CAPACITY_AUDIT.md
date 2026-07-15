# InitLNS GBDT Model Capacity Audit

## Scope

This audit tests whether the failed explicit-neighborhood ranker was limited by
the capacity of its `HistGradientBoostingClassifier`. It does not collect new
data, use static context, change the pairwise labels, tune hyperparameters, or
train RL.

The registered inputs contain 288 policy-Train states on 12 maps, 23 historical
anchor states used only for fitting, 154 sealed development-Validation states,
7,914 explicit neighborhoods, and 31,656 repair-order trials. The four models
use the same 139 `realized_dynamic` features and equal total weight per state.

## Registered Capacities

| Capacity | Trees | Maximum leaves | Minimum leaf samples |
| --- | ---: | ---: | ---: |
| `small` | 50 | 7 | 40 |
| `current` | 100 | 15 | 20 |
| `large` | 300 | 31 | 10 |
| `very_large` | 500 | 63 | 5 |

All models use learning rate `0.05`, L2 regularization `0.1`, and random seed
`20260714`. Model capacity is the only changed factor.

## Evaluation Protocol

Capacity selection uses leave-one-map-out evaluation over the 12 Train maps.
All candidates and states from one map are held out together. The 23 historical
states are included in every training fold and never evaluated. The current
model must reproduce the registered LOMO result before challengers are judged.

A larger model must improve Pareto top-1 by at least 3 percentage points or
reduce mean conflict regret by at least 5%, keep the other metric within its
degradation bound, avoid bootstrap evidence of degradation, be no worse on at
least 8 of 12 maps, and avoid unsupported neighborhood-size collapse.

Development Validation can be read for a one-time frozen evaluation only after
a larger model passes every Train LOMO gate. The independent `20270421` dataset
can be generated only after that Validation gate also passes.

## Formal Result

The formal run completed in 1,044.2 seconds. The registered `current` result was
reproduced exactly: top-1 `49.6528%` and mean conflict regret `0.282439`.

| Capacity | Train top-1 | Train regret | Train pair accuracy | LOMO top-1 | LOMO regret | LOMO pair accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `small` | 50.6944% | 0.281736 | 0.6854 | 48.6111% | 0.295270 | 0.6766 |
| `current` | 52.7778% | 0.259974 | 0.7067 | 49.6528% | 0.282439 | 0.6784 |
| `large` | 58.3333% | 0.212470 | 0.7885 | 43.4028% | 0.347564 | 0.6748 |
| `very_large` | 68.4028% | 0.136256 | 0.8727 | 38.5417% | 0.365694 | 0.6696 |

Relative to `current`, `large` loses 6.25 percentage points of LOMO top-1 and
worsens conflict regret by 23.06%. It is no worse on only 5 of 12 maps. The 95%
map-bootstrap intervals are entirely negative for both top-1 change and regret
improvement. `very_large` loses 11.11 percentage points, worsens regret by
29.48%, and is no worse on only 2 of 12 maps. Neither model collapses to one
neighborhood size, so size collapse does not explain the failure.

The registered decision is `overfit` and `stop_tabular_capacity_tuning`.
Validation was not evaluated, no portable winner was exported, and independent
confirmation generation remained disabled.

## Interpretation

Additional tree capacity substantially improves fitting on the known Train
maps while consistently reducing performance on a completely held-out map.
The evidence therefore rejects simple GBDT under-capacity as the main cause of
the current failure. It does not prove that no model can learn the task; it shows
that increasing capacity over the existing 139 hand-aggregated features makes
cross-map generalization worse.

The next model study should replace the flat summary representation with a
conflict-graph and selected-agent-set representation. Further tabular GBDT
capacity tuning and RL expansion remain stopped until that representation has
its own preregistered offline gate.

The generated predictions and JSON report remain under the ignored directory
`build/initlns-model-capacity-audit-v1`.
