# STRIDE RepairDependency Predictability v1 Execution Amendment

The first formal `analyze` invocation stopped before writing any scientific
row or report. All context and evaluation jobs had completed in memory, but
report assembly passed `project_root` positionally to the keyword-only
`producer_identity` helper and raised `TypeError`.

The implementation change supplies `project_root` by keyword and explicitly
records `native_required=false`, because this audit reads frozen JSON states
and performs no native PP call. Predictor definitions, input hashes, cohort,
labels, gates, worker count, task order, and output schemas are unchanged. No
predictor metric was emitted or inspected before this amendment.
