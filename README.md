# Structure-Guided LNS2

This repository evaluates neighborhood selectors around an official-behavior
MAPF-LNS2 native solver boundary. The active branch deliberately contains only
four controller identities:

- `official_adaptive`: native LNS2 Adaptive neighborhood selection.
- `v2-full`: the frozen full-load pairwise selector.
- `mixed-full-v2`: the retained mixed-load v2 bundle.
- `v3-s3`: the sequence-aware S3 controller; runnable, but not promoted as the
  default quality controller.

LNS2 is the solver. V2 and V3-S3 are selector policies inside that solver, so
they are organized under one layered package instead of parallel version trees.

## Current research status

The latest warehouse confirmation did not pass the Dual16 promotion gate.
Retained positive results are conditional on their registered inputs and timing
boundaries, not evidence of a universal LNS2 replacement. Repeated post-stall
rescue diagnostics are paused; runnable historical tools are not pending tasks.
See the [Chinese report outline and claim boundaries](docs/WAREHOUSE_REPAIR_REPORT_PLAN_ZH.md)
and the [research decision review](docs/RESEARCH_DECISION_REVIEW_ZH.md) for the
current conclusions and requirements before any new experiment.

The latest bounded SA diagnostic found certified multi-agent capacity shortages
under fixed outsider paths, but no promoted controller or new TTF result. See
the [capacity follow-up and paired intervention report](docs/SA_CAPACITY_FOLLOWUP_RESULTS_ZH.md).
The [bounded outsider-release follow-up](docs/SA_CAPACITY_RELEASE_RESULTS_ZH.md)
finds input-derived candidates that remove some capacity contradictions; it
does not establish native repair success or authorize an online controller.
The [history-controlled native discrimination](docs/SA_CAPACITY_NATIVE_RESULTS_ZH.md)
completed 264 paired diagnostic repairs without a qualifying capacity-label
benefit. Longer continuation and online deployment remain stopped.

## Layout

```text
lns2_selector/
  solver/          native solver boundary
  runtime/         contracts, metrics, fingerprints, portable inference
  controllers/     official, v2, mixed-v2, v3-s3 adapters
  training/        shared retained training utilities
  evaluation/      wall-clock evaluation entry points
  compatibility/   read-only historical metric support
experiments/        retained collection, training, and audit implementations
scripts/            thin command-line entry points
tests/
  solver/           official native boundary tests
  runtime/          shared selector runtime tests
  controllers/      four-controller contract tests
  evaluation/       metrics and paired-report tests
  integration/      data, pipeline, resume, and maintenance tests
  data/             native and generator fixtures only
third_party/mapf_lns2/
                    upstream native source boundary
```

The retention rules and per-file classification are recorded in
`docs/RETENTION_POLICY.md` and `configs/retention_manifest.json`.

## Environment

The verified Linux environment is Ubuntu 22.04 under WSL2. Test the current
working tree through its mounted Windows path; `/home/beta/LNS2-RL` is a
separate user checkout and must not be overwritten for validation:

```powershell
$repo = (Get-Location).Path
$wslRepo = (wsl.exe -d Ubuntu-22.04 -- wslpath -a $repo).Trim()
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/python3 scripts/check_environment.py --profile runtime-wsl
```

An empty WSL distribution list observed from a restricted sandbox is a known
false negative. Do not reinstall or register another distribution in response.

## Build and test

Run native build and tests against that same mounted working tree:

```bash
cmake -S . -B build/linux/project -DCMAKE_BUILD_TYPE=Release
cmake --build build/linux/project -j
ctest --test-dir build/linux/project --output-on-failure
/usr/bin/python3 -m pytest -q
```

The repository hygiene audit is read-only:

```bash
/usr/bin/python3 scripts/audit_repository_hygiene.py \
  --config configs/repository_hygiene.json
```

## Wall-clock evaluation

New evaluations use the complete deadline window. They do not stop at 100
repairs and do not use the historical 100-repair AUC for promotion.

```bash
/usr/bin/python3 scripts/run_lns2_tradeoff_evaluation.py \
  --mode quick \
  --evaluation-tracks wall-clock \
  --controllers official_adaptive,v2-full \
  --wall-clock-seconds 300 \
  --output build/selector-wall-clock-quick-v1
```

Optional retained controllers can be included explicitly:

```bash
/usr/bin/python3 scripts/run_lns2_tradeoff_evaluation.py \
  --mode quick \
  --evaluation-tracks wall-clock \
  --controllers official_adaptive,v2-full,mixed-full-v2,v3-s3 \
  --v3-bundle build/initlns-v3-s3-mixed-load-pilot-v5-adaptive/controller \
  --wall-clock-seconds 300 \
  --output build/selector-four-controller-quick-v1
```

Reports include success rate, time to feasible, final conflicts, repair rounds,
full wall-clock conflict AUC, PP time, neighborhood-selection time, and timing
closure checks.

## V3-S3 collection and training

The retained pipeline has four stages: source generation, collection, training,
and native audit.

```bash
/usr/bin/python3 scripts/run_v3_training_pipeline.py source \
  --config <dataset-config> --output <pipeline-output>
/usr/bin/python3 scripts/run_v3_training_pipeline.py collect \
  --config <dataset-config> --output <pipeline-output> --resume
/usr/bin/python3 scripts/run_v3_training_pipeline.py train \
  --config <dataset-config> --output <pipeline-output> --resume
/usr/bin/python3 scripts/run_v3_training_pipeline.py native-audit \
  --config <dataset-config> --output <pipeline-output> --resume
```

Producer identity binds direct source dependencies, the loaded native binary,
native timing schema, Python, and key training libraries. A changed identity
cannot resume into an existing current-schema output.

## Historical results

Retired value/receding-Q and old stall/rescue/V3 execution chains are not active
code. Their decisions, configurations, and hashes are frozen under
`artifacts/initlns-receding-q-frozen-v1` and documented in
`docs/RETIRED_RESEARCH_EVIDENCE.md`.

Historical source remains recoverable from the remote cleanup checkpoints,
starting with `backup/selector-cleanup-00-original`. Historical 100-repair AUC
is retained only in `lns2_selector.compatibility` for reading registered old
reports.
