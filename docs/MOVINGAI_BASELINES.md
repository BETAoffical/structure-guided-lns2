# MovingAI Feasibility Baselines

## Solvers

Official MAPF-LNS2 remains pinned at commit
`1369823985a15944f9a339226d521f61605a6d17`. Official GPBS is vendored separately
at commit `43f2a6fea50893871219b674535f83920175ae04` and built as
`gpbs_official`. Both retain their USC Research Licenses.

GPBS is an end-to-end feasibility baseline. It is not linked into the LNS2 policy
API and is never described as Target/Collision/Random/Adaptive.

## Standard development set

`configs/movingai_devset.json` pins six maps and the `random-1` scenario for each:

- `random-32-32-20`
- `maze-32-32-2`
- `room-32-32-4`
- `warehouse-10-20-10-2-1`
- `warehouse-20-40-10-2-1`
- `den520d`

The map and random-scenario archive SHA256 values are pinned from the
[MovingAI MAPF benchmark](https://movingai.com/benchmarks/mapf/index.html).
Benchmark files stay under ignored `build/`; they are not presented as generated
warehouse data.

```bash
python3 scripts/fetch_movingai_devset.py --output build/movingai-dev
```

## Common runner

```bash
python3 scripts/run_feasibility_benchmark.py \
  --dataset build/movingai-dev \
  --output build/movingai-feasibility \
  --solver both \
  --lns2-binary build/linux/project/lns2_repair \
  --gpbs-binary build/linux/project/gpbs_official \
  --seeds 0,1 --time-limit 60
```

Each solver receives the same map, scenario, prefix agent count, time limit, and
seed. LNS2 uses repair-only PP+SIPPS, neighborhood size 8, and official Adaptive.
GPBS uses its official GPBS solver with SIPPS, target reasoning, induced
constraints, and soft restart enabled.

The runner writes one JSONL row per run and records process timeout/error,
success, native runtime plus preprocessing, wall runtime, SOC, and available
low/high-level counters. GPBS always exits with process code 0, so success is
determined from its CSV `solution cost >= 0`; this prevents timed-out or unsolved
runs from being counted as successes.

A two-run smoke on `random-32-32-20`, 100 agents, seed 0 completed both solvers
successfully. It validates integration only and is not a performance claim.
