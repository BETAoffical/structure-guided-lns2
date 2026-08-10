# MarginalPool action replay v1 execution amendment

Date: 2026-08-10

This amendment changes only the collection concurrency from 2 workers to 4
workers. The registered scientific protocol remains unchanged: the frozen
states, candidate sets, PP/SIPPS implementation, trial indices, paired PP
seeds, 124-dimensional features, labels, integrity checks, and analysis gates
are identical.

## Reason for the amendment

The decision was made from resource telemetry only, without inspecting repair
outcomes or selecting examples. Ubuntu-22.04 WSL exposes 20 logical CPUs and
had approximately 21 GiB of memory available. Each active replay worker used
approximately one logical CPU and 1.1 GiB RSS, while total host CPU utilization
was approximately 20 percent. Four workers therefore leave substantial CPU and
memory headroom.

## Safe interruption checkpoint

- Source HEAD: `6761dcc52fd8d78cb8c077c85b06dd17c44731d3`
- Run fingerprint: `0d6cec32caca5ea743c46c5d89c2ab8439e350452da68066e6561f0844f4adec`
- Completed states: 6 / 78
- Completed candidates: 173 / 2502
- Completed trials: 2768 / 40032
- Errors: 0
- Timeouts: 0
- Complete state artifacts retained: 6
- Partial state artifacts: 0
- Interrupted active states:
  - `1739b278b808f81c5017bee0e99bd0360cce28e71f69ac518a9f13d19a7e190c`
  - `17f11d995fbd7a99463e4213f820dccc7e1663a0a253e403581dc50587c2d24d`

The two interrupted states have no committed partial artifact and are rerun in
full after resume. Completed state files are immutable and reused.

## Resume command

```text
wsl.exe -d Ubuntu-22.04 --cd "/mnt/c/Users/18448/Documents/lns2 2/structure-guided-lns2" -- /usr/bin/env PYTHONPATH=build/wsl-release /usr/bin/python3 scripts/run_stride_marginalpool_action_replay.py collect --config configs/stride_marginalpool_action_replay_v1_registration.json --output build/stride-marginalpool-action-replay-v1 --mode full --workers 4 --preflight-output build/stride-marginalpool-action-replay-preflight-v1 --resume
```

Worker count affects throughput only and is not an input to a candidate label
or repair result. The original registration specified 2 workers; this document
is the explicit, pre-resume disclosure of that operational deviation. Final
analysis still requires complete state/candidate/trial coverage, strictly
paired seeds, zero errors and timeouts, semantic identity, and artifact hashes.
