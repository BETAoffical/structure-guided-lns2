"""Run the v1 observation protocol with independent per-search list snapshots."""
import contextlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, write_json
from experiments.local_path_compatibility import sha256_file
from experiments.search_occupancy_snapshot import SnapshotObservingProbe
from scripts import diagnose_search_occupancy as runner


def configure():
    runner.OUT = ROOT / 'build/initlns-search-occupancy-observer-v2'
    runner.REG = ROOT / 'artifacts/initlns-search-occupancy-observer-v2/registration.json'
    runner.ObservingProbe = SnapshotObservingProbe
    runner.__file__ = __file__


def main():
    configure()
    if len(sys.argv) > 1 and sys.argv[1] == 'prepare':
        with contextlib.redirect_stdout(io.StringIO()):
            runner.prepare()
        path = runner.OUT / 'plan.json'
        plan = read_json(path)
        plan['schema'] = 'lns2.search_occupancy.v2'
        plan['recorder_fix'] = 'Copy fixed-agent list at each call; no search or cap change.'
        plan['predecessor_plan_sha256'] = sha256_file(ROOT / 'build/initlns-search-occupancy-observer-v1/plan.json')
        for name in ('experiments/search_occupancy_snapshot.py', 'scripts/diagnose_search_occupancy_snapshot.py',
                     'tests/test_search_occupancy_snapshot.py', 'docs/SEARCH_OCCUPANCY_SNAPSHOT_FIX_ZH.md'):
            plan['files'][name] = sha256_file(ROOT / name)
        write_json(path, plan)
        print(json.dumps(dict(plan_sha256=sha256_file(path), jobs=len(plan['jobs']),
                              event_cap=plan['event_cap'], module_sha256=plan['files'][plan['module_file']])) )
    else:
        runner.main()


if __name__ == '__main__':
    main()
