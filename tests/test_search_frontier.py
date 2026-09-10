import importlib
from pathlib import Path
import sys
import pytest
from experiments.search_frontier_observation import FrontierProbe, interval_event, profiles
from src.search_frontier.instrument import transform, HOOKS


def test_reversible_transform_and_ambiguous_source_rejected():
    source = (Path(__file__).resolve().parents[1] / 'third_party/mapf_lns2/src/SIPP.cpp').read_text()
    generated = transform(source)
    restored = generated.removeprefix('#include "observer.h"\n')
    for anchor, addition in reversed(HOOKS):
        restored = restored.replace(anchor + addition, anchor)
    assert restored == source
    assert generated.count('frontier_observer::record(2, curr)') == 1
    with pytest.raises(ValueError):
        transform(source + source)


def test_fixed_envelope_and_event_conversion():
    rows = [[1, 3, 0, 4, 2, 2, 7, 2, 1, 1, 0, 0],
            [1, 3, 0, 4, 2, 2, 9, 1, 0, 0, 0, 0],
            [2, 3, 0, 4, 2, 2, 7, 2, 1, 1, 0, 0]]
    p = profiles(dict(events=rows, goal_conflicts=1, goal_cost=10))
    assert len(p['queued_all']) == 2
    assert len(p['queued_envelope']) == len(p['popped_envelope']) == 1
    assert interval_event(rows[1])[-2:] == [0, 1]
    bad = list(rows[0]); bad[7] = 5
    with pytest.raises(ValueError):
        interval_event(bad)


@pytest.fixture
def native(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'build/linux/search-frontier-observer-v1'))
    sys.path.insert(0, str(root / 'build/linux/native-path-probe-v1'))
    module = pytest.importorskip('lns2_search_frontier_native')
    old = pytest.importorskip('lns2_path_probe_native')
    m = tmp_path / 'toy.map'; s = tmp_path / 'toy.scen'
    m.write_text('type octile\nheight 2\nwidth 3\nmap\n...\n...\n')
    s.write_text('version 1\n0\ttoy.map\t3\t2\t0\t0\t2\t0\t2\n0\ttoy.map\t3\t2\t2\t0\t0\t0\t2\n')
    args = (str(m), str(s), [[0, 1, 2], [2, 1, 0]])
    return module, module.NativePathProbe(*args), old.NativePathProbe(*args)


def signature(result):
    return {k: result[k] for k in ('status', 'path', 'cost', 'generated', 'expanded', 'low_level_collisions')}


def test_native_parity_counters_buffer_and_snapshot(native):
    module, probe, old = native
    old.seed_rng(23); expected = old.plan(0, [1], {}, False)
    probe.seed_rng(23); off = probe.plan(0, [1], {}, False)
    fixed = [1]
    wrapped = FrontierProbe(module, probe, {0: [0, 1, 2], 1: [2, 1, 0]}, {1}, True, 32768)
    wrapped.seed_rng(23); on = wrapped.plan(0, fixed, {}, False)
    fixed.clear()
    assert wrapped.captures[0]['fixed'] == [1]
    c = wrapped.captures[0]['capture']
    assert c['popped'] == on['expanded'] and c['goal_cost'] == on['cost']
    assert c['offered'] > 0 and not c['truncated']
    assert signature(expected) == signature(off) == signature(on)
    module.begin_observation(0); probe.seed_rng(23)
    try:
        limited = probe.plan(0, [1], {}, False)
    finally:
        c = module.end_observation()
    assert c['truncated'] and c['events'] == []
    assert signature(limited) == signature(expected)


def test_native_invalid_action_cleans_capture(native):
    module, probe, _ = native
    wrapped = FrontierProbe(module, probe, {0: [0, 1, 2], 1: [2, 1, 0]}, {1}, True, 32768)
    with pytest.raises(ValueError):
        wrapped.plan(99, [1], {}, False)
    module.begin_observation(1); module.end_observation()
    with pytest.raises(ValueError):
        module.begin_observation(32769)
