from experiments.search_occupancy_snapshot import SnapshotObservingProbe


class Module:
    def begin_observation(self, limit):
        pass

    def end_observation(self):
        return dict(events=[], offered=0, queries=0, truncated=False)


class Probe:
    def plan(self, *args, **kwargs):
        return dict(status='path', path=[0, 1])


def test_fixed_lists_are_independent_per_call_and_external_input_is_not_modified():
    wrapped = SnapshotObservingProbe(Module(), Probe(), {0: [0, 1], 7: [2, 3]}, {7}, True)
    fixed = [7]
    wrapped.plan(0, fixed, {}, False)
    assert fixed == [7]
    fixed.append(0)
    wrapped.plan(0, fixed, {}, False)
    fixed.clear()
    assert wrapped.captures[0]['fixed'] == [7]
    assert wrapped.captures[1]['fixed'] == [7, 0]


def test_disabled_recorder_remains_empty():
    wrapped = SnapshotObservingProbe(Module(), Probe(), {0: [0, 1]}, set(), False)
    assert wrapped.plan(0, [], {}, False)['status'] == 'path'
    assert wrapped.captures == []
