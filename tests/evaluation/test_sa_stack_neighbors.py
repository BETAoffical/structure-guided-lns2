import pytest
from scripts import probe_sa_stack_neighbors as probe


def test_generated_source_changes_exactly_one_loop():
    path = probe.ROOT / "third_party/mapf_lns2/src/SIPP.cpp"
    old = path.read_text()
    new = probe.transformed(old)
    assert old != new
    assert new.count(probe.REPLACEMENT) == 1
    assert new.replace(probe.REPLACEMENT, probe.PATTERN) == old
    assert old[old.index("Path SIPP::findOptimalPath"):] == new[new.index("Path SIPP::findOptimalPath"):]
    with pytest.raises(ValueError): probe.transformed(new)


@pytest.mark.parametrize("rows,cols", [(1,1),(1,9),(9,1),(3,4),(32,32)])
def test_stack_order_matches_original_grid_filter(rows, cols):
    # Exhaust boundary and obstacle choices; duplicate neighbors on 1-wide grids
    # must be preserved, not deduplicated as an apparent improvement.
    size = rows * cols
    for blocked in (set(), set(range(0, size, 3))):
        for curr in range(size):
            if curr in blocked: continue
            candidates = [curr+1,curr-1,curr+cols,curr-cols]
            def valid(n):
                return 0 <= n < size and n not in blocked and abs(curr//cols-n//cols)+abs(curr%cols-n%cols)<2
            expected = [n for n in candidates if valid(n)]
            buffer = [0]*4
            count = 0
            for n in candidates:
                if valid(n): buffer[count]=n; count+=1
            assert buffer[:count] == expected
