from pathlib import Path
import importlib
import sys
import pytest

from experiments.search_occupancy_observation import OccupancyIndex, summarize_contacts, rank_contacts, ObservingProbe


def row(src,dst,t,v=1,e=0):
    return [0,src,dst,t,t+1,t,t+1,t+1,v,e]


def test_vertex_swap_terminal_and_noncontinuous_ids():
    i=OccupancyIndex({7:[2,1],19:[3,2,1]})
    assert i.resolve(row(0,1,9))=={'vertex':{7,19}}
    assert i.resolve(row(1,2,1,0,1))=={'edge':{7}}
    assert i.resolve(row(0,2,0))=={'vertex':{7}}


def test_internal_and_returned_contacts_excluded_and_duplicates_deduped():
    s=summarize_contacts(2,{7:[2,1],19:[3,2,1]},[0,0],
                         [row(0,1,1)]*3+[row(0,2,1)],{7})
    assert len(s['contacts'][7])==1 and 19 not in s['contacts']
    assert rank_contacts([s,s])==[dict(agent=7,planner_count=1,contact_count=1)]
    assert not summarize_contacts(2,{7:[2,1]},[0,1],[row(0,1,1)],{7})['contacts']


def test_unknown_owner_is_reported():
    s=summarize_contacts(2,{7:[2,1]},[0,0],[row(0,3,1)],{7})
    assert len(s['unmatched'])==1 and not s['contacts']


def test_rank_tie_is_agent_id_not_insertion_order():
    s=dict(contacts={19:[(2,'vertex',0,1,1)],7:[(2,'vertex',0,1,1)]})
    assert [r['agent'] for r in rank_contacts([s])]==[7,19]


@pytest.fixture
def native(tmp_path):
    root=Path(__file__).resolve().parents[1]
    path=root/'build/linux/search-occupancy-observer-v1'
    sys.path.insert(0,str(path))
    module=pytest.importorskip('lns2_search_observer_native')
    oldpath=root/'build/linux/native-path-probe-v1'; sys.path.insert(0,str(oldpath))
    old=pytest.importorskip('lns2_path_probe_native')
    mapfile=tmp_path/'toy.map'; scen=tmp_path/'toy.scen'
    mapfile.write_text('type octile\nheight 2\nwidth 3\nmap\n...\n...\n')
    scen.write_text('version 1\n0\ttoy.map\t3\t2\t0\t0\t2\t0\t2\n0\ttoy.map\t3\t2\t2\t0\t0\t0\t2\n')
    args=(str(mapfile),str(scen),[[0,1,2],[2,1,0]])
    return module,module.NativePathProbe(*args),old.NativePathProbe(*args)


def signature(r):
    return {k:r[k] for k in ('status','path','cost','generated','expanded','low_level_collisions')}


def test_native_on_off_frozen_and_bounded_capture(native):
    module,probe,old=native
    old.seed_rng(23); expected=old.plan(0,[1],{},False)
    probe.seed_rng(23); disabled=probe.plan(0,[1],{},False)
    module.begin_observation(65536)
    probe.seed_rng(23)
    try: observed=probe.plan(0,[1],{},False)
    finally: capture=module.end_observation()
    assert signature(expected)==signature(disabled)==signature(observed)
    assert capture['offered'] and not capture['truncated'] and capture['queries']
    module.begin_observation(0)
    probe.seed_rng(23)
    try: truncated=probe.plan(0,[1],{},False)
    finally: c=module.end_observation()
    assert c['truncated'] and c['events']==[]
    assert signature(truncated)==signature(expected)


def test_capture_cleanup_on_exception_and_maximum(native):
    module,probe,_=native
    wrapped=ObservingProbe(module,probe,{0:[0,1,2],1:[2,1,0]},{1},True)
    with pytest.raises(ValueError): wrapped.plan(99,[1],{},False)
    module.begin_observation(1); module.end_observation()
    with pytest.raises(ValueError): module.begin_observation(65537)
