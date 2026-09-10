import json

import pytest

from generators.io import map_document, task_document, write_json, write_movingai_map, write_movingai_scen
from generators.models import MapData, TaskData
from scripts import generate_warehouse_repair_confirmation as prep


def test_geometry_normalizes_line_endings_but_not_obstacles():
    text = "type octile\nheight 2\nwidth 3\nmap\n...\n.@.\n"
    assert prep.geometry_hash(text) == prep.geometry_hash(text.replace("\n", "\r\n"))
    assert prep.geometry_hash(text) != prep.geometry_hash(text.replace(".@.", "..."))
    with pytest.raises(ValueError):
        prep.geometry_hash(text.replace("width 3", "width 4"))


def test_seed_inventory_and_schedule_are_deterministic():
    config = {"master_seed": 2026102101}
    schedule = prep.assert_disjoint(config, [])
    assert len(schedule) == 8
    assert schedule == prep.seed_schedule(config)
    with pytest.raises(ValueError, match="collision"):
        prep.assert_disjoint(config, [{"seeds": [schedule[3]["task_seeds"][0]]}])
    assert prep.seeds_in({"nested": [{"task_seeds": [1, 2]}, {"master_seed": 3}], "solver_seed": 4}) == {1, 2, 3}


def test_inventory_does_not_read_labels_or_new_cohort(tmp_path):
    names = ["build/old/maps/a.map", "build/old/instances/a.json", "build/old/manifest.jsonl",
             "build/old/outcomes/outcome.json", "build/old/episodes/trace.jsonl",
             "build/venv-graph/maps/a.map", "build/warehouse-repair-confirmation-v1/dataset/a.map",
             "configs/old.json"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    found = {p.relative_to(tmp_path).as_posix() for p in prep.input_paths(tmp_path)}
    assert found == set(names[:3] + names[-1:])


def test_sidecar_seed_is_included(tmp_path):
    path = tmp_path / "task.json"
    path.write_text(json.dumps({"task_id": "task", "seed": 85}))
    assert prep.inspect_input(path, tmp_path)["seeds"] == [85]


@pytest.fixture
def bundle(tmp_path):
    metadata = {"service_cells": [], "stations": [], "semantic_cell_types": ["............"] * 2}
    m = MapData("tiny", 1, ["............"] * 2, metadata)
    t = TaskData("tiny_task", "tiny", 2, [(0, 0), (0, 1), (1, 0), (1, 1)],
                 [(0, 8), (0, 9), (1, 8), (1, 9)], {})
    write_json(tmp_path / "maps/tiny.json", map_document(m))
    write_json(tmp_path / "instances/tiny_task.json", task_document(t))
    write_movingai_map(tmp_path / "maps/tiny.map", m)
    write_movingai_scen(tmp_path / "instances/tiny_task.scen", m, t)
    row = dict(map_metadata_file="maps/tiny.json", task_file="instances/tiny_task.json",
               map_file="maps/tiny.map", scenario_file="instances/tiny_task.scen",
               map_id="tiny", task_id="tiny_task", map_seed=1, task_seed=2, agent_count=4)
    return tmp_path, row


def test_persisted_bundle_validation(bundle):
    root, row = bundle
    result = prep.verify_task(row, root)
    assert result["agents"] == 4 and result["shortest_distance_min"] == 8


@pytest.mark.parametrize("tamper", ["seed", "scenario", "map"])
def test_bundle_tampering_is_rejected(bundle, tamper):
    root, row = bundle
    if tamper == "seed":
        row["task_seed"] = 3
    elif tamper == "scenario":
        path = root / row["scenario_file"]
        path.write_text(path.read_text().replace("\t8\n", "\t9\n"))
    else:
        path = root / row["map_file"]
        path.write_text(path.read_text().replace("............", "@...........", 1))
    with pytest.raises(ValueError):
        prep.verify_task(row, root)
