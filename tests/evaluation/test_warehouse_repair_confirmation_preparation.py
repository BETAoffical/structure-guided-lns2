import copy
from collections import Counter

import pytest

from scripts import prepare_warehouse_repair_confirmation as prep


def inputs():
    config = prep.read_json(prep.ROOT / prep.CONFIG)
    return config, prep.read_json(prep.ROOT / config["base_dataset_config"])


def test_derivation_preserves_map_and_od_semantics():
    config, base = inputs()
    original = copy.deepcopy(base)
    result = prep.dataset_config(config, base)
    assert base == original
    assert result["map"] == base["map"]
    assert result["task"] == base["task"]
    assert len(result["task_variants"]) == 2
    assert result["task_variants"][0]["task"] == result["task_variants"][1]["task"] == base["task_variants"][2]["task"]


def test_schedule_paired_seeds_and_bounded_orders():
    config, _ = inputs()
    rows = prep.schedule_slots(config)
    assert len(rows) == 96
    for key in range(32):
        group = [r for r in rows if r["checkpoint_slot"] == key]
        assert len(group) == 3
        assert {r["controller"] for r in group} == set(config["controllers"])
        assert len({r["solver_seed"] for r in group}) == 1
        assert len({r["selection_seed"] for r in group}) == 1
        assert not any(r["bound_to_checkpoint"] for r in group)
    orders = Counter(tuple(r["controller"] for r in rows[i:i + 3]) for i in range(0, 96, 3))
    assert len(orders) == 6 and max(orders.values()) - min(orders.values()) == 1


@pytest.mark.parametrize("section,key,value", [
    (None, "timed_execution_authorized", True),
    ("checkpoint", "replace_failed_candidates", True),
    ("runtime", "max_repair_iterations", 100),
    ("runtime", "timed_workers", 20),
    (None, "no_model_selection_from_confirmation", False),
])
def test_preparation_rejects_scope_expansion(section, key, value):
    config, _ = inputs()
    prep.validate(config)
    (config if section is None else config[section])[key] = value
    with pytest.raises(ValueError):
        prep.validate(config)
