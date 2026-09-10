import hashlib
from unittest.mock import patch

import pytest

from scripts import profile_retained_controller_components as profile


def test_candidate_inputs_exclude_saved_outputs():
    candidate = dict(candidate_id="a", agents=[2, 7], selection_rank_by_family={"target:4": 1},
                     score=12.5, frozen_score=12.5, selected=True, retained=True,
                     feature_out_of_range_fraction=0.5, outcome={"conflicts": 0})
    assert profile.input_candidates([candidate]) == [
        dict(candidate_id="a", agents=[2, 7], selection_rank_by_family={"target:4": 1})
    ]
    assert candidate["score"] == 12.5


def test_changed_input_is_rejected(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("{}")
    with patch.object(profile, "ROOT", tmp_path):
        profile.verify_files({"input.json": hashlib.sha256(b"{}").hexdigest()})
        source.write_text("[]")
        with pytest.raises(ValueError, match="SHA mismatch"):
            profile.verify_files({"input.json": hashlib.sha256(b"{}").hexdigest()})


def test_profile_preserves_return_and_removes_absolute_filenames():
    result, stats = profile.profile_call(lambda: sum(range(5)))
    assert result == 10
    assert stats
    assert all(not profile.Path(row["file"]).is_absolute() for row in stats)


def test_repeats_are_bounded_and_preserve_outputs():
    outputs, measurements = profile.measure(lambda: ("unchanged",))
    assert outputs == [("unchanged",)] * 3
    assert len(measurements["seconds"]) == 3
    assert measurements["median_seconds"] >= 0
