import pytest

from experiments.diagnostic_integrity import pair_absent,snapshot_sources,verify_inputs
from experiments._common import sha256_file


def test_pair_membership_is_identical_for_native_tuples_and_json_lists():
    for edges in ([[233,302]],[(233,302)],[[302,233]],[(302,233)]):
        assert not pair_absent(edges,[233,302])
    assert pair_absent([(1,2)],[233,302])


def test_archive_only_authorizes_evidence_reading(tmp_path):
    source=tmp_path/"runner.py"
    source.write_text("old_source")
    inputs={"runner.py":sha256_file(source)}
    folder=tmp_path/"registered_sources"
    snapshot_sources(tmp_path,inputs,folder)
    source.write_text("new_source")
    with pytest.raises(ValueError,match="registered input changed"):
        verify_inputs(tmp_path,inputs)
    assert verify_inputs(tmp_path,inputs,folder)==["runner.py"]
    next(folder.iterdir()).write_text("tampered")
    with pytest.raises(ValueError): verify_inputs(tmp_path,inputs,folder)


def test_snapshot_cannot_mask_changed_native_or_data(tmp_path):
    for name in ("native.so","data.json","grid.map"):
        path=tmp_path/name
        path.write_text("original")
        inputs={name:sha256_file(path)}
        snapshot_sources(tmp_path,inputs,tmp_path/"snapshots")
        path.write_text("changed")
        with pytest.raises(ValueError): verify_inputs(tmp_path,inputs,tmp_path/"snapshots")
