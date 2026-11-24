import json

import pytest
from pydantic import ValidationError

from datanaut.core.log import PipelineLog


def test_step_ids_sequence_from_one():
    log = PipelineLog()
    ids = [log.append("op", f"step {i}").step_id for i in range(4)]
    assert ids == [1, 2, 3, 4]
    assert [e.step_id for e in log.entries] == ids
    assert len(log) == 4


def test_json_round_trip():
    log = PipelineLog()
    log.append(
        "load",
        "Loaded 10 rows.",
        params={"delimiter": ";", "sheet": None},
        counts={"rows": 10, "converted": 3.5},
        rows_before=0,
        rows_after=10,
    )
    log.append("profile", "Profiled.", read_only=True)
    restored = PipelineLog.from_json(log.to_json())
    assert restored.entries == log.entries
    assert restored.metadata == log.metadata
    assert restored.to_json() == log.to_json()


def test_metadata_has_versions():
    meta = PipelineLog().metadata
    assert set(meta) == {"datanaut", "python", "pandas", "numpy"}
    assert all(meta.values())


def test_created_at_is_utc_iso_and_json_valid():
    log = PipelineLog()
    entry = log.append("x", "y")
    assert entry.created_at.endswith("+00:00")
    data = json.loads(log.to_json())
    assert data["entries"][0]["operation"] == "x"


def test_entries_is_a_copy_and_entries_are_frozen():
    log = PipelineLog()
    log.append("x", "y")
    log.entries.clear()
    assert len(log) == 1
    with pytest.raises(ValidationError):
        log.entries[0].operation = "z"


def test_extend_renumbers():
    src = PipelineLog()
    src.append("a", "A")
    src.append("b", "B")
    dst = PipelineLog()
    dst.append("first", "F")
    dst.extend(src.entries)
    assert [e.step_id for e in dst] == [1, 2, 3]
    assert [e.operation for e in dst] == ["first", "a", "b"]


def test_extend_preserves_original_timestamps():
    from datanaut.core.log import StepLog

    old = StepLog(step_id=7, operation="load", summary="x", created_at="2020-01-01T00:00:00+00:00")
    log = PipelineLog()
    log.append("first", "F")
    (added,) = log.extend([old])
    assert added.step_id == 2
    assert added.created_at == "2020-01-01T00:00:00+00:00"
    assert log.entries[1] == added
