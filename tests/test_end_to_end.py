import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datanaut.core.characterize import characterize
from datanaut.core.fingerprint import fingerprint
from datanaut.core.loader import load_file
from datanaut.core.log import PipelineLog
from datanaut.core.profiler import profile
from datanaut.core.timeindex import (
    build_time_index,
    detect_time_columns,
    detect_value_columns,
)

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_OPERATIONS = [
    "load",
    "normalize_missing",
    "convert_numeric",
    "build_time_index",
    "profile",
    "characterize",
]


def run_pipeline(source, filename=None):
    """Load -> time index -> profile -> characterize, collecting one pipeline log."""
    loaded = load_file(source, filename=filename)
    cands = detect_time_columns(loaded.df)
    indexed = build_time_index(loaded.df, cands[0].columns)
    prof = profile(indexed.df)
    value_col = detect_value_columns(indexed.df)[0]
    char = characterize(indexed.df, value_col)
    log = PipelineLog()
    log.extend(loaded.log)
    log.extend(indexed.log)
    log.extend([prof.log_entry, char.log_entry])
    return loaded, indexed, prof, char, log


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "make_sample", ROOT / "scripts" / "make_sample.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_variant_runs_end_to_end(variant, tmp_path):
    path = tmp_path / variant.filename
    path.write_bytes(variant.data)
    digest_before = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded, indexed, prof, char, log = run_pipeline(path)

    # the input file is never modified
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest_before
    assert path.read_bytes() == variant.data

    ops = [e.operation for e in log]
    assert ops[:3] == EXPECTED_OPERATIONS[:3]  # loader entries come first
    for operation in EXPECTED_OPERATIONS:
        assert operation in ops, f"missing log entry for {operation}"
    assert [e.step_id for e in log] == list(range(1, len(log) + 1))
    assert ops[-2:] == ["profile", "characterize"]
    assert all(e.read_only for e in log if e.operation in ("profile", "characterize"))

    # the log is a real, exportable record
    restored = PipelineLog.from_json(log.to_json())
    assert restored.entries == log.entries
    assert set(restored.metadata) == {"datanaut", "python", "pandas", "numpy"}

    # a clean time index, a quality summary and a characterization came out
    assert (
        isinstance(indexed.df.index, pd.DatetimeIndex) and indexed.df.index.is_monotonic_increasing
    )
    assert prof.rows == variant.expect["rows"]
    assert prof.sampling.median_interval_seconds == 3600.0
    assert char.frequency_label == "hourly" and char.seasonal_period == 24


def test_bytes_and_filelike_sources_leave_the_input_untouched(variants):
    data = variants["xlsx"].data
    copy = bytes(data)
    run_pipeline(data, filename="x.xlsx")
    assert data == copy


def test_pipeline_is_deterministic(variants):
    a = run_pipeline(variants["question_marks"].data)
    b = run_pipeline(variants["question_marks"].data)
    assert fingerprint(a[1].df) == fingerprint(b[1].df)

    def stable(log):
        return [e.model_dump(exclude={"created_at"}) for e in log]

    assert stable(a[4]) == stable(b[4])
    assert a[3].model_dump(exclude={"log_entry"}) == b[3].model_dump(exclude={"log_entry"})


def test_sample_script_is_deterministic_and_matches_the_committed_file():
    script = _load_script()
    first = script.generate()
    assert first == script.generate()
    assert first != script.generate(seed=1)
    committed = ROOT / "sample_data" / "household_energy_sample.csv"
    assert committed.read_bytes() == first
    assert 2000 < len(first.splitlines()) < 5000


def test_sample_script_writes_where_asked(tmp_path):
    script = _load_script()
    out = tmp_path / "nested" / "s.csv"
    assert script.main(["--out", str(out), "--days", "10"]) == 0
    assert out.read_bytes() == script.generate(days=10)


@pytest.fixture(scope="module")
def sample_run():
    return run_pipeline(ROOT / "sample_data" / "household_energy_sample.csv")


def test_sample_dataset_profile_shows_the_planted_mess(sample_run):
    loaded, indexed, prof, char, log = sample_run
    assert loaded.source_info.delimiter == "," and loaded.source_info.encoding == "utf-8"
    assert indexed.choices.time_columns == ["Date", "Time"]
    assert indexed.choices.dayfirst is True and indexed.choices.dayfirst_source == "auto"
    assert indexed.flags.rows_moved_by_sort > 0 and indexed.flags.ambiguous is False
    assert prof.duplicates.full_row_duplicates == 6
    assert prof.duplicates.duplicate_timestamps_conflicting == 0
    assert prof.gaps.missing_timestamps == 10 and prof.gaps.longest_gap_timestamps == 10
    power = prof.column("global_active_power")
    assert power.longest_missing_run >= 8 and power.missing_count > 50
    assert prof.column("global_active_power").min < 0  # impossible negatives are present
    assert indexed.df.index.min() == pd.Timestamp("2024-01-01")
    assert len(loaded.df) == len(indexed.df) + len(indexed.rejected)
    assert char.seasonal_period == 24 and char.seasonal_strength > 0.4
    assert np.isfinite(char.stationarity.p_value)
