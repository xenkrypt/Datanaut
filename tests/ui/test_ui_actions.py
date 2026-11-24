import pytest

from datanaut.errors import FileTooLargeError, ParseError, UnsupportedFormatError
from datanaut.ui import actions, display
from datanaut.ui.limits import MAX_COLUMNS, check_dimensions
from datanaut.ui.state import WizardState

AMBIGUOUS = b"when,v\n01/02/2024,1\n03/04/2024,2\n05/06/2024,3\n07/08/2024,4\n"
ISO = b"when,v,w\n2024-01-01 00:00,1,5\n2024-01-01 01:00,2,6\n2024-01-01 02:00,3,7\n"


@pytest.fixture
def sample() -> WizardState:
    s = WizardState()
    assert actions.load_sample(s)
    return s


def test_sample_loads_once_and_a_repeat_is_a_no_op(sample):
    assert "Upload" in sample.completed and sample.loaded.source_info.filename
    before = len(sample.log)
    assert actions.load_sample(sample) is False
    assert len(sample.log) == before and sample.log.entries[0].operation == "load"


def test_confirming_the_mapping_builds_v1_and_logs_the_index(sample):
    time_cols, values = actions.suggest_mapping(sample.loaded.df)
    assert time_cols == ["Date", "Time"] and "global_active_power" in values
    created = actions.confirm_mapping(sample, time_cols, None, values[:2], {values[0]: True})
    assert created and sample.store.head == "V1" and "Map columns" in sample.completed
    ops = [e.operation for e in sample.log]
    assert (
        ops[0] == "load"
        and "profile" in ops
        and ops.index("profile") < ops.index("build_time_index")
    )
    assert sample.consumption == {values[0]: True, values[1]: False}
    assert sample.profile.sampling is not None  # re-profiled on the indexed data, not logged twice
    assert ops.count("profile") == 1


def test_changing_only_the_value_columns_keeps_the_versions(sample):
    time_cols, values = actions.suggest_mapping(sample.loaded.df)
    actions.confirm_mapping(sample, time_cols, None, values[:2], {})
    store = sample.store
    assert actions.confirm_mapping(sample, time_cols, None, values[:1], {values[0]: True}) is False
    assert sample.store is store and sample.value_columns == values[:1]


def test_changing_the_time_column_rebuilds_with_a_message():
    s = WizardState()
    actions.load_upload(s, b"a,b,v\n2024-01-01,2024-02-01,1\n2024-01-02,2024-02-02,2\n", "x.csv")
    actions.confirm_mapping(s, ["a"], None, ["v"], {})
    log_len = len(s.log)
    assert actions.confirm_mapping(s, ["b"], None, ["v"], {}) is True
    assert s.time_columns == ["b"] and len(s.log) == log_len  # the old index entry was replaced
    assert [m.level for m in s.messages] == ["warning"]


def test_ambiguous_dates_must_be_chosen_before_confirming():
    s = WizardState()
    actions.load_upload(s, AMBIGUOUS, "a.csv")
    auto = actions.preview_time_index(s, ["when"], None)
    assert auto.flags.ambiguous and auto.flags.ambiguous_preview
    with pytest.raises(ParseError, match="day-first or month-first"):
        actions.confirm_mapping(s, ["when"], None, ["v"], {})
    assert s.store is None
    assert actions.confirm_mapping(s, ["when"], True, ["v"], {})
    assert s.indexed.df.index[0].month == 2  # day-first: 01/02 is 1 February


@pytest.mark.parametrize(
    ("time_cols", "values", "text"),
    [([], ["v"], "one timestamp"), (["when"], [], "at least one"), (["when"], ["when"], "numeric")],
)
def test_bad_mappings_are_refused(time_cols, values, text):
    s = WizardState()
    actions.load_upload(s, ISO, "a.csv")
    with pytest.raises(ParseError, match=text):
        actions.confirm_mapping(s, time_cols, None, values, {})
    assert s.store is None


def test_the_preview_is_cached_and_does_not_change_the_state(sample):
    first = actions.preview_time_index(sample, ["Date", "Time"], None)
    assert actions.preview_time_index(sample, ["Date", "Time"], None) is first
    assert sample.store is None


def test_errors_become_friendly_messages_without_tracebacks():
    s = WizardState()
    assert actions.guard(s, "Upload", actions.load_upload, s, b"", "e.csv") is None
    assert actions.guard(s, "Upload", actions.load_upload, s, b"%PDF-1.4 ...", "x.pdf") is None
    assert actions.guard(s, "Profile", lambda: 1 / 0) is None
    texts = [m.text for m in s.messages]
    assert all(m.level == "error" for m in s.messages) and len(texts) == 3
    assert "Traceback" not in " ".join(texts) and "division" not in " ".join(texts)
    assert "Upload" in texts[0] and "Profile" in texts[2]
    assert "CSV or Excel" in actions.friendly_error(UnsupportedFormatError(), "Upload")
    assert "smaller" in actions.friendly_error(FileTooLargeError(), "Upload")


def test_limits_are_enforced_in_code(monkeypatch):
    import pandas as pd

    with pytest.raises(FileTooLargeError, match="columns"):
        check_dimensions(pd.DataFrame([[0] * (MAX_COLUMNS + 1)]))
    monkeypatch.setattr(actions, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(FileTooLargeError, match="limit"):
        actions.load_upload(WizardState(), b"x" * 11, "a.csv")


def test_display_rows_reshape_the_core_results(sample):
    assert display.format_bytes(0) == "0 B" and display.format_bytes(1536) == "1.5 KB"
    time_cols, values = actions.suggest_mapping(sample.loaded.df)
    actions.confirm_mapping(sample, time_cols, None, values, {})
    rows = display.version_rows(sample.store)
    assert [r["id"] for r in rows] == ["V1"] and rows[0]["is_head"]
    log = display.log_rows(sample.log)
    assert [r["step"] for r in log] == list(range(1, len(log) + 1))
    cards = dict(display.profile_cards(sample.profile))
    assert cards["Rows"] == f"{sample.profile.rows:,}" and "%" in cards["Missing values"]
    assert list(display.column_table(sample.profile).columns)[:2] == ["Column", "Type"]
    assert list(display.gap_table(sample.profile).columns) == ["From", "To", "Missing timestamps"]
