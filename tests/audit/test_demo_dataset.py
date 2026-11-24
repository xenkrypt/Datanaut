"""The demo dataset: its planted events exist and the pipeline finds them."""

import importlib.util
import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from datanaut.core.anomalies import stl_anomalies
from datanaut.core.changepoints import detect_change_points
from datanaut.core.characterize import characterize
from datanaut.core.cleaner import OpSpec, propose_fixes
from datanaut.core.loader import load_file
from datanaut.core.profiler import profile
from datanaut.core.timeindex import build_time_index, detect_time_columns
from datanaut.core.validation import RangeRule, rule_action_spec
from datanaut.core.versioning import VersionStore

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "make_demo", ROOT / "scripts" / "make_demo_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # a dataclass in the script looks its module up here
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo(script):
    return script.build()


@pytest.fixture(scope="module")
def pipeline(demo):
    """Load the demo file and clean it the way a user would, then return the frames and results."""
    loaded = load_file(demo.raw.to_csv(index=False).encode("utf-8"), "demo.csv")
    indexed = build_time_index(loaded.df, detect_time_columns(loaded.df)[0].columns)
    prof = profile(indexed.df)
    before = characterize(indexed.df, "power_kw")
    store = VersionStore(indexed.df)
    for proposal in propose_fixes(indexed.df, prof, before):
        if proposal.default_selected:
            store.apply(proposal.spec)
    # the mislabelled rows hold watts: a plausibility rule finds them, then short gaps are filled
    rule = RangeRule(id="plausible", name="a household uses under 50 kW", column="power_kw",
                     min=0.0, max=50.0)  # fmt: skip
    store.apply(rule_action_spec(rule, "set_missing"))
    store.apply(
        OpSpec(
            operation="handle_missing",
            params={"column": "power_kw", "method": "interpolate_time", "max_gap": 6},
        )
    )
    df = store.current()
    char = characterize(df, "power_kw")
    return {"raw": indexed.df, "profile": prof, "df": df, "char": char, "loaded": loaded}


# --- the file and its planted problems -------------------------------------------------------


def test_the_demo_is_about_twelve_months_of_hourly_readings(demo, script):
    assert 8700 <= len(demo.raw) <= 8800
    assert list(demo.raw.columns) == ["Date", "Time", "power_kw", "unit", "temperature_c"]
    kinds = [e["kind"] for e in demo.events]
    assert kinds.count("spike") == 6 and kinds.count("mean_shift") == 1
    assert kinds.count("gap") == 3 and kinds.count("watts_not_kilowatts") == 4
    assert "heatwave" in kinds and "missing_markers" in kinds and "duplicate_rows" in kinds


def test_the_demo_is_seeded_and_changes_with_the_seed(script):
    a, b, c = script.build(7), script.build(7), script.build(8)
    pd.testing.assert_frame_equal(a.raw, b.raw)
    assert a.events == b.events and not a.raw.equals(c.raw)


def test_the_messy_features_really_are_in_the_file(pipeline, demo):
    prof = pipeline["profile"]
    assert prof.duplicates.full_row_duplicates == 12
    assert prof.gaps.gap_count == 3 and prof.gaps.missing_timestamps == 7 + 5 + 9
    raw = pipeline["raw"]
    assert raw["power_kw"].isna().sum() == 25  # the "?" markers
    assert int((raw["power_kw"] > 1000).sum()) == 4  # watts written as if they were kilowatts
    assert set(demo.raw["unit"]) == {"kW", "W"} and (demo.raw["unit"] == "W").sum() == 4
    assert pipeline["loaded"].source_info.format == "csv"


def test_the_excel_variant_holds_the_same_readings(demo):
    buffer = io.BytesIO()
    demo.raw.to_excel(buffer, index=False, sheet_name="readings")
    from_csv = load_file(demo.raw.to_csv(index=False).encode("utf-8"), "d.csv")
    from_xlsx = load_file(buffer.getvalue(), "d.xlsx")
    assert from_xlsx.source_info.format == "xlsx" and from_xlsx.df.shape == from_csv.df.shape
    pd.testing.assert_series_equal(
        from_xlsx.df["power_kw"].astype(float),
        from_csv.df["power_kw"].astype(float),
        check_names=False,
    )


def test_the_script_writes_both_files_and_prints_the_planted_events(
    script, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr("sys.argv", ["make_demo_dataset.py", "--out-dir", str(tmp_path)])
    script.main()
    assert {p.name for p in tmp_path.iterdir()} == {
        "demo_energy_12_months.csv", "demo_energy_12_months.xlsx"}  # fmt: skip
    out = capsys.readouterr().out
    assert "planted events" in out and out.count('"kind": "spike"') == 6
    assert json.loads(out.split("\n")[2].strip())["kind"] == "mean_shift"


# --- the pipeline finds them -----------------------------------------------------------------


def test_the_daily_pattern_is_found_once_the_wrong_unit_rows_are_dealt_with(pipeline):
    assert pipeline["char"].seasonal_period == 24
    assert pipeline["df"]["power_kw"].isna().sum() == 0 and pipeline["df"]["power_kw"].max() < 20


def test_every_planted_spike_is_found_with_about_the_right_size(pipeline, script):
    stl = stl_anomalies(pipeline["df"], "power_kw", pipeline["char"].seasonal_period)
    flagged = {pd.Timestamp(p.timestamp): p for p in stl.points}
    flagged_all = set(stl.flagged_timestamps())
    for when, size in script.SPIKES:
        stamp = pd.Timestamp(when)
        assert stamp in flagged_all, when
        if stamp in flagged:
            assert flagged[stamp].residual == pytest.approx(size, rel=0.15)
    top = [pd.Timestamp(p.timestamp) for p in stl.points[:6]]
    assert set(top) == {pd.Timestamp(w) for w, _ in script.SPIKES}  # they are also the 6 strongest


def test_the_mean_shift_is_found_near_its_date_with_about_the_right_size(pipeline, script):
    cp = detect_change_points(
        pipeline["df"], "power_kw", cost="l2", period=pipeline["char"].seasonal_period
    )
    shift = pd.Timestamp(script.MEAN_SHIFT_AT)
    near = [
        b for b in cp.breaks if abs(pd.Timestamp(b.timestamp) - shift) <= pd.Timedelta(hours=24)
    ]
    assert len(near) == 1 and near[0].direction == "increase" and near[0].kind == "mean"
    assert near[0].mean_change == pytest.approx(script.MEAN_SHIFT_KW, abs=0.15)


def test_the_heatwave_shows_up_as_a_rise_and_a_fall(pipeline, script):
    cp = detect_change_points(
        pipeline["df"], "power_kw", cost="l2", period=pipeline["char"].seasonal_period
    )
    start, end = (pd.Timestamp(d) for d in script.HEATWAVE)
    rise = [
        b for b in cp.breaks if abs(pd.Timestamp(b.timestamp) - start) <= pd.Timedelta(hours=24)
    ]
    fall = [b for b in cp.breaks if abs(pd.Timestamp(b.timestamp) - end) <= pd.Timedelta(hours=24)]
    assert rise and rise[0].direction == "increase" and fall and fall[0].direction == "decrease"
