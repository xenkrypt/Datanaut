import copy
import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from datanaut.core.fingerprint import fingerprint
from datanaut.core.whatif import Tariff, peak_shift
from datanaut.errors import AnalysisError


def days(values_per_day, n_days=3, name="kw"):
    idx = pd.date_range("2024-01-01", periods=24 * n_days, freq="h")
    return pd.DataFrame({name: list(values_per_day) * n_days}, index=idx)


def evening_peak_day():
    return [10.0] * 17 + [50.0, 60.0, 50.0] + [10.0] * 4  # three peak hours, quiet otherwise


def test_energy_is_preserved_and_the_peak_falls():
    df = days(evening_peak_day())
    out = peak_shift(df, "kw", shift_fraction=0.5, percentile=90)
    assert out.total_after == pytest.approx(out.total_before, abs=1e-9)
    assert out.new_series.sum() == pytest.approx(df["kw"].sum(), abs=1e-9)
    assert out.peak_before == 60 and out.peak_after < 60 and out.peak_change < 0
    assert out.peak_change_pct == pytest.approx(100 * out.peak_change / 60)
    assert out.load_factor_after > out.load_factor_before and out.load_factor_change > 0
    assert out.days_affected == 3 and out.energy_not_placed == 0
    assert out.new_series.min() >= 10 - 1e-12 and out.new_series.max() <= out.peak_before


def test_moved_energy_goes_to_the_same_days_quiet_hours_and_each_day_balances():
    out = peak_shift(days(evening_peak_day(), n_days=2), "kw", shift_fraction=1.0, percentile=90)
    day1 = out.new_series.iloc[:24].to_numpy()
    assert day1[17:20].max() <= out.threshold + 1e-9  # peaks come down to the threshold
    quiet = np.r_[day1[:17], day1[20:]]
    assert quiet.min() >= 10 and quiet.max() <= out.threshold + 1e-9
    assert day1.sum() == pytest.approx(sum(evening_peak_day()))


def test_a_zero_fraction_returns_identical_data():
    df = days(evening_peak_day())
    out = peak_shift(df, "kw", shift_fraction=0.0)
    pd.testing.assert_series_equal(out.new_series, df["kw"], check_names=False)
    assert out.energy_moved == 0 and out.peak_change == 0 and out.days_affected == 0


def test_limited_room_moves_only_what_fits_and_keeps_total_energy():
    flat_high = [95.0] * 23 + [100.0]
    out = peak_shift(days(flat_high, n_days=2), "kw", shift_fraction=1.0, percentile=40)
    assert out.total_after == pytest.approx(out.total_before)
    assert out.energy_not_placed >= 0 and out.new_series.max() <= 100


def test_missing_values_are_left_alone_and_counted():
    df = days(evening_peak_day())
    df.iloc[[3, 30], 0] = np.nan
    out = peak_shift(df, "kw", 0.5, 90)
    assert out.excluded_missing == 2
    assert out.new_series.isna().tolist() == df["kw"].isna().tolist()
    assert out.total_after == pytest.approx(out.total_before)


def test_no_cost_without_a_tariff_and_an_exact_cost_with_one():
    df = days(evening_peak_day())
    assert peak_shift(df, "kw", 0.5, 90).cost is None
    out = peak_shift(
        df, "kw", 0.5, 90, tariff=Tariff(peak_rate=0.30, offpeak_rate=0.10, currency="EUR")
    )
    assert out.cost.currency == "EUR" and out.cost.energy_moved == pytest.approx(out.energy_moved)
    assert out.cost.estimated_saving == pytest.approx(out.energy_moved * 0.20)
    as_dict = peak_shift(df, "kw", 0.5, 90, tariff={"peak_rate": 0.3, "offpeak_rate": 0.1})
    assert as_dict.cost.estimated_saving == pytest.approx(out.cost.estimated_saving)
    assert "never assumed" in " ".join(out.assumptions) and out.label == "scenario estimate"
    with pytest.raises(Exception, match="peak_rate"):
        peak_shift(df, "kw", 0.5, 90, tariff={"offpeak_rate": 0.1})


def test_energy_for_the_cost_scales_with_the_interval_length():
    idx = pd.date_range("2024-01-01", periods=96, freq="30min")
    df = pd.DataFrame({"kw": ([10.0] * 34 + [60.0, 70.0] + [10.0] * 12) * 2}, index=idx)
    out = peak_shift(df, "kw", 1.0, 90, tariff=Tariff(peak_rate=1.0, offpeak_rate=0.0))
    assert out.cost.energy_moved == pytest.approx(out.energy_moved * 0.5)


def test_parameter_validation():
    df = days(evening_peak_day())
    for kwargs, message in (
        ({"shift_fraction": 1.5}, "between 0 and 1"),
        ({"shift_fraction": 0.5, "percentile": 100}, "between 0 and 100"),
        ({"shift_fraction": 0.5, "window": "same_week"}, "same_day"),
    ):
        with pytest.raises(AnalysisError, match=message):
            peak_shift(df, "kw", **kwargs)
    with pytest.raises(AnalysisError, match="too few usable values"):
        peak_shift(df.assign(kw=np.nan), "kw", 0.5)


def test_evidence_log_json_and_the_input_is_untouched():
    df = days(evening_peak_day())
    snapshot, before = copy.deepcopy(df), fingerprint(df)
    out = peak_shift(df, "kw", 0.5, 90)
    pd.testing.assert_frame_equal(df, snapshot)
    assert fingerprint(df) == before
    assert out.evidence.method == "peak_shift" and out.evidence.n_observations == len(df)
    assert out.evidence.parameters == {
        "shift_fraction": 0.5,
        "percentile": 90,
        "window": "same_day",
    }
    assert out.log_entry.read_only and "Scenario estimate" in out.log_entry.summary
    assert "new_series" not in json.loads(out.model_dump_json())


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.floats(0, 1000, allow_nan=False), min_size=48, max_size=96),
    st.floats(0, 1),
    st.floats(50, 99),
)
def test_property_energy_preserved_and_the_peak_never_rises(values, fraction, percentile):
    n = (len(values) // 24) * 24
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame({"kw": values[:n]}, index=idx)
    out = peak_shift(df, "kw", fraction, percentile)
    assert out.new_series.sum() == pytest.approx(df["kw"].sum(), rel=1e-9, abs=1e-6)
    assert out.new_series.max() <= df["kw"].max() + 1e-9
    assert out.peak_change <= 1e-9
    assert out.energy_moved >= 0 and out.energy_not_placed >= -1e-9
    if fraction == 0:
        assert np.array_equal(out.new_series.to_numpy(), df["kw"].to_numpy())
