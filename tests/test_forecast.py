import copy
import json

import numpy as np
import pandas as pd
import pytest

import datanaut.core.forecast as fc
from datanaut.core.fingerprint import fingerprint
from datanaut.core.forecast import backtest, forecast, recommend_model
from datanaut.errors import AnalysisError


def periodic(period=24, cycles=14, noise=0.0, seed=0, trend=0.0):
    t = np.arange(period * cycles)
    rng = np.random.default_rng(seed)
    values = 50 + 10 * np.sin(2 * np.pi * t / period) + trend * t + rng.normal(0, noise, len(t))
    return pd.DataFrame({"v": values}, index=pd.date_range("2024-01-01", periods=len(t), freq="h"))


def test_seasonal_naive_is_exact_on_a_perfectly_periodic_series():
    bt = backtest(periodic(), "v", ["seasonal_naive"], horizon=24, n_folds=3, period=24)
    (score,) = bt.scores
    assert score.mae < 1e-9 and score.rmse < 1e-9 and score.smape < 1e-9
    assert [f.mae for f in score.folds] == pytest.approx([0, 0, 0], abs=1e-9)


def test_folds_are_strictly_chronological_and_never_leak_test_data():
    df = periodic(noise=1.0)
    bt = backtest(df, "v", ["seasonal_naive", "holt_winters"], horizon=24, n_folds=3, period=24)
    assert [f.fold for f in bt.folds] == [1, 2, 3]
    for f in bt.folds:
        assert pd.Timestamp(f.train_end) < pd.Timestamp(f.test_start) <= pd.Timestamp(f.test_end)
    starts = [pd.Timestamp(f.test_start) for f in bt.folds]
    assert starts == sorted(starts)
    assert pd.Timestamp(bt.folds[0].test_end) + pd.Timedelta(hours=1) == starts[1]
    assert pd.Timestamp(bt.folds[-1].test_end) == df.index[-1]
    assert [f.n_train for f in bt.folds] == [len(df) - 72, len(df) - 48, len(df) - 24]


def test_a_model_is_only_ever_given_the_data_before_its_test_window(monkeypatch):
    seen = []
    real = fc._fit_predict

    def spy(name, train, horizon, period, **kw):
        seen.append(len(train))
        return real(name, train, horizon, period, **kw)

    monkeypatch.setattr(fc, "_fit_predict", spy)
    df = periodic()
    backtest(df, "v", ["seasonal_naive"], horizon=24, n_folds=3, period=24)
    assert seen == [len(df) - 72, len(df) - 48, len(df) - 24]


def test_metrics_are_computed_correctly_on_a_known_case():
    df = pd.DataFrame(
        {"v": [10.0] * 30 + [20.0] * 6}, index=pd.date_range("2024-01-01", periods=36, freq="h")
    )
    (score,) = backtest(df, "v", ["seasonal_naive"], horizon=6, n_folds=1, period=None).scores
    assert score.mae == 10 and score.rmse == 10  # forecast 10 against actual 20
    assert score.smape == pytest.approx(100 * 2 * 10 / 30)


def test_a_failing_model_is_reported_and_the_others_still_run(monkeypatch):
    real = fc._fit_predict

    def flaky(name, train, horizon, period, **kw):
        if name == "holt_winters":
            raise RuntimeError("optimizer exploded")
        return real(name, train, horizon, period, **kw)

    monkeypatch.setattr(fc, "_fit_predict", flaky)
    bt = backtest(
        periodic(noise=0.5), "v", ["holt_winters", "seasonal_naive"], horizon=24, period=24
    )
    assert [s.model for s in bt.scores] == ["seasonal_naive"]
    assert len(bt.failures) == 1 and bt.failures[0].model == "holt_winters"
    assert "optimizer exploded" in bt.failures[0].reason and not bt.failures[0].skipped
    assert bt.log_entry.counts["models_failed"] == 1


def test_sarima_is_skipped_with_advice_when_too_long_and_runs_when_short():
    df = periodic(period=6, cycles=20, noise=0.3)
    skipped = backtest(
        df, "v", ["sarima", "seasonal_naive"], horizon=6, n_folds=2, period=6, max_sarima_points=50
    )
    assert skipped.failures[0].skipped
    assert "Aggregate it to a coarser frequency" in skipped.failures[0].reason
    assert [s.model for s in skipped.scores] == ["seasonal_naive"]
    ran = backtest(df, "v", ["sarima"], horizon=6, n_folds=2, period=6)
    assert ran.scores and ran.scores[0].mae < 3 and ran.scores[0].fit_seconds > 0


def test_models_accept_data_with_gaps_and_missing_values():
    df = periodic(noise=0.5)
    df.iloc[100:104, 0] = np.nan
    df = df.drop(df.index[200:203])
    bt = backtest(df, "v", ["seasonal_naive", "holt_winters"], horizon=24, period=24)
    assert len(bt.scores) == 2 and any("interpolated" in n for n in bt.notes)
    sparse = periodic()
    sparse.iloc[::2, 0] = np.nan
    with pytest.raises(AnalysisError, match="more than 30% missing"):
        backtest(sparse, "v", ["seasonal_naive"], horizon=24, period=24)


def test_backtest_validation():
    df = periodic()
    with pytest.raises(AnalysisError, match="too few for 3 folds"):
        backtest(df.iloc[:100], "v", horizon=24, n_folds=3, period=24)
    with pytest.raises(AnalysisError, match="Unknown model"):
        backtest(df, "v", ["prophet"], horizon=24)
    with pytest.raises(AnalysisError, match="at least 1"):
        backtest(df, "v", horizon=0)
    with pytest.raises(AnalysisError, match="time-indexed"):
        backtest(pd.DataFrame({"v": range(300)}), "v")


def test_recommendation_prefers_a_simpler_model_within_two_percent():
    bt = backtest(
        periodic(noise=0.5), "v", ["seasonal_naive", "holt_winters"], horizon=24, period=24
    )
    rec = recommend_model(bt)
    assert (
        rec.ranking[0].worse_than_best_pct == 0
        and rec.reason
        and rec.model
        in {
            "seasonal_naive",
            "holt_winters",
        }
    )
    near = bt.model_copy(deep=True)
    near.scores[0] = near.scores[0].model_copy(update={"model": "seasonal_naive", "mae": 10.1})
    near.scores[1] = near.scores[1].model_copy(update={"model": "holt_winters", "mae": 10.0})
    pick = recommend_model(near)
    assert pick.model == "seasonal_naive" and "simpler" in pick.reason
    assert "within 1.0%" in pick.reason
    near.scores[0] = near.scores[0].model_copy(update={"mae": 12.0})
    assert recommend_model(near).model == "holt_winters"
    assert recommend_model(bt.model_copy(update={"scores": []})).model is None


def test_forecast_is_labelled_an_estimate_and_follows_the_pattern():
    df = periodic(noise=0.5)
    out = forecast(df, "v", "seasonal_naive", horizon=24, period=24)
    assert out.label == "estimate" and "not guarantees" in out.note and len(out.points) == 24
    assert pd.Timestamp(out.points[0].timestamp) == df.index[-1] + pd.Timedelta(hours=1)
    assert all(p.lower <= p.forecast <= p.upper for p in out.points)
    truth = 50 + 10 * np.sin(2 * np.pi * np.arange(len(df), len(df) + 24) / 24)
    assert np.abs(np.array([p.forecast for p in out.points]) - truth).mean() < 2
    assert out.interval_method == "empirical quantiles of backtest errors"


def test_interval_covers_roughly_the_stated_share_of_held_out_points():
    # loose tolerance: an 80% interval should cover between 50% and 100% of 120 held-out points
    full = periodic(cycles=24, noise=1.0, seed=4)
    train, held = full.iloc[:-120], full.iloc[-120:]
    covered = []
    for start in range(0, 120, 24):
        window = train if start == 0 else pd.concat([train, held.iloc[:start]])
        out = forecast(window, "v", "holt_winters", horizon=24, period=24, interval_level=0.8)
        truth = held["v"].iloc[start : start + 24].to_numpy()
        covered += [p.lower <= y <= p.upper for p, y in zip(out.points, truth, strict=True)]
    assert 0.5 <= np.mean(covered) <= 1.0


def test_sarima_uses_its_own_interval_and_forecast_validation():
    df = periodic(period=6, cycles=20, noise=0.3)
    out = forecast(df, "v", "sarima", horizon=6, period=6)
    assert out.interval_method == "model prediction interval"
    assert all(p.lower < p.forecast < p.upper for p in out.points)
    with pytest.raises(AnalysisError, match="at least 1"):
        forecast(df, "v", "seasonal_naive", horizon=0)
    with pytest.raises(AnalysisError, match="between 0 and 1"):
        forecast(df, "v", "seasonal_naive", horizon=6, interval_level=1.5)


def test_results_have_evidence_logs_json_and_the_input_is_untouched():
    df = periodic(noise=0.5)
    snapshot, before = copy.deepcopy(df), fingerprint(df)
    bt = backtest(df, "v", ["seasonal_naive"], horizon=24, period=24)
    out = forecast(df, "v", "seasonal_naive", horizon=12, period=24, backtest_result=bt)
    pd.testing.assert_frame_equal(df, snapshot)
    assert fingerprint(df) == before
    for result, method in ((bt, "rolling_origin_backtest"), (out, "seasonal_naive")):
        assert result.evidence.method == method and result.evidence.columns == ["v"]
        assert result.evidence.n_observations == len(df) and result.log_entry.read_only
        json.loads(result.model_dump_json())
    assert bt.evidence.parameters == {
        "models": ["seasonal_naive"],
        "horizon": 24,
        "n_folds": 3,
        "period": 24,
    }
