"""Forecast benchmarking: compare simple models fairly, then forecast with the best one.

Models are scored with rolling-origin backtests: several chronological train/test splits, each
model refit on the data before the split and scored on the period after it. Nothing is ever
shuffled, so no model sees its own test data. Every forecast is an estimate and is returned
together with an interval; none of this says why the series moves.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from datanaut.core.evidence import (
    Evidence,
    analysis_log,
    make_evidence,
    numeric_series,
    to_regular_grid,
)
from datanaut.core.log import StepLog
from datanaut.errors import AnalysisError

ModelName = Literal["seasonal_naive", "holt_winters", "sarima"]
MODELS: tuple[str, ...] = ("seasonal_naive", "holt_winters", "sarima")
SIMPLICITY = {"seasonal_naive": 0, "holt_winters": 1, "sarima": 2}
"""Lower is simpler. Used to prefer a simple model that is nearly as accurate."""

MAX_SARIMA_POINTS = 5000
SIMPLE_TOLERANCE = 0.02
"""A simpler model within this share of the best model's error is preferred."""
MAX_MISSING_SHARE = 0.3
ESTIMATE_NOTE = "These are estimates from past patterns, not guarantees."


# ------------------------------------------------------------------------------ models


def _fit_predict(
    name: str,
    train: np.ndarray,
    horizon: int,
    period: int | None,
    *,
    damped: bool = True,
    interval: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Forecast `horizon` steps after `train`. Returns (point, lower, upper); bounds may be None."""
    seasonal = period is not None and period >= 2
    if name == "seasonal_naive":
        p = period if seasonal else 1
        if len(train) < p:
            raise AnalysisError(f"Seasonal naive needs at least {p} observations.")
        last = train[-p:]
        return np.array([last[h % p] for h in range(horizon)]), None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "holt_winters":
            use_season = seasonal and len(train) >= 2 * period  # type: ignore[operator]
            fit = ExponentialSmoothing(
                train,
                trend="add",
                damped_trend=damped,
                seasonal="add" if use_season else None,
                seasonal_periods=period if use_season else None,
            ).fit()
            return np.asarray(fit.forecast(horizon)), None, None
        if name == "sarima":
            seasonal_order = (0, 1, 1, period) if seasonal else (0, 0, 0, 0)
            fit = SARIMAX(
                train,
                order=(1, 1, 1),
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=100)
            result = fit.get_forecast(horizon)
            point = np.asarray(result.predicted_mean)
            if interval is None:
                return point, None, None
            bounds = np.asarray(result.conf_int(alpha=1 - interval))
            return point, bounds[:, 0], bounds[:, 1]
    raise AnalysisError(f"Unknown model '{name}'. Choose from {', '.join(MODELS)}.")


def _scores(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    error = actual - predicted
    denom = np.abs(actual) + np.abs(predicted)
    ratio = np.divide(2 * np.abs(error), denom, out=np.zeros_like(error), where=denom > 0)
    return (
        float(np.mean(np.abs(error))),
        float(np.sqrt(np.mean(error**2))),
        float(100 * ratio.mean()),
    )


# ----------------------------------------------------------------------------- data


@dataclass
class _Prepared:
    values: np.ndarray
    index: pd.DatetimeIndex
    step: pd.Timedelta
    missing_filled: int
    rows_total: int


def _prepare(df: pd.DataFrame, column: str) -> _Prepared:
    raw = numeric_series(df, column).replace([np.inf, -np.inf], np.nan)
    grid, step = to_regular_grid(raw)
    missing = int(grid.isna().sum())
    if missing / len(grid) > MAX_MISSING_SHARE:
        raise AnalysisError(
            f"'{column}' is more than {MAX_MISSING_SHARE:.0%} missing; clean it first."
        )
    filled = grid.interpolate(method="time", limit_direction="both")
    return _Prepared(
        values=filled.to_numpy(dtype=float),
        index=grid.index,
        step=step,
        missing_filled=missing,
        rows_total=len(raw),
    )


# ---------------------------------------------------------------------------- backtest


class FoldScore(BaseModel):
    """Errors of one model on one test window."""

    fold: int
    mae: float
    rmse: float
    smape: float


class ModelScore(BaseModel):
    """A model's errors across all folds (smape is a percentage)."""

    model: str
    folds: list[FoldScore]
    mae: float
    rmse: float
    smape: float
    fit_seconds: float


class ModelFailure(BaseModel):
    """A model that could not be scored, and why. Other models are unaffected."""

    model: str
    reason: str
    skipped: bool = False


class FoldInfo(BaseModel):
    """One chronological split: the training data always ends before the test window starts."""

    fold: int
    n_train: int
    train_end: str
    test_start: str
    test_end: str


class Backtest(BaseModel):
    """Rolling-origin comparison of forecasting models."""

    column: str
    horizon: int
    n_folds: int
    period: int | None
    folds: list[FoldInfo]
    scores: list[ModelScore]
    failures: list[ModelFailure]
    notes: list[str] = Field(default_factory=list)
    evidence: Evidence
    log_entry: StepLog
    _errors: dict[str, list[float]] = PrivateAttr(default_factory=dict)

    def errors(self, model: str) -> list[float]:
        """Signed errors (actual - forecast) of one model over every test point."""
        return list(self._errors.get(model, []))


def backtest(
    df: pd.DataFrame,
    column: str,
    models: list[str] | tuple[str, ...] = MODELS,
    horizon: int = 24,
    n_folds: int = 3,
    period: int | None = None,
    *,
    max_sarima_points: int = MAX_SARIMA_POINTS,
) -> Backtest:
    """Score models on `n_folds` consecutive test windows of `horizon` points at the end.

    Fold k trains on everything before its window (an expanding window) and is scored on the
    window itself. Each model is scored with MAE, RMSE and sMAPE (%); a model that fails, or
    that is skipped (SARIMA on a very long series), is reported without stopping the others.
    `period` is the number of points per season (None fits non-seasonal versions).
    """
    if horizon < 1 or n_folds < 1:
        raise AnalysisError("horizon and n_folds must be at least 1.")
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise AnalysisError(f"Unknown model '{unknown[0]}'. Choose from {', '.join(MODELS)}.")
    data = _prepare(df, column)
    n = len(data.values)
    min_train = max(2 * (period or 1), 10)
    if n - n_folds * horizon < min_train:
        raise AnalysisError(
            f"'{column}' has {n} time steps: too few for {n_folds} folds of {horizon} with at "
            f"least {min_train} training points. Use a shorter horizon or fewer folds."
        )
    folds, bounds = [], []
    for k in range(n_folds):
        end = n - (n_folds - 1 - k) * horizon
        start = end - horizon
        bounds.append((start, end))
        folds.append(
            FoldInfo(
                fold=k + 1,
                n_train=start,
                train_end=str(data.index[start - 1]),
                test_start=str(data.index[start]),
                test_end=str(data.index[end - 1]),
            )
        )
    scores: list[ModelScore] = []
    failures: list[ModelFailure] = []
    errors: dict[str, list[float]] = {}
    for name in models:
        if name == "sarima" and n > max_sarima_points:
            failures.append(
                ModelFailure(
                    model=name,
                    skipped=True,
                    reason=(
                        f"SARIMA was skipped: the series has {n} points, over the limit of "
                        f"{max_sarima_points}. Aggregate it to a coarser frequency (for example "
                        "daily) and try again."
                    ),
                )
            )
            continue
        started = time.perf_counter()
        fold_scores: list[FoldScore] = []
        model_errors: list[float] = []
        try:
            for k, (start, end) in enumerate(bounds):
                point, _, _ = _fit_predict(name, data.values[:start], horizon, period)
                actual = data.values[start:end]
                if not np.all(np.isfinite(point)):
                    raise AnalysisError("the model produced non-finite forecasts")
                mae, rmse, smape = _scores(actual, point)
                fold_scores.append(FoldScore(fold=k + 1, mae=mae, rmse=rmse, smape=smape))
                model_errors.extend((actual - point).tolist())
        except Exception as exc:  # one model failing must not stop the others
            failures.append(ModelFailure(model=name, reason=f"{type(exc).__name__}: {exc}"))
            continue
        scores.append(
            ModelScore(
                model=name,
                folds=fold_scores,
                mae=float(np.mean([f.mae for f in fold_scores])),
                rmse=float(np.mean([f.rmse for f in fold_scores])),
                smape=float(np.mean([f.smape for f in fold_scores])),
                fit_seconds=time.perf_counter() - started,
            )
        )
        errors[name] = model_errors
    notes = []
    if data.missing_filled:
        notes.append(
            f"{data.missing_filled} missing time step(s) were interpolated before fitting."
        )
    evidence = make_evidence(
        data.index,
        [column],
        n,
        "rolling_origin_backtest",
        {"models": list(models), "horizon": horizon, "n_folds": n_folds, "period": period},
    )
    ranked = sorted(scores, key=lambda s: s.mae)
    summary = (
        f"Backtested {len(scores)} model(s) on '{column}' ({n_folds} chronological folds of "
        f"{horizon}): "
        + (
            f"lowest average error {ranked[0].model} (MAE {ranked[0].mae:.4g}). "
            if ranked
            else "none could be scored. "
        )
        + (f"{len(failures)} model(s) failed or were skipped." if failures else "")
    )
    result = Backtest(
        column=column,
        horizon=horizon,
        n_folds=n_folds,
        period=period,
        folds=folds,
        scores=scores,
        failures=failures,
        notes=notes,
        evidence=evidence,
        log_entry=analysis_log(
            "forecast_backtest",
            summary,
            evidence,
            counts={"models_scored": len(scores), "models_failed": len(failures), "folds": n_folds},
            rows=data.rows_total,
        ),
    )
    result._errors = errors
    return result


# ------------------------------------------------------------------- recommendation


class RankedModel(BaseModel):
    """A model's place in the ranking by average backtest error."""

    model: str
    mae: float
    worse_than_best_pct: float


class Recommendation(BaseModel):
    """Which model to use and why."""

    model: str | None
    ranking: list[RankedModel]
    reason: str


def recommend_model(result: Backtest, tolerance: float = SIMPLE_TOLERANCE) -> Recommendation:
    """Rank by average MAE; prefer the simplest model within `tolerance` of the best one."""
    if not result.scores:
        return Recommendation(model=None, ranking=[], reason="No model could be scored.")
    ranked = sorted(result.scores, key=lambda s: (s.mae, SIMPLICITY[s.model]))
    best = ranked[0]
    base = best.mae if best.mae > 0 else 1.0
    ranking = [
        RankedModel(model=s.model, mae=s.mae, worse_than_best_pct=100 * (s.mae - best.mae) / base)
        for s in ranked
    ]
    near = [s for s in ranked if s.mae <= best.mae * (1 + tolerance) or s.mae == best.mae]
    choice = min(near, key=lambda s: (SIMPLICITY[s.model], s.mae))
    if choice.model == best.model:
        reason = f"{choice.model} has the lowest average backtest error (MAE {choice.mae:.4g})."
    else:
        pct = 100 * (choice.mae - best.mae) / base
        reason = (
            f"{choice.model} is within {pct:.1f}% of the most accurate model ({best.model}, MAE "
            f"{best.mae:.4g}) and is simpler, so it is preferred (rule: within {tolerance:.0%})."
        )
    return Recommendation(model=choice.model, ranking=ranking, reason=reason)


# --------------------------------------------------------------------------- forecast


class ForecastPoint(BaseModel):
    """One forecast time with its interval."""

    timestamp: str
    forecast: float
    lower: float
    upper: float


class Forecast(BaseModel):
    """A forecast, always labelled as an estimate."""

    column: str
    model: str
    horizon: int
    label: Literal["estimate"] = "estimate"
    note: str = ESTIMATE_NOTE
    interval_level: float
    interval_method: str
    points: list[ForecastPoint]
    evidence: Evidence
    log_entry: StepLog


def forecast(
    df: pd.DataFrame,
    column: str,
    best_model: str,
    horizon: int,
    period: int | None = None,
    *,
    interval_level: float = 0.8,
    backtest_result: Backtest | None = None,
) -> Forecast:
    """Forecast `horizon` steps ahead with a prediction interval.

    SARIMA uses its own interval. For the other models the interval comes from the empirical
    quantiles of their backtest errors (pass `backtest_result` to reuse one).
    """
    if not 0 < interval_level < 1:
        raise AnalysisError("interval_level must be between 0 and 1.")
    if horizon < 1:
        raise AnalysisError("horizon must be at least 1.")
    data = _prepare(df, column)
    if best_model == "sarima" and len(data.values) > MAX_SARIMA_POINTS:
        raise AnalysisError("SARIMA is not available for a series this long; aggregate it first.")
    point, lower, upper = _fit_predict(
        best_model, data.values, horizon, period, interval=interval_level
    )
    if lower is None or upper is None:
        bt = backtest_result or backtest(df, column, [best_model], horizon=horizon, period=period)
        errors = np.asarray(bt.errors(best_model))
        if len(errors) < 5:
            raise AnalysisError("Too few backtest errors to build a prediction interval.")
        lo_q, hi_q = np.quantile(errors, [(1 - interval_level) / 2, 1 - (1 - interval_level) / 2])
        lower, upper = point + lo_q, point + hi_q
        method = "empirical quantiles of backtest errors"
    else:
        method = "model prediction interval"
    stamps = data.index[-1] + data.step * np.arange(1, horizon + 1)
    points = [
        ForecastPoint(timestamp=str(t), forecast=float(f), lower=float(lo), upper=float(hi))
        for t, f, lo, hi in zip(stamps, point, lower, upper, strict=True)
    ]
    evidence = make_evidence(
        data.index,
        [column],
        len(data.values),
        best_model,
        {
            "horizon": horizon,
            "period": period,
            "interval_level": interval_level,
            "interval": method,
        },
    )
    summary = (
        f"Estimate: {horizon}-step forecast of '{column}' with {best_model}, "
        f"{interval_level:.0%} interval from {method}."
    )
    return Forecast(
        column=column,
        model=best_model,
        horizon=horizon,
        interval_level=interval_level,
        interval_method=method,
        points=points,
        evidence=evidence,
        log_entry=analysis_log(
            "forecast", summary, evidence, counts={"points": horizon}, rows=data.rows_total
        ),
    )
