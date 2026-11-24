"""Anomaly detection: unusual points in one series, and unusual combinations across several.

`stl_anomalies` removes the trend and repeating pattern and flags readings whose leftover is
large for that series. `multivariate_anomalies` looks at several columns together with an
Isolation Forest, so a point that is ordinary in each column but unusual jointly is found.
`compare_detectors` shows what each approach finds that the other misses. Flags mean "unusual
compared with the rest of the data", never "wrong" and never "caused by" anything.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from datanaut.core.cleaner import MAD_TO_SIGMA, seasonal_fit
from datanaut.core.evidence import (
    Evidence,
    analysis_log,
    make_evidence,
    numeric_series,
    to_regular_grid,
)
from datanaut.core.log import StepLog
from datanaut.errors import AnalysisError, CleaningError

MAX_LISTED = 1000
"""At most this many flagged points are listed in a result; counts are always complete."""

MIN_MULTIVARIATE_ROWS = 30
N_ESTIMATORS = 200


# --------------------------------------------------------------------- STL residuals


class AnomalyPoint(BaseModel):
    """One unusual reading."""

    timestamp: str
    value: float
    residual: float
    score: float
    direction: Literal["high", "low"]


class DayAnomalies(BaseModel):
    """How many unusual readings fell on one calendar day."""

    date: str
    count: int
    max_score: float


class STLAnomalies(BaseModel):
    """Readings far from the series' own trend-plus-seasonal pattern."""

    column: str
    period: int
    threshold: float
    n_anomalies: int
    points: list[AnomalyPoint] = Field(default_factory=list)
    truncated: bool = False
    days: list[DayAnomalies] = Field(default_factory=list)
    residual_scale: float
    rows_total: int
    excluded_missing: int
    evidence: Evidence
    log_entry: StepLog
    _flagged: pd.DatetimeIndex = PrivateAttr(default_factory=lambda: pd.DatetimeIndex([]))

    def flagged_timestamps(self) -> pd.DatetimeIndex:
        """Every flagged timestamp (the `points` list may be shortened, this is not)."""
        return self._flagged


def stl_anomalies(
    df: pd.DataFrame, column: str, period: int, threshold: float = 3.5
) -> STLAnomalies:
    """Flag points whose seasonal-trend residual is more than `threshold` robust deviations out.

    The score is |residual - median residual| / (1.4826 x MAD of the residuals). `period` is the
    number of readings per cycle (for example from characterize). The series is put on an even
    time grid first; readings missing from that grid are filled only to fit the pattern and are
    never flagged. Raises AnalysisError if the series is too short or more than half missing.
    """
    if period < 2:
        raise AnalysisError("period must be at least 2 readings per cycle.")
    if threshold <= 0:
        raise AnalysisError("threshold must be positive.")
    raw = numeric_series(df, column).replace([np.inf, -np.inf], np.nan)
    grid, _ = to_regular_grid(raw)  # keeps missing stretches at the edges as missing
    values = grid.to_numpy(dtype=float)
    if len(values) < 2 * period + 2:
        raise AnalysisError(
            f"'{column}' has {len(values)} time steps, too few for a seasonal fit with period "
            f"{period}: at least {2 * period + 2} are needed."
        )
    present = ~np.isnan(values)
    if present.mean() < 0.5:
        raise AnalysisError(f"'{column}' is more than half missing, too sparse for a seasonal fit.")
    positions = np.arange(len(values))
    filled = np.interp(positions, positions[present], values[present])
    try:
        fitted, center, scale = seasonal_fit(filled, period, threshold)
    except CleaningError as exc:
        raise AnalysisError(str(exc)) from exc
    residual = values - fitted
    score = np.abs(residual - center) / scale
    with np.errstate(invalid="ignore"):
        flagged = present & (score > threshold)
    index = grid.index
    order = np.flatnonzero(flagged)
    order = order[np.argsort(-score[order], kind="stable")]
    points = [
        AnomalyPoint(
            timestamp=str(index[i]),
            value=float(values[i]),
            residual=float(residual[i] - center),
            score=float(score[i]),
            direction="high" if residual[i] - center > 0 else "low",
        )
        for i in order[:MAX_LISTED]
    ]
    days = _day_summary(index[flagged], score[flagged])
    excluded = int((~present).sum())
    evidence = make_evidence(
        index[present],
        [column],
        int(present.sum()),
        "stl_residual_mad",
        {"period": period, "threshold": threshold, "robust": True},
    )
    summary = (
        f"STL anomalies in '{column}' (period {period}, threshold {threshold:g}): "
        f"{int(flagged.sum())} of {int(present.sum())} reading(s) unusual"
        + (f", most days: {', '.join(d.date for d in days[:3])}" if days else "")
        + f"; {excluded} time step(s) without a reading were not flagged."
    )
    result = STLAnomalies(
        column=column,
        period=period,
        threshold=threshold,
        n_anomalies=int(flagged.sum()),
        points=points,
        truncated=int(flagged.sum()) > MAX_LISTED,
        days=days,
        residual_scale=float(scale),
        rows_total=len(raw),
        excluded_missing=excluded,
        evidence=evidence,
        log_entry=analysis_log(
            "stl_anomalies",
            summary,
            evidence,
            counts={
                "anomalies": int(flagged.sum()),
                "observations": int(present.sum()),
                "excluded_missing": excluded,
            },
            rows=len(raw),
        ),
    )
    result._flagged = index[flagged]
    return result


def _day_summary(
    stamps: pd.DatetimeIndex, scores: np.ndarray, limit: int = 20
) -> list[DayAnomalies]:
    if len(stamps) == 0:
        return []
    frame = pd.DataFrame({"day": stamps.normalize(), "score": scores})
    grouped = frame.groupby("day")["score"].agg(["size", "max"]).reset_index()
    grouped = grouped.sort_values(["size", "max", "day"], ascending=[False, False, True]).head(
        limit
    )
    return [
        DayAnomalies(date=str(row.day.date()), count=int(row["size"]), max_score=float(row["max"]))
        for _, row in grouped.iterrows()
    ]


# ------------------------------------------------------------------ multivariate


class ColumnContribution(BaseModel):
    """How far one column is from usual at a flagged point."""

    column: str
    robust_z: float
    share: float


class MultivariatePoint(BaseModel):
    """A point that is unusual across the columns taken together."""

    timestamp: str
    score: float
    values: dict[str, float]
    contributions: list[ColumnContribution]


class MultivariateAnomalies(BaseModel):
    """Isolation Forest result over several columns, or a note that it was not applicable."""

    columns: list[str]
    applicable: bool
    reason: str | None = None
    contamination: float
    random_state: int
    n_flagged: int = 0
    points: list[MultivariatePoint] = Field(default_factory=list)
    truncated: bool = False
    rows_total: int
    rows_used: int = 0
    rows_excluded: int = 0
    evidence: Evidence
    log_entry: StepLog
    _flagged: pd.DatetimeIndex = PrivateAttr(default_factory=lambda: pd.DatetimeIndex([]))

    def flagged_timestamps(self) -> pd.DatetimeIndex:
        """Every flagged timestamp (the `points` list may be shortened, this is not)."""
        return self._flagged


def _not_applicable(
    df: pd.DataFrame, columns: list[str], reason: str, contamination: float, random_state: int
) -> MultivariateAnomalies:
    evidence = make_evidence(
        None,
        columns,
        0,
        "isolation_forest",
        {"contamination": contamination, "random_state": random_state},
    )
    return MultivariateAnomalies(
        columns=columns,
        applicable=False,
        reason=reason,
        contamination=contamination,
        random_state=random_state,
        rows_total=len(df),
        evidence=evidence,
        log_entry=analysis_log(
            "multivariate_anomalies",
            f"Not applicable: {reason}",
            evidence,
            counts={"applicable": 0},
            rows=len(df),
        ),
    )


def _robust_scale(values: np.ndarray) -> np.ndarray:
    """Per-column robust z-scores: (x - median) / (1.4826 x MAD), falling back to the std."""
    center = np.median(values, axis=0)
    spread = MAD_TO_SIGMA * np.median(np.abs(values - center), axis=0)
    fallback = values.std(axis=0)
    spread = np.where(spread > 0, spread, fallback)
    spread = np.where(spread > 0, spread, 1.0)
    return (values - center) / spread


def multivariate_anomalies(
    df: pd.DataFrame,
    columns: list[str],
    contamination: float = 0.01,
    random_state: int = 0,
    decorrelate: bool = True,
) -> MultivariateAnomalies:
    """Isolation Forest on robustly scaled columns; flagged points list their top columns.

    Rows missing any of the columns are left out (and counted). With fewer than two usable
    columns, or too few complete rows, the result says it is not applicable instead of failing.
    `contamination` is the share of rows expected to be unusual (0 to 0.5). With `decorrelate`
    (the default) the scaled columns are rotated onto their principal axes before the forest
    runs: a forest splits one axis at a time, so without the rotation it misses points that
    are ordinary in each column but break the usual relationship between columns.
    """
    if not 0 < contamination <= 0.5:
        raise AnalysisError("contamination must be greater than 0 and at most 0.5.")
    columns = [str(c) for c in columns]
    for c in columns:
        numeric_series(df, c)  # validates the time index, existence and dtype
    if len(set(columns)) < 2:
        return _not_applicable(
            df,
            columns,
            "Multivariate detection needs at least two different columns.",
            contamination,
            random_state,
        )
    data = df[columns].astype(float).replace([np.inf, -np.inf], np.nan)
    complete = data.dropna()
    if len(complete) < MIN_MULTIVARIATE_ROWS:
        return _not_applicable(
            df,
            columns,
            f"Only {len(complete)} row(s) have every column; at least "
            f"{MIN_MULTIVARIATE_ROWS} are needed.",
            contamination,
            random_state,
        )
    values = complete.to_numpy()
    constant = [c for c, s in zip(columns, values.std(axis=0), strict=True) if s == 0]
    if len(columns) - len(constant) < 2:
        return _not_applicable(
            df,
            columns,
            "Fewer than two columns vary, so there is nothing to combine.",
            contamination,
            random_state,
        )
    z = _robust_scale(values)
    forest = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=contamination,
        random_state=random_state,
        n_jobs=1,
    )
    features = PCA(random_state=random_state).fit_transform(z) if decorrelate else z
    labels = forest.fit_predict(features)
    score = -forest.score_samples(features)
    flagged = np.flatnonzero(labels == -1)
    flagged = flagged[np.argsort(-score[flagged], kind="stable")]
    points = []
    for i in flagged[:MAX_LISTED]:
        magnitude = np.abs(z[i])
        total = float(magnitude.sum()) or 1.0
        ranked = sorted(zip(columns, z[i], magnitude, strict=True), key=lambda t: -t[2])
        points.append(
            MultivariatePoint(
                timestamp=str(complete.index[i]),
                score=float(score[i]),
                values={c: float(v) for c, v in zip(columns, values[i], strict=True)},
                contributions=[
                    ColumnContribution(column=c, robust_z=float(zz), share=float(m) / total)
                    for c, zz, m in ranked
                ],
            )
        )
    evidence = make_evidence(
        complete.index,
        columns,
        len(complete),
        "isolation_forest",
        {
            "contamination": contamination,
            "random_state": random_state,
            "n_estimators": N_ESTIMATORS,
            "decorrelate": decorrelate,
            "scaling": "median and scaled MAD per column",
        },
    )
    summary = (
        f"Isolation Forest over {', '.join(columns)}: {len(flagged)} of {len(complete)} row(s) "
        f"are unusual in combination; {len(df) - len(complete)} row(s) with a missing value "
        "were left out."
    )
    result = MultivariateAnomalies(
        columns=columns,
        applicable=True,
        contamination=contamination,
        random_state=random_state,
        n_flagged=int(len(flagged)),
        points=points,
        truncated=len(flagged) > MAX_LISTED,
        rows_total=len(df),
        rows_used=len(complete),
        rows_excluded=len(df) - len(complete),
        evidence=evidence,
        log_entry=analysis_log(
            "multivariate_anomalies",
            summary,
            evidence,
            counts={
                "flagged": len(flagged),
                "observations": len(complete),
                "excluded": len(df) - len(complete),
            },
            rows=len(df),
        ),
    )
    result._flagged = complete.index[np.sort(flagged)]
    return result


# ------------------------------------------------------------------------ comparison


class UnusualCombination(BaseModel):
    """A point only the multivariate model flagged, with the columns that stand out."""

    timestamp: str
    top_columns: list[str] = Field(default_factory=list)


class DetectorComparison(BaseModel):
    """What each detector finds that the other does not."""

    only_multivariate: list[UnusualCombination]
    only_univariate: list[str]
    both: list[str]
    n_only_multivariate: int
    n_only_univariate: int
    n_both: int
    truncated: bool = False
    note: str = (
        "Points flagged only by the multivariate model are ordinary in each column on its own "
        "but unusual together."
    )
    evidence: Evidence
    log_entry: StepLog


FlagSource = (
    STLAnomalies
    | MultivariateAnomalies
    | Mapping[str, STLAnomalies]
    | Iterable[Any]
    | pd.DatetimeIndex
)


def _flag_index(source: Any) -> pd.DatetimeIndex:
    if isinstance(source, STLAnomalies | MultivariateAnomalies):
        return source.flagged_timestamps()
    if isinstance(source, Mapping):
        parts = [_flag_index(v) for v in source.values()]
        return (
            pd.DatetimeIndex([]) if not parts else parts[0].append(parts[1:]).unique().sort_values()
        )
    return pd.DatetimeIndex(pd.to_datetime(list(source)))


def _source_columns(source: Any) -> list[str]:
    if isinstance(source, STLAnomalies):
        return [source.column]
    if isinstance(source, MultivariateAnomalies):
        return list(source.columns)
    if isinstance(source, Mapping):
        return [c for v in source.values() for c in _source_columns(v)]
    return []


def compare_detectors(
    univariate_flags: FlagSource, multivariate_flags: FlagSource
) -> DetectorComparison:
    """Split flagged timestamps into only-multivariate, only-univariate and both.

    Each argument may be a result (or a dict of per-column STL results) or plain timestamps.
    Points only the multivariate model flagged are the unusual combinations.
    """
    uni = _flag_index(univariate_flags)
    multi = _flag_index(multivariate_flags)
    both = uni.intersection(multi).sort_values()
    only_multi = multi.difference(uni).sort_values()
    only_uni = uni.difference(multi).sort_values()
    columns_by_time: dict[str, list[str]] = {}
    if isinstance(multivariate_flags, MultivariateAnomalies):
        columns_by_time = {
            p.timestamp: [c.column for c in p.contributions[:2]] for p in multivariate_flags.points
        }
    combos = [
        UnusualCombination(timestamp=str(t), top_columns=columns_by_time.get(str(t), []))
        for t in only_multi[:MAX_LISTED]
    ]
    union = uni.union(multi)
    columns = list(
        dict.fromkeys(_source_columns(univariate_flags) + _source_columns(multivariate_flags))
    )
    evidence = make_evidence(union, columns, len(union), "compare_detectors", {})
    summary = (
        f"Detector comparison: {len(both)} point(s) flagged by both, {len(only_multi)} only in "
        f"combination across columns, {len(only_uni)} only by single-column detection."
    )
    return DetectorComparison(
        only_multivariate=combos,
        only_univariate=[str(t) for t in only_uni[:MAX_LISTED]],
        both=[str(t) for t in both[:MAX_LISTED]],
        n_only_multivariate=len(only_multi),
        n_only_univariate=len(only_uni),
        n_both=len(both),
        truncated=max(len(only_multi), len(only_uni), len(both)) > MAX_LISTED,
        evidence=evidence,
        log_entry=analysis_log(
            "compare_detectors",
            summary,
            evidence,
            counts={
                "both": len(both),
                "only_multivariate": len(only_multi),
                "only_univariate": len(only_uni),
            },
        ),
    )
