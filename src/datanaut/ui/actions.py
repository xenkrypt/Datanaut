"""What a click does: plain functions that call the core and update the WizardState.

No streamlit import. Errors become friendly messages; technical details go to the server log
without any cell values.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

import datanaut
from datanaut import limits
from datanaut.core.loader import load_file
from datanaut.core.profiler import DataProfile, profile
from datanaut.core.timeindex import (
    TimeIndexResult,
    build_time_index,
    detect_time_columns,
    detect_value_columns,
)
from datanaut.core.versioning import VersionStore
from datanaut.errors import DatanautError, FileTooLargeError, ParseError, UnsupportedFormatError
from datanaut.ui.state import WizardState, cache_key

_LOG = logging.getLogger("datanaut.ui")
SAMPLE_NAME = "household_energy_sample.csv"

_SUGGESTIONS: tuple[tuple[type[Exception], str], ...] = (
    (FileTooLargeError, "Try a smaller file, a shorter time range or fewer columns."),
    (UnsupportedFormatError, "Save the data as CSV or Excel (.xlsx) and upload it again."),
    (ParseError, "Check that the file opens in a spreadsheet and has a header row."),
)


def friendly_error(exc: Exception, step: str) -> str:
    """A short message for the user: what failed, where, and what to try next."""
    if isinstance(exc, DatanautError):
        hint = next((h for kind, h in _SUGGESTIONS if isinstance(exc, kind)), "")
        return f"{exc.message} ({step}) {hint}".strip()
    return (
        f"Something unexpected went wrong in the {step} step. Your data was not changed. "
        "Go back a step and try again, or try a different file."
    )


def guard(state: WizardState, step: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run `fn`; on failure add a friendly error message and return None. Never raises."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # the UI must never show a traceback
        _LOG.error("%s failed in step %s", type(exc).__name__, step)  # no data values
        state.add_message("error", friendly_error(exc, step))
        return None


def guarded_render(state: WizardState, step: str, render: Callable[[WizardState], Any]) -> bool:
    """Draw a screen; if it fails, add a friendly message instead of a traceback.

    Returns False on failure. Streamlit's own rerun and stop signals are not caught.
    """
    try:
        render(state)
    except Exception as exc:
        _LOG.error("%s failed while drawing the %s screen", type(exc).__name__, step)
        state.add_message("error", friendly_error(exc, step))
        return False
    return True


# --- loading ------------------------------------------------------------------------------


def sample_path() -> Path:
    """The bundled sample file (it sits next to the package in a source checkout)."""
    return Path(datanaut.__file__).resolve().parents[2] / "sample_data" / SAMPLE_NAME


def load_upload(
    state: WizardState, data: bytes, filename: str | None, sheet: str | int | None = None
) -> bool:
    """Load uploaded bytes into the state. Returns False (and does nothing) if unchanged."""
    limits.check_upload_size(len(data))
    digest = hashlib.sha256(data).hexdigest()
    if digest == state.source_digest and sheet == state.sheet and state.loaded is not None:
        return False
    loaded = load_file(data, filename, max_bytes=limits.MAX_UPLOAD_BYTES, sheet=sheet)
    limits.check_dimensions(loaded.df)
    chosen = loaded.source_info.sheet if sheet is not None else None
    state.set_loaded(loaded, data, digest, chosen)
    state.mark_done("Upload")
    return True


def load_sample(state: WizardState) -> bool:
    return load_upload(state, sample_path().read_bytes(), SAMPLE_NAME)


# --- profile ------------------------------------------------------------------------------


def tentative_index(state: WizardState) -> pd.DataFrame:
    """The raw data with its most likely time index, or the raw data if that is not clear."""
    assert state.loaded is not None
    raw = state.loaded.df
    candidates = detect_time_columns(raw)
    if not candidates:
        return raw
    try:
        result = build_time_index(raw, candidates[0].columns)
    except ParseError:
        return raw
    return raw if result.flags.ambiguous else result.df


def ensure_profile(state: WizardState) -> DataProfile:
    """Profile the data once (read-only) and log it once."""
    if state.profile is None:
        assert state.loaded is not None
        key = cache_key(state.source_digest, "profile")
        state.profile = state.get_or_compute(key, lambda: profile(tentative_index(state)))
    if not state.profile_logged:
        state.log.extend([state.profile.log_entry])
        state.base_log_length = len(state.log)
        state.profile_logged = True
    return state.profile


# --- mapping ------------------------------------------------------------------------------


def suggest_mapping(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Suggested timestamp column(s) and value columns."""
    candidates = detect_time_columns(df)
    time_cols = candidates[0].columns if candidates else []
    return time_cols, detect_value_columns(df, time_cols)


def preview_time_index(
    state: WizardState, time_columns: list[str], dayfirst: bool | None
) -> TimeIndexResult:
    """Parse the chosen time column(s) without changing anything (cached)."""
    assert state.loaded is not None
    key = cache_key(state.source_digest, "time_index", cols=list(time_columns), dayfirst=dayfirst)
    return state.get_or_compute(
        key, lambda: build_time_index(state.loaded.df, list(time_columns), dayfirst)
    )


def confirm_mapping(
    state: WizardState,
    time_columns: list[str],
    dayfirst: bool | None,
    value_columns: list[str],
    consumption: dict[str, bool],
) -> bool:
    """Build the time index and create V1 (or just update the value columns and flags).

    Returns True when a new V1 was created. Raises ParseError for choices that cannot work.
    """
    if state.loaded is None:
        raise ParseError("Upload a file first.")
    raw = state.loaded.df
    if not 1 <= len(time_columns) <= 2:
        raise ParseError("Choose one timestamp column, or a date column and a time column.")
    if not value_columns:
        raise ParseError("Choose at least one value column to analyze.")
    bad = [c for c in value_columns if c not in raw.columns or not _is_numeric(raw[c])]
    if bad:
        raise ParseError(f"Column '{bad[0]}' is not numeric, so it cannot be analyzed.")
    flags = {c: bool(consumption.get(c, False)) for c in value_columns}

    same_time = state.store is not None and (state.time_columns, state.dayfirst) == (
        list(time_columns),
        dayfirst,
    )
    if same_time:
        state.value_columns, state.consumption = list(value_columns), flags
        return False

    indexed = preview_time_index(state, time_columns, dayfirst)
    if indexed.flags.ambiguous:
        raise ParseError("Choose whether dates are day-first or month-first.")
    if indexed.df.empty:
        raise ParseError("No row has a usable timestamp in the chosen column(s).")
    ensure_profile(state)
    if state.store is not None:
        state.invalidate_mapping("The time column choice changed, so the time index was rebuilt.")
    state.log.extend(indexed.log)
    state.store = VersionStore(indexed.df, state.log)
    state.indexed = indexed
    state.time_columns, state.dayfirst = list(time_columns), dayfirst
    state.value_columns, state.consumption = list(value_columns), flags
    state.profile = state.get_or_compute(
        cache_key(state.fingerprint, "profile"), lambda: profile(indexed.df)
    )
    state.mark_done("Map columns")
    return True


def _is_numeric(series: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series)) and not pd.api.types.is_bool_dtype(series)
