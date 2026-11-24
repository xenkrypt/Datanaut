"""Helpers for the report tests."""

from __future__ import annotations

from datetime import UTC, datetime

from datanaut.ui import analysis
from datanaut.ui.state import WizardState

FIXED_DATE = datetime(2026, 1, 15, 9, 30, tzinfo=UTC)


def run_everything(state: WizardState, weather_csv: bytes) -> None:
    analysis.characterization(state)
    analysis.hour_profile(state)
    analysis.weekday_profile(state)
    analysis.metrics(state)
    analysis.peaks(state)
    analysis.stl(state)
    analysis.change_points(state, "medium", "l2")
    analysis.set_weather_file(state, weather_csv)
    analysis.weather_result(state)
    analysis.run_forecast(state, 24)
    analysis.what_if(state, 0.3, 95.0)
    analysis.findings(state)
