"""Fixtures for the report tests: a session where the user has already run the analyses."""

from __future__ import annotations

import pytest
from reporthelpers import run_everything  # noqa: E402

from datanaut.ui import actions
from datanaut.ui.state import WizardState


@pytest.fixture(scope="module")
def state(phase3) -> WizardState:
    s = WizardState()
    actions.load_upload(s, phase3.csv_bytes, "household.csv")
    actions.confirm_mapping(s, ["timestamp"], None, ["kw"], {"kw": True})
    run_everything(s, phase3.weather_csv)
    s.take_messages()
    return s
