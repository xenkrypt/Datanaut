"""Logic for the Report screen: gather existing results for a version and build the PDF.

No streamlit import. Nothing is analysed here: results come from what the user already ran, and
the PDF stays in session memory.
"""

from __future__ import annotations

from datanaut.core.weather import align_weather
from datanaut.report.model import (
    SECTION_TITLES,
    ReportInputs,
    available_sections,
    build_report_data,
)
from datanaut.report.pdf import build_pdf
from datanaut.ui import analysis
from datanaut.ui.state import WizardState

_KINDS = {
    "findings": "findings",
    "peaks": "peaks",
    "profile_hour": "hour",
    "profile_weekday": "weekday",
    "metrics": "metrics",
    "stl": "stl",
    "multivariate": "multivariate",
    "change_points": "changes",
    "backtest": "backtest",
    "forecast": "forecast",
    "peak_shift": "whatif",
    "investigate": "investigation",
    "weather_link": "weather_link",
}


def version_ids(state: WizardState) -> list[str]:
    assert state.store is not None
    return [n.version.version_id for n in state.store.history()]


def inputs_for(state: WizardState, version_id: str) -> ReportInputs:
    """The results the user already has for one version (a version never analysed has none)."""
    assert state.store is not None and state.loaded is not None
    store = state.store
    fp = store.version(version_id).fingerprint
    results = {
        kind: state.latest[(fp, name)]
        for name, kind in _KINDS.items()
        if (fp, name) in state.latest
    }
    weather_x = None
    link = results.get("weather_link")
    if link is not None and state.weather is not None:
        weather_x = align_weather(store.get(version_id), state.weather).df.get(link.variable)
    info = state.loaded.source_info
    return ReportInputs(
        store=store,
        log=state.log,
        column=analysis.column(state),
        dataset_name=info.filename or "dataset",
        value_columns=list(state.value_columns),
        consumption=dict(state.consumption),
        source={
            "filename": info.filename,
            "sheet": info.sheet,
            "time_columns": state.indexed.choices.time_columns if state.indexed else [],
            "dayfirst": state.indexed.choices.dayfirst if state.indexed else None,
        },
        characterization=state.latest.get((fp, "characterize")),
        validation=state.report if state.report_fingerprint == fp else None,
        suggestions=state.ai_suggestions if state.ai_suggestions_fingerprint == fp else None,
        weather_x=weather_x,
        results=results,
    )


def section_counts(state: WizardState, version_id: str) -> dict[str, int]:
    """Sections that have something to show for this version, with how many items each has."""
    return available_sections(inputs_for(state, version_id), version_id)


def preview_line(selected: list[str], counts: dict[str, int]) -> str:
    if not selected:
        return "Nothing is selected."
    parts = [f"{SECTION_TITLES[s]} ({counts[s]})" for s in SECTION_TITLES if s in selected]
    return f"{len(parts)} section(s): " + "; ".join(parts) + "."


def generate_pdf(state: WizardState, version_id: str, selected: list[str]) -> bytes:
    """Build the PDF for a version and keep it in memory."""
    data = build_report_data(inputs_for(state, version_id), version_id, selected)
    pdf = build_pdf(data)
    state.report_pdf = (pdf_key(version_id, selected), pdf)
    return pdf


def pdf_key(version_id: str, selected: list[str]) -> str:
    return version_id + "|" + ",".join(sorted(selected))
