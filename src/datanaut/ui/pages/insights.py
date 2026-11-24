"""Insights screen: ranked findings, a peak-shifting what-if and an investigation tool."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from datanaut.ui import actions, analysis, charts
from datanaut.ui.pages.common import evidence_box, timing
from datanaut.ui.state import WizardState

STEP = "Insights"


def render(state: WizardState) -> None:
    st.header("Insights")
    st.write(
        "The most notable findings in your data, a what-if for shifting peak load, and a tool to "
        "see what is associated with a change. Findings describe association, not cause."
    )
    if state.store is None:
        st.info("Map your columns first.")
        return
    options = list(state.value_columns)
    state.analysis_column = st.selectbox(
        "Column", options, index=options.index(analysis.column(state)), key="ins_col"
    )
    findings, what_if, investigate = st.tabs(["Findings", "What-if", "Investigate"])
    with findings:
        _findings(state)
    with what_if:
        _what_if(state)
    with investigate:
        _investigate(state)
    st.button("Back to Analyze", on_click=state.go, args=("Analyze",), key="ins_back")


# --- findings -----------------------------------------------------------------------------


def _findings(state: WizardState) -> None:
    st.write(
        "This runs the standard analyses (reusing any you already ran) and ranks what it found. "
        "Change points use a slower method and can take a few seconds."
    )
    if st.button("Generate findings", type="primary", key="gen_findings"):
        with st.spinner("Working..."):
            actions.guard(state, STEP, analysis.findings, state)
            state.mark_done(STEP)
    cached = analysis.peek(state, "findings", weather=state.weather_id)
    if cached is None:
        st.info("Click the button to see ranked findings.")
        return
    report = cached.result
    timing(cached)
    ctx = _chart_context(state)
    if not report.findings:
        st.info("No finding was strong enough to report for this column.")
    for i, f in enumerate(report.findings, start=1):
        with st.container(border=True):
            st.subheader(f"{i}. {f.title}")
            st.write(f.statement)
            fig = charts.finding_chart(f.chart, ctx)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key=f"find_{f.id}")
            evidence_box(f)
    evidence_box(report)


def _chart_context(state: WizardState) -> dict:
    """The cached results the finding charts draw from (nothing is computed here)."""
    p = analysis.period(state)

    def got(name: str, **params):
        r = analysis.peek(state, name, **params)
        return None if r is None else r.result

    weather = got("weather_link", weather=state.weather_id, variable=None)
    ctx = {
        "series": analysis.series(state),
        "peaks": got("peaks", percentile=95.0),
        "weekday": got("profile_weekday"),
        "anomalies": got("stl", threshold=3.5, period=p),
        "changes": got("change_points", sensitivity="medium", cost="rbf", period=p),
        "link": weather,
        "forecast": got("forecast", horizon=24, model="seasonal_naive", period=p)
        or got("forecast", horizon=24, model="holt_winters", period=p),
    }
    if weather is not None:
        ctx["weather_x"] = analysis.aligned_weather(state, weather.variable)
    return ctx


# --- what-if ------------------------------------------------------------------------------


def _what_if(state: WizardState) -> None:
    st.write(
        "A scenario: move part of the load above a high-load threshold to the quietest "
        "readings of the same day. Total energy stays the same."
    )
    left, right = st.columns(2)
    fraction = left.slider("Share of the peak to shift", 0.0, 1.0, 0.2, 0.05, key="wi_frac")
    pct = right.slider("Peak threshold (percentile)", 80, 99, 95, key="wi_pct")
    use_tariff = st.checkbox("Estimate a cost saving with my own rates", key="wi_tariff")
    peak_rate = offpeak_rate = None
    currency = ""
    if use_tariff:
        a, b, c = st.columns(3)
        peak_rate = a.number_input("Peak rate per unit", 0.0, value=0.30, key="wi_peak")
        offpeak_rate = b.number_input("Off-peak rate per unit", 0.0, value=0.15, key="wi_off")
        currency = c.text_input("Currency (optional)", key="wi_cur")
    res = actions.guard(
        state,
        STEP,
        analysis.what_if,
        state,
        fraction,
        float(pct),
        peak_rate,
        offpeak_rate,
        currency,
    )
    if res is None:
        return
    shift = res.result
    timing(res)
    cols = st.columns(3)
    cols[0].metric("Peak", _num(shift.peak_after), _delta(shift.peak_change, shift.peak_change_pct))
    cols[1].metric(
        "Load factor",
        _num(shift.load_factor_after, 3),
        _num(shift.load_factor_change, 3, signed=True),
    )
    cols[2].metric("Energy moved", _num(shift.energy_moved))
    st.plotly_chart(
        charts.whatif_chart(analysis.series(state), shift.new_series, shift.column),
        use_container_width=True,
    )
    st.caption(f"{shift.label.capitalize()}.")
    with st.expander("Assumptions"):
        for line in shift.assumptions:
            st.write(f"- {line}")
    if shift.cost is not None:
        st.write(
            f"Estimated saving: **{shift.cost.estimated_saving:,.2f} {shift.cost.currency}**. "
            f"{shift.cost.note}"
        )
    evidence_box(shift)


def _num(value: float | None, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}g}" if signed else f"{value:,.{digits}g}"


def _delta(change: float | None, pct: float | None) -> str | None:
    if change is None:
        return None
    return f"{change:+.3g}" + (f" ({pct:+.1f}%)" if pct is not None else "")


# --- investigation ------------------------------------------------------------------------


def _investigate(state: WizardState) -> None:
    st.write(
        "Pick a period and see which factors are associated with how it differs from a "
        "baseline. The result describes association, not cause."
    )
    series = analysis.series(state)
    first, last = series.index.min(), series.index.max()
    events = analysis.event_options(state)
    labels = ["Choose dates myself"] + [e[0] for e in events]
    choice = st.selectbox("Period", labels, key="inv_event")
    default_start, default_end = last - pd.Timedelta(days=7), last
    if choice != labels[0]:
        _, s, e = events[labels.index(choice) - 1]
        default_start, default_end = pd.Timestamp(s), pd.Timestamp(e)
    a, b = st.columns(2)
    start = a.date_input(
        "From",
        default_start.date(),
        min_value=first.date(),
        max_value=last.date(),
        key=f"inv_from_{choice}",
    )
    end = b.date_input(
        "To",
        default_end.date(),
        min_value=first.date(),
        max_value=last.date(),
        key=f"inv_to_{choice}",
    )
    baseline = st.radio(
        "Compare with",
        ["previous", "same_weekdays"],
        format_func=lambda x: (
            "the period just before" if x == "previous" else "the same weekdays in earlier weeks"
        ),
        horizontal=True,
        key="inv_base",
    )
    if not events:
        st.caption(
            "Run anomaly detection or change points on the Analyze screen to list them here."
        )
    if st.button("Investigate", type="primary", key="run_inv"):
        stop = pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        with st.spinner("Working..."):
            actions.guard(
                state, STEP, analysis.run_investigation, state, pd.Timestamp(start), stop, baseline
            )
    stop = pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    shown = analysis.peek(
        state,
        "investigate",
        start=str(pd.Timestamp(start)),
        end=str(stop),
        baseline=baseline,
        weather=state.weather_id,
        with_change_points=bool(analysis._cached_change_points(state)),
    )
    if shown is None:
        return
    res = shown.result
    timing(shown)
    ch = res.change
    pct = f" ({ch.percent:+.1f}%)" if ch.percent is not None else ""
    st.write(
        f"Average {res.column} was **{ch.mean_period:,.4g}** in the period against "
        f"**{ch.mean_baseline:,.4g}** in {res.baseline.description}: {ch.direction}{pct}."
    )
    st.dataframe(
        [
            {
                "Check": f.name.replace("_", " "),
                "Verdict": f.verdict,
                "Effect": None if f.effect_size is None else round(f.effect_size, 3),
                "Measure": f.effect_measure,
                "What it shows": f.statement,
            }
            for f in res.factors
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.info(res.conclusion)
    evidence_box(res)
