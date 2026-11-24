"""Analyze screen: patterns, anomalies, change points, forecast and weather for one column."""

from __future__ import annotations

import streamlit as st

from datanaut.ui import actions, analysis, charts
from datanaut.ui.pages.common import evidence_box, run_button, timing
from datanaut.ui.state import WizardState

STEP = "Analyze"


def render(state: WizardState) -> None:
    st.header("Analyze")
    st.write(
        "Explore one column: its daily pattern, unusual readings, sudden changes, a short "
        "forecast and its link to the weather. Results describe association, not cause."
    )
    if state.store is None:
        st.info("Map your columns first.")
        return
    options = list(state.value_columns)
    chosen = st.selectbox("Column to analyze", options, index=options.index(analysis.column(state)))
    state.analysis_column = chosen
    char = actions.guard(state, STEP, analysis.characterization, state)
    if char is not None:
        c = char.result
        st.caption(
            f"{c.frequency_label} readings, {c.n_observations:,} observations; "
            f"seasonal period: {c.seasonal_period or 'none detected'}."
        )

    tabs = st.tabs(["Patterns", "Anomalies", "Change points", "Forecast", "Weather"])
    with tabs[0]:
        _patterns(state)
    with tabs[1]:
        _anomalies(state)
    with tabs[2]:
        _change_points(state)
    with tabs[3]:
        _forecast(state)
    with tabs[4]:
        _weather(state)
    st.button(
        "Continue to Insights",
        type="primary",
        on_click=_continue,
        args=(state,),
        key="an_next",
    )


def _continue(state: WizardState) -> None:
    state.mark_done(STEP)
    state.go("Insights")


# --- patterns -----------------------------------------------------------------------------


def _patterns(state: WizardState) -> None:
    hour = actions.guard(state, STEP, analysis.hour_profile, state)
    weekday = actions.guard(state, STEP, analysis.weekday_profile, state)
    if hour is not None:
        st.plotly_chart(charts.profile_bars(hour.result), use_container_width=True)
        evidence_box(hour.result)
    if weekday is not None:
        st.plotly_chart(charts.profile_bars(weekday.result), use_container_width=True)
    heat = actions.guard(state, STEP, analysis.heat_profile, state)
    if heat is not None:
        st.plotly_chart(charts.profile_heatmap(heat.result), use_container_width=True)

    st.subheader("Energy metrics")
    m = actions.guard(state, STEP, analysis.metrics, state)
    if m is not None:
        res = m.result
        if not res.applicable:
            st.info(
                res.reason
                or "Mark this column as consumption on the Map columns step to see energy metrics."
            )
        else:
            for col, (label, value) in zip(
                st.columns(4),
                [
                    ("Peak", res.peak),
                    ("Baseload", res.baseload),
                    ("Mean", res.mean),
                    ("Load factor", res.load_factor),
                ],
                strict=True,
            ):
                col.metric(label, "n/a" if value is None else f"{value:,.3g}")
            for note in res.notes:
                st.caption(note)
        evidence_box(res)

    st.subheader("Peak periods")
    pct = st.slider("Count a reading as a peak above this percentile", 80, 99, 95, key="peak_pct")
    pk = actions.guard(state, STEP, analysis.peaks, state, float(pct))
    if pk is not None:
        res = pk.result
        st.write(
            f"Peaks are most common around **{', '.join(res.top_hours) or 'n/a'}** "
            f"and on **{', '.join(res.top_weekdays) or 'n/a'}**."
        )
        left, right = st.columns(2)
        left.plotly_chart(
            charts.share_bars(res.hours, f"{res.column}: peaks by hour", "Hour of day"),
            use_container_width=True,
        )
        right.plotly_chart(
            charts.share_bars(res.weekdays, f"{res.column}: peaks by weekday", "Weekday"),
            use_container_width=True,
        )
        evidence_box(res)


# --- anomalies ----------------------------------------------------------------------------


def _anomalies(state: WizardState) -> None:
    st.write("Two checks: one against the column's own pattern, one across all columns together.")
    left, right = st.columns(2)
    thr = left.slider(
        "Pattern check threshold (higher flags fewer)", 2.5, 6.0, 3.5, 0.5, key="an_thr"
    )
    contamination = right.slider(
        "Share of rows the multi-column check may flag", 0.005, 0.05, 0.01, 0.005, key="an_contam"
    )
    if st.button("Run anomaly detection", type="primary", key="run_anom"):
        with st.spinner("Working..."):
            actions.guard(state, STEP, analysis.stl, state, thr)
            if len(state.value_columns) >= 2:
                actions.guard(state, STEP, analysis.multivariate, state, contamination)
                actions.guard(state, STEP, analysis.detector_comparison, state, thr, contamination)
    period = analysis.period(state)
    stl = analysis.peek(state, "stl", threshold=thr, period=period)
    if stl is None:
        st.info("Click the button to look for unusual readings.")
        return
    res = stl.result
    timing(stl)
    st.write(f"**{res.n_anomalies:,}** reading(s) differ strongly from the usual pattern.")
    st.plotly_chart(
        charts.time_series(
            analysis.series(state),
            res.column,
            res.flagged_timestamps(),
            title=f"{res.column}: pattern-based anomalies",
        ),
        use_container_width=True,
    )
    if res.points:
        st.dataframe(
            [
                {
                    "Time": p.timestamp,
                    "Value": p.value,
                    "Away from pattern by": p.residual,
                    "Score": p.score,
                    "Direction": p.direction,
                }
                for p in res.points
            ],
            hide_index=True,
            use_container_width=True,
        )
    comps = actions.guard(state, STEP, analysis.stl_components, state)
    if comps is not None:
        with st.expander("Trend and seasonal components"):
            st.plotly_chart(
                charts.stl_components(comps[0], comps[1], res.column), use_container_width=True
            )
    evidence_box(res)

    multi = analysis.peek(
        state, "multivariate", contamination=contamination, columns=list(state.value_columns)
    )
    if multi is not None:
        _multivariate(state, multi.result, thr, contamination)
    elif len(state.value_columns) < 2:
        st.caption("The multi-column check needs at least two value columns.")


def _multivariate(state: WizardState, res, thr: float, contamination: float) -> None:
    st.subheader("Across all columns together")
    if not res.applicable:
        st.info(res.reason or "The multi-column check could not run on this data.")
        return
    st.write(f"**{res.n_flagged:,}** row(s) have an unusual combination of values.")
    if res.points:
        st.plotly_chart(charts.multivariate_view(res), use_container_width=True)
        st.dataframe(
            [
                {
                    "Time": p.timestamp,
                    "Score": p.score,
                    "Main contributors": ", ".join(c.column for c in p.contributions[:2]),
                }
                for p in res.points
            ],
            hide_index=True,
            use_container_width=True,
        )
    cmp = analysis.peek(state, "compare_detectors", threshold=thr, contamination=contamination)
    if cmp is not None:
        st.plotly_chart(charts.detector_bars(cmp.result), use_container_width=True)
        st.caption(cmp.result.note)
    evidence_box(res)


# --- change points ------------------------------------------------------------------------


def _change_points(state: WizardState) -> None:
    st.write("Look for moments when the level or the variability of the readings shifts.")
    left, right = st.columns(2)
    sens = left.selectbox("Sensitivity", ["low", "medium", "high"], index=1, key="cp_sens")
    cost = right.selectbox(
        "Method",
        ["rbf", "l2"],
        format_func=lambda c: "Level and spread (slower)" if c == "rbf" else "Level only (faster)",
        key="cp_cost",
    )
    if st.button("Find change points", type="primary", key="run_cp"):
        with st.spinner("Working..."):
            actions.guard(state, STEP, analysis.change_points, state, sens, cost)
    cp = analysis.peek(
        state, "change_points", sensitivity=sens, cost=cost, period=analysis.period(state)
    )
    if cp is None:
        st.info("Click the button to look for changes.")
        return
    res = cp.result
    timing(cp)
    st.write(f"**{res.n_breaks}** change(s) found.")
    st.plotly_chart(
        charts.change_point_segments(analysis.series(state), res), use_container_width=True
    )
    for note in res.notes:
        st.caption(note)
    if res.breaks:
        st.dataframe(
            [
                {
                    "When": b.timestamp,
                    "Mean before": b.mean_before,
                    "Mean after": b.mean_after,
                    "Change": b.mean_change,
                    "Kind": b.kind,
                    "Direction": b.direction,
                }
                for b in res.breaks
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.caption(res.p_value_note)
    st.dataframe(
        [{"From": s.start, "To": s.end, "Mean": s.mean, "Readings": s.n} for s in res.segments],
        hide_index=True,
        use_container_width=True,
    )
    evidence_box(res)


# --- forecast -----------------------------------------------------------------------------


def _forecast(state: WizardState) -> None:
    st.write("Test simple models on the end of your data, then forecast with the best one.")
    horizon = st.number_input("How many readings ahead", 1, 500, 24, key="fc_horizon")
    if st.button("Run backtest and forecast", type="primary", key="run_fc"):
        with st.spinner("Working..."):
            actions.guard(state, STEP, analysis.run_forecast, state, int(horizon))
    p = analysis.period(state)
    bt = analysis.peek(state, "backtest", horizon=int(horizon), period=p)
    if bt is None:
        st.info("Click the button to compare models and forecast.")
        return
    timing(bt)
    scores = [
        {"Model": s.model, "MAE": s.mae, "RMSE": s.rmse, "sMAPE %": s.smape}
        for s in bt.result.scores
    ]
    st.dataframe(scores, hide_index=True, use_container_width=True)
    for f in bt.result.failures:
        st.caption(f"{f.model}: {f.reason}")
    st.plotly_chart(charts.backtest_bars(bt.result), use_container_width=True)
    rec = analysis.recommend(bt.result)
    st.success(f"Recommended model: {rec.model}. {rec.reason}")
    fc = analysis.peek(state, "forecast", horizon=int(horizon), model=rec.model, period=p)
    if fc is not None:
        st.plotly_chart(
            charts.forecast_chart(analysis.series(state), fc.result), use_container_width=True
        )
        st.caption(fc.result.note)
        evidence_box(fc.result)
    evidence_box(bt.result)


# --- weather ------------------------------------------------------------------------------


def _weather(state: WizardState) -> None:
    st.write("Check whether the readings are associated with the weather (temperature).")
    left, right = st.columns(2)
    lat = left.number_input("Latitude", -90.0, 90.0, 52.5, key="wx_lat")
    lon = right.number_input("Longitude", -180.0, 180.0, 13.4, key="wx_lon")
    start, end = analysis.date_range(state)
    st.caption(
        f"Weather will be requested for {start} to {end}. Only the coordinates and these "
        "dates are sent."
    )
    if st.button("Fetch weather", key="wx_fetch"):
        with st.spinner("Contacting the weather service..."):
            if actions.guard(state, STEP, analysis.fetch_weather, state, lat, lon) is None and (
                state.weather is None
            ):
                state.add_message("info", "You can upload a weather CSV file instead.")
    upload = st.file_uploader(
        "Or upload a weather CSV (first column time, then values)", type=["csv"], key="wx_file"
    )
    if upload is not None and st.button("Use this weather file", key="wx_use"):
        actions.guard(state, STEP, analysis.set_weather_file, state, upload.getvalue())
    if state.weather is None:
        st.info("No weather data yet.")
        return
    variables = list(state.weather.columns)
    variable = st.selectbox(
        "Weather variable",
        variables,
        index=variables.index("temperature_2m") if "temperature_2m" in variables else 0,
        key="wx_var",
    )
    run_button(state, STEP, "Analyze the weather link", "wx_run", analysis.weather_result, variable)
    shown = analysis.peek(state, "weather_link", weather=state.weather_id, variable=variable)
    if shown is None:
        return
    link = shown.result
    timing(shown)
    st.write(
        f"Readings are associated with **{link.variable}**: Pearson {link.pearson:.2f}, "
        f"Spearman {link.spearman:.2f} over {link.n_pairs:,} paired readings."
        if link.pearson is not None and link.spearman is not None
        else "Too few paired readings to measure an association."
    )
    st.caption(link.caveat)
    x = analysis.aligned_weather(state, link.variable)
    if x is not None:
        st.plotly_chart(
            charts.weather_scatter(x, analysis.series(state), link.variable, link.column),
            use_container_width=True,
        )
    st.plotly_chart(charts.lag_bars(link), use_container_width=True)
    if link.best_lag_hours is not None:
        st.write(
            f"The strongest association is when the weather leads by {link.best_lag_hours:g} h."
        )
    evidence_box(link)
