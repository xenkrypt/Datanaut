from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "app.py")


def _at_analyze() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300).run()
    for label in ("Use sample data", "Continue to Profile", "Continue to Map columns"):
        [b for b in at.button if b.label == label][0].click()
        at = at.run()
    [b for b in at.button if b.label == "Build time index and create V1"][0].click()
    at = at.run()
    [b for b in at.button if b.label == "Continue to Analyze"][0].click()
    return at.run()


def test_the_analyze_flow_runs_on_the_sample_and_renders_results():
    at = _at_analyze()
    assert not at.exception and at.header[0].value == "Analyze"
    state = at.session_state["wizard"]
    assert state.step == "Analyze" and "Clean" in state.completed
    assert len(at.tabs) >= 5  # Patterns, Anomalies, Change points, Forecast, Weather
    assert any("Peak periods" == s.value for s in at.subheader)  # patterns render without a click

    at.button(key="run_anom").click()
    at = at.run()
    assert not at.exception
    assert any("differ strongly" in m.value for m in at.markdown)

    at.selectbox(key="cp_cost").select("l2")
    at = at.run()
    at.button(key="run_cp").click()
    at = at.run()
    assert not at.exception
    assert any("change(s) found" in m.value for m in at.markdown)

    at.number_input(key="fc_horizon").set_value(12)
    at = at.run()
    at.button(key="run_fc").click()
    at = at.run()
    assert not at.exception
    assert any("Recommended model" in s.value for s in at.success)
    state = at.session_state["wizard"]
    ops = [e.operation for e in state.log]
    assert {"stl_anomalies", "change_points", "forecast_backtest", "forecast"} <= set(ops)


def test_insights_shows_ranked_findings_and_the_what_if_tool():
    at = _at_analyze()
    [b for b in at.button if b.label == "Continue to Insights"][0].click()
    at = at.run()
    assert not at.exception and at.header[0].value == "Insights"
    at.button(key="gen_findings").click()
    at = at.run()
    assert not at.exception
    state = at.session_state["wizard"]
    assert any(s.value.startswith("1. ") for s in at.subheader)
    assert "Insights" in state.completed
    text = " ".join(m.value for m in at.markdown).lower()
    assert "caused by" not in text
    assert any(m.label == "Peak" for m in at.metric)  # the what-if metrics render


def test_a_weather_file_can_be_set_without_the_network():
    at = _at_analyze()
    state = at.session_state["wizard"]
    from datanaut.ui import analysis

    raw = state.store.current().index
    lines = ["time,temperature_2m"] + [f"{t:%Y-%m-%dT%H:%M},10.0" for t in raw[:48]]
    analysis.set_weather_file(state, ("\n".join(lines) + "\n").encode())
    at = at.run()
    assert not at.exception and state.weather is not None
