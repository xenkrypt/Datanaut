from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from datanaut.ui import charts

APP = str(Path(__file__).resolve().parents[2] / "app.py")


def test_missingness_heatmap_has_a_title_axis_labels_and_one_trace():
    m = pd.DataFrame({"a": [0.0, 0.5], "b": [1.0, 0.0]}, index=["t1", "t2"])
    fig = charts.missingness_heatmap(m)
    assert len(fig.data) == 1 and fig.layout.title.text
    assert fig.layout.xaxis.title.text and fig.layout.yaxis.title.text
    assert fig.data[0].z.shape == (2, 2)


def _click(at: AppTest, label: str) -> AppTest:
    [b for b in at.button if b.label == label][0].click()
    return at.run()


def test_the_app_starts_loads_the_sample_and_renders_profile_and_map_columns():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and at.header[0].value == "Upload"
    assert at.session_state["wizard"].loaded is None

    at = _click(at, "Use sample data")
    assert not at.exception
    assert at.session_state["wizard"].loaded is not None
    assert any("Rows" == m.label for m in at.metric)

    at = _click(at, "Continue to Profile")
    assert not at.exception and at.header[0].value == "Profile"
    assert any(m.label == "Missing values" for m in at.metric)

    at = _click(at, "Continue to Map columns")
    assert not at.exception and at.header[0].value == "Map columns"
    at = _click(at, "Build time index and create V1")
    assert not at.exception
    state = at.session_state["wizard"]
    assert state.store.head == "V1" and state.step == "Clean"
    assert any("Time index built" in s.value for s in at.success)


def test_the_sidebar_shows_the_privacy_note_and_locks_unreachable_steps():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert any("not stored" in c.value for c in at.sidebar.caption)
    nav = {b.label: b for b in at.sidebar.button}
    assert nav["▶ 1. Upload"].disabled is False
    assert nav["🔒 4. Clean"].disabled is True
