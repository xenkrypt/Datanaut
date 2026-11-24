from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "app.py")


def _mapped_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    for key in ("Use sample data", "Continue to Profile", "Continue to Map columns"):
        [b for b in at.button if b.label == key][0].click()
        at = at.run()
    [b for b in at.button if b.label == "Build time index and create V1"][0].click()
    return at.run()


def test_apply_default_fixes_then_undo_on_the_sample():
    at = _mapped_app()
    assert not at.exception and at.header[0].value == "Clean"
    state = at.session_state["wizard"]
    v1, log_before = state.fingerprint, len(state.log)

    at.button(key="apply_sel").click()
    at = at.run()
    assert not at.exception
    state = at.session_state["wizard"]
    assert state.store.head != "V1" and len(state.store.history()) > 1
    assert len(state.log) > log_before and state.fingerprint != v1
    head = state.store.head
    assert any(head in m.value and "current" in m.value for m in at.sidebar.markdown)
    parent_id = state.store.version(head).parent_id
    parent_fp = state.store.version(parent_id).fingerprint

    at.button(key="undo").click()
    at = at.run()
    state = at.session_state["wizard"]
    assert not at.exception and state.store.head == parent_id and state.fingerprint == parent_fp
    assert len(state.store.history()) == int(head[1:])  # nothing was deleted


def test_validation_runs_on_an_accepted_suggested_rule_set():
    at = _mapped_app()
    state = at.session_state["wizard"]
    assert not at.exception and state.step == "Clean"
    first = [b for b in at.button if b.label == "Accept"][0]
    first.click()
    at = at.run()
    assert at.session_state["wizard"].rules
    at.button(key="run_val").click()
    at = at.run()
    state = at.session_state["wizard"]
    assert not at.exception and state.report is not None
    assert len(state.report.results) == len(state.rules)
    assert any(e.operation == "validate" or "valid" in e.operation for e in state.log)


def test_the_version_picker_and_downloads_render_after_applying():
    at = _mapped_app()
    at.button(key="apply_sel").click()
    at = at.run()
    assert not at.exception
    labels = [d.label for d in at.get("download_button")]
    assert {"Cleaned data (CSV)", "Pipeline log (JSON)", "Recipe (JSON)"} <= set(labels)
