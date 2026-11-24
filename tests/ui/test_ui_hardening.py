import ast
import logging
import tomllib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from datanaut.errors import FileTooLargeError
from datanaut.ui import actions, limits
from datanaut.ui.state import STEPS, WizardState

ROOT = Path(__file__).resolve().parents[2]
APP = str(ROOT / "app.py")
SECRET = "secret-cell-value-4711"


def _mapped() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    for label in ("Use sample data", "Continue to Profile", "Continue to Map columns"):
        [b for b in at.button if b.label == label][0].click()
        at = at.run()
    [b for b in at.button if b.label == "Build time index and create V1"][0].click()
    return at.run()


# --- errors never reach the user as tracebacks ---------------------------------------------


def test_a_failing_screen_becomes_a_friendly_message_and_logs_no_values(caplog):
    state = WizardState()

    def broken(_):
        raise RuntimeError(f"bad cell {SECRET}")

    with caplog.at_level(logging.ERROR, logger="datanaut.ui"):
        ok = actions.guarded_render(state, "Clean", broken)
    assert ok is False
    text = state.messages[-1].text
    assert "Clean" in text and "Go back" in text and SECRET not in text
    assert "RuntimeError" in caplog.text and SECRET not in caplog.text


def test_the_app_shows_a_message_and_recovery_buttons_when_a_screen_crashes(monkeypatch, caplog):
    from datanaut.ui.pages import upload

    def broken(_state):
        raise RuntimeError(f"bad cell {SECRET}")

    monkeypatch.setattr(upload, "render", broken)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception  # no traceback block
    assert any("unexpected" in e.value for e in at.error)
    assert SECRET not in " ".join(e.value for e in at.error) and SECRET not in caplog.text
    labels = {b.label for b in at.button}
    assert "Start over with a new file" in labels and "Go back to Upload" in labels
    assert any("Steps" == s.value for s in at.sidebar.subheader)  # the sidebar still draws


def test_core_errors_in_a_click_keep_the_session_usable():
    state = WizardState()
    assert actions.guard(state, "Upload", actions.load_upload, state, b"\x00\x01\x02", "x") is None
    assert state.loaded is None and state.messages[-1].level == "error"
    assert actions.load_sample(state) and state.loaded is not None


# --- limits --------------------------------------------------------------------------------


def test_row_and_column_limits_are_enforced_when_loading(monkeypatch):
    monkeypatch.setattr(limits, "MAX_ROWS", 5)
    csv = ("t,v\n" + "\n".join(f"2024-01-0{d},{d}" for d in range(1, 9)) + "\n").encode()
    with pytest.raises(FileTooLargeError, match="rows"):
        actions.load_upload(WizardState(), csv, "a.csv")
    monkeypatch.setattr(limits, "MAX_ROWS", 1_000_000)
    monkeypatch.setattr(limits, "MAX_COLUMNS", 2)
    wide = b"t,a,b\n2024-01-01,1,2\n2024-01-02,3,4\n"
    with pytest.raises(FileTooLargeError, match="columns"):
        actions.load_upload(WizardState(), wide, "w.csv")


def test_the_server_config_matches_the_limit_in_code():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["maxUploadSize"] * 1_000_000 == limits.MAX_UPLOAD_BYTES
    assert config["browser"]["gatherUsageStats"] is False
    assert config["client"]["toolbarMode"] == "minimal"


# --- sessions are isolated -----------------------------------------------------------------


def test_two_app_sessions_do_not_share_state():
    a = AppTest.from_file(APP, default_timeout=60).run()
    b = AppTest.from_file(APP, default_timeout=60).run()
    [x for x in a.button if x.label == "Use sample data"][0].click()
    a = a.run()
    assert a.session_state["wizard"].loaded is not None
    assert b.session_state["wizard"].loaded is None
    assert a.session_state["wizard"] is not b.session_state["wizard"]


def _module_level_containers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        value = getattr(node, "value", None)
        mutable = isinstance(value, ast.List | ast.Dict | ast.Set | ast.ListComp | ast.DictComp)
        mutable |= (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in {"list", "dict", "set", "OrderedDict", "defaultdict"}
        )
        if mutable:
            found |= {t.id for t in targets if isinstance(t, ast.Name)}
    return found


def test_there_is_no_module_level_mutable_state_beyond_read_only_lookup_tables():
    files = list((ROOT / "src" / "datanaut" / "ui").rglob("*.py")) + [ROOT / "app.py"]
    found = set().union(*(_module_level_containers(f) for f in files))
    assert found <= {"_PREREQUISITES", "_ICONS", "_PAGES", "_MARGIN", "_BASE", "_SCALE", "_KIND"}, (
        found
    )  # fixed tables, never changed


def test_only_app_and_pages_import_streamlit():
    ui = ROOT / "src" / "datanaut" / "ui"
    importing = {
        p.relative_to(ui).as_posix()
        for p in ui.rglob("*.py")
        if "import streamlit" in p.read_text(encoding="utf-8")
    }
    assert all(name.startswith("pages/") for name in importing), importing


def test_the_formal_title_is_only_where_it_was_allowed():
    title = "An AI-Powered Intelligent Platform for Time-Series Data Analysis"
    hits = {
        p.relative_to(ROOT).as_posix()
        for pattern in ("**/*.py", "**/*.toml")
        for p in ROOT.glob(pattern)
        if ".venv" not in p.parts and "tests" not in p.parts and title in p.read_text("utf-8")
    }
    assert hits == {
        "pyproject.toml",
        "app.py",
        "src/datanaut/__init__.py",
        "src/datanaut/cli.py",
    }


# --- clarity -------------------------------------------------------------------------------


def test_every_screen_opens_with_a_one_sentence_explanation():
    at = _mapped()
    seen = []
    for step in STEPS:
        at.sidebar.button(key=f"nav_{step}").click()
        at = at.run()
        assert not at.exception
        assert at.header[0].value == step
        first = next(m.value for m in at.main.markdown if "dn-board" not in m.value)
        assert first.rstrip().endswith(".") and len(first) > 30, (step, first)
        seen.append(step)
    assert seen == list(STEPS)


def test_locked_steps_say_why_and_working_buttons_say_what_they_do():
    at = AppTest.from_file(APP, default_timeout=60).run()
    locked = [b for b in at.sidebar.button if b.disabled]
    assert locked and all("earlier steps" in (b.help or "") for b in locked)
    labels = [b.label for b in at.button if not b.disabled]
    assert "Use sample data" in labels
