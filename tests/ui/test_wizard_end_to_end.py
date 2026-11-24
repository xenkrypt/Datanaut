import builtins
import io
import os
import tempfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from datanaut.core.log import PipelineLog
from datanaut.core.versioning import Recipe, replay_recipe
from datanaut.ui import cleaning, rules

APP = str(Path(__file__).resolve().parents[2] / "app.py")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Empty working and temp folders, plus a record of every file opened for writing."""
    cwd, tmp = tmp_path / "cwd", tmp_path / "tmp"
    cwd.mkdir()
    tmp.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    writes: list[str] = []
    real_open = builtins.open

    def spy(file, mode="r", *args, **kwargs):
        if isinstance(file, str | os.PathLike) and any(c in mode for c in "wax+"):
            writes.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    return cwd, tmp, writes


def _click(at: AppTest, label: str) -> AppTest:
    [b for b in at.button if b.label == label][0].click()
    return at.run()


def test_the_whole_wizard_on_the_sample_without_writing_any_file(sandbox):
    cwd, tmp, writes = sandbox

    at = AppTest.from_file(APP, default_timeout=600).run()
    assert not at.exception and at.header[0].value == "Upload"
    at = _click(at, "Use sample data")
    at = _click(at, "Continue to Profile")
    assert at.header[0].value == "Profile" and not at.exception
    at = _click(at, "Continue to Map columns")
    at.checkbox(key="cons_global_active_power").check()
    at = at.run()
    at = _click(at, "Build time index and create V1")
    assert at.header[0].value == "Clean" and not at.exception

    at.button(key="apply_sel").click()  # the default fixes
    at = at.run()
    at = [b for b in at.button if b.label == "Accept"][0].click().run()
    at.button(key="run_val").click()
    at = at.run()
    assert not at.exception and at.session_state["wizard"].report is not None

    at = _click(at, "Continue to Analyze")
    assert at.header[0].value == "Analyze"
    at.button(key="run_anom").click()
    at = at.run()
    at.selectbox(key="cp_cost").select("l2")
    at = at.run()
    at.button(key="run_cp").click()
    at = at.run()
    at.number_input(key="fc_horizon").set_value(12)
    at = at.run()
    at.button(key="run_fc").click()
    at = at.run()
    assert not at.exception

    at = _click(at, "Continue to Insights")
    at.button(key="gen_findings").click()
    at = at.run()
    assert at.header[0].value == "Insights" and not at.exception
    assert any(s.value.startswith("1. ") for s in at.subheader)

    state = at.session_state["wizard"]
    store = state.store

    # the version chain: V1 ... Vn, each made from the one before, no repeated data
    chain = store.chain()
    assert [v.version_id for v in chain] == [f"V{i}" for i in range(1, len(chain) + 1)]
    assert chain[0].op is None and all(
        v.parent_id == p.version_id for p, v in zip(chain, chain[1:], strict=False)
    )
    assert len(chain) > 1 and len({v.fingerprint for v in chain}) == len(chain)

    # the log: complete, numbered without gaps, and survives a JSON round trip
    ops = [e.operation for e in state.log]
    assert [e.step_id for e in state.log] == list(range(1, len(ops) + 1))
    for expected in (
        "load",
        "profile",
        "build_time_index",
        "drop_duplicates",
        "handle_missing",
        "validate",
        "characterize",
        "stl_anomalies",
        "change_points",
        "forecast_backtest",
        "forecast",
        "findings",
        "peak_shift",
    ):
        assert expected in ops, expected
    assert PipelineLog.from_json(state.log.to_json()).entries == state.log.entries

    # downloads can be built, and the recipe reproduces the working version
    csv = cleaning.cleaned_csv(state)
    assert csv.startswith(b"timestamp") and len(csv.splitlines()) > 100
    assert rules.rules_json(state) and cleaning.log_json(state)
    recipe = Recipe.from_json(cleaning.recipe_json(state))
    replayed = replay_recipe(store.raw(), recipe)
    assert recipe.head_fingerprint == store.version(store.head).fingerprint
    assert replayed is not None

    # privacy: nothing was written, anywhere
    assert list(cwd.iterdir()) == [] and list(tmp.iterdir()) == []
    assert writes == [], writes


def test_the_write_spy_would_catch_a_file_write(sandbox):
    cwd, tmp, writes = sandbox
    with open("leak.txt", "w", encoding="utf-8") as f:
        f.write("x")
    assert writes == ["leak.txt"] and list(cwd.iterdir()) != []
