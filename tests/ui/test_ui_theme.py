import re
import tomllib
from pathlib import Path

from datanaut.ui import actions, overview, theme
from datanaut.ui.state import WizardState

ROOT = Path(__file__).resolve().parents[2]


def test_the_dark_theme_is_configured():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["theme"]["base"] == "dark"
    assert config["theme"]["backgroundColor"].startswith("#0")


def test_motion_uses_the_compositor_and_respects_reduced_motion():
    css = theme.CSS
    assert "prefers-reduced-motion: reduce" in css and "animation-duration: .001ms" in css
    frames = re.findall(r"@keyframes ([\w-]+) \{(.*?)\}\s*(?=\n|$)", css, flags=re.S)
    assert frames
    animated = set()
    bodies = re.findall(r"@keyframes [\w-]+ \{((?:[^{}]|\{[^{}]*\})*)\}", css)
    assert len(bodies) >= 8
    for body in bodies:
        animated |= set(re.findall(r"([a-z-]+)\s*:", body))
    # layout properties are never animated; only transform, opacity, strokes and the counters
    assert not animated & {
        "width",
        "height",
        "top",
        "left",
        "margin",
        "padding",
        "box-shadow",
        "filter",
    }
    assert {"transform", "opacity"} <= animated


def test_dashboard_text_is_escaped():
    evil = '<img src=x onerror="alert(1)">'
    for card in (
        theme.count_card(evil, 5, evil),
        theme.ring_card(evil, 12.5, evil),
        theme.spark_card(evil, [1, 2, 3], evil),
    ):
        assert "<img" not in card and "&lt;img" in card
    assert "<img" not in theme.board([theme.count_card(evil, 1)])


def test_count_cards_split_the_number_for_the_counter():
    assert "--ta:2876;--tb:0" in theme.count_card("Rows", 2876)
    assert "--ta:12;--tb:34" in theme.count_card("Share", 12.34)
    assert "2,876" in theme.count_card("Rows", 2876)  # screen readers get the real number
    ring = theme.ring_card("Missing", 1.02)
    assert "--ta:1;--tb:2" in ring and "--off:98.98" in ring


def test_sparklines_scale_and_survive_flat_or_short_data():
    pts = theme.sparkline_points([1, 5, 3, 9])
    assert len(pts) == 4 and pts[0][0] == 0 and pts[-1][0] == 120
    assert all(0 <= y <= 40 for _, y in pts)
    assert theme.sparkline_points([2, 2, 2])[0][1] == 20  # flat: a centre line
    assert theme.sparkline_points([1]) == [] and "polyline" not in theme.spark_card("x", [1])
    assert theme.sparkline_points([1, float("nan"), 3]) != []


def test_the_overview_shows_only_after_mapping_and_follows_the_data():
    s = WizardState()
    assert overview.overview_html(s) is None
    actions.load_sample(s)
    assert overview.overview_html(s) is None
    time_cols, values = actions.suggest_mapping(s.loaded.df)
    actions.confirm_mapping(s, time_cols, None, values, {})
    html = overview.overview_html(s)
    assert html.count('class="dn-card"') == 5 and "Readings" in html and "polyline" in html
    assert overview.overview_html(s) == html  # stable between reruns
