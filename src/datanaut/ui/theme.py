# ruff: noqa: E501  (the stylesheet below is CSS text with long rules)
"""Look and motion: the dark theme's CSS and the animated dashboard cards (plain strings).

No streamlit import. Motion uses only `transform` and `opacity` (which the browser can run on
the GPU, off the main thread) plus a few registered custom properties for number count-ups.
Everything is switched off for people who ask their system for reduced motion.
User-controlled text (column names) is always HTML-escaped before it is put in markup.
"""

from __future__ import annotations

import html
from typing import Any

import numpy as np

ACCENT = "#7c9cff"
ACCENT_2 = "#b28cff"
GOOD = "#4ade80"
WARN = "#ffb454"

CSS = """
@property --a { syntax: "<integer>"; initial-value: 0; inherits: false; }
@property --b { syntax: "<integer>"; initial-value: 0; inherits: false; }

:root {
  --dn-bg: #0a0c10;
  --dn-panel: rgba(255,255,255,0.035);
  --dn-panel-hi: rgba(255,255,255,0.065);
  --dn-line: rgba(255,255,255,0.09);
  --dn-text: #e6e9ef;
  --dn-dim: #8d95a5;
  --dn-accent: #7c9cff;
  --dn-accent-2: #b28cff;
  --dn-ease: cubic-bezier(.22,1,.36,1);
}

html { scroll-behavior: smooth; }
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] > [data-testid="stMarkdownContainer"] > style) { display: none; }
.stApp, .stApp * { -webkit-font-smoothing: antialiased; }
.stApp {
  font-family: "Inter","SF Pro Display","Segoe UI Variable","Segoe UI",system-ui,sans-serif;
}

/* ---- ambient background: two soft light pools drifting on the compositor ---- */
.stApp::before, .stApp::after {
  content: ""; position: fixed; z-index: 0; pointer-events: none;
  width: 70vmax; height: 70vmax; border-radius: 50%; opacity: .5;
  will-change: transform;
}
.stApp::before {
  top: -35vmax; left: -20vmax;
  background: radial-gradient(closest-side, rgba(124,156,255,.16), transparent 70%);
  animation: dn-drift-a 38s ease-in-out infinite alternate;
}
.stApp::after {
  bottom: -40vmax; right: -25vmax;
  background: radial-gradient(closest-side, rgba(178,140,255,.12), transparent 70%);
  animation: dn-drift-b 46s ease-in-out infinite alternate;
}
@keyframes dn-drift-a { to { transform: translate3d(18vmax, 12vmax, 0) scale(1.15); } }
@keyframes dn-drift-b { to { transform: translate3d(-16vmax, -10vmax, 0) scale(1.1); } }
[data-testid="stMain"], [data-testid="stSidebar"] { position: relative; z-index: 1; }

/* ---- page entrance: elements rise in one after another ---- */
@keyframes dn-rise { from { opacity: 0; transform: translate3d(0, 14px, 0); } to { opacity: 1; transform: none; } }
[data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"],
[data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] {
  animation: dn-rise .6s var(--dn-ease) both;
}
[data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"] > :nth-child(2) { animation-delay: .04s; }
[data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"] > :nth-child(3) { animation-delay: .08s; }
[data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"] > :nth-child(4) { animation-delay: .12s; }
[data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"] > :nth-child(5) { animation-delay: .16s; }
[data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"] > :nth-child(n+6) { animation-delay: .2s; }

/* ---- type ---- */
h1 { width: fit-content; letter-spacing: -.03em; font-weight: 700;
  background: linear-gradient(100deg, #fff 30%, var(--dn-accent) 70%, var(--dn-accent-2));
  -webkit-background-clip: text; background-clip: text; color: transparent !important; }
h2, h3 { letter-spacing: -.02em; font-weight: 600; }
[data-testid="stCaptionContainer"] { color: var(--dn-dim); }

/* ---- buttons ---- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  position: relative; overflow: hidden; isolation: isolate;
  border-radius: 12px; border: 1px solid var(--dn-line);
  background: var(--dn-panel); color: var(--dn-text);
  transition: transform .25s var(--dn-ease), box-shadow .25s var(--dn-ease),
              border-color .25s var(--dn-ease), background-color .25s var(--dn-ease);
  will-change: transform;
}
/* a light sweep crosses the button on hover */
.stButton > button::after, .stDownloadButton > button::after {
  content: ""; position: absolute; inset: 0; z-index: -1; pointer-events: none;
  background: linear-gradient(105deg, transparent 30%, rgba(255,255,255,.14) 50%, transparent 70%);
  transform: translate3d(-120%, 0, 0);
  transition: transform .7s var(--dn-ease);
}
.stButton > button:hover:not(:disabled)::after,
.stDownloadButton > button:hover:not(:disabled)::after { transform: translate3d(120%, 0, 0); }
.stButton > button:hover:not(:disabled), .stDownloadButton > button:hover:not(:disabled) {
  transform: translate3d(0, -2px, 0);
  border-color: rgba(124,156,255,.55);
  box-shadow: 0 8px 28px -10px rgba(124,156,255,.55);
}
.stButton > button:active:not(:disabled), .stDownloadButton > button:active:not(:disabled) {
  transform: translate3d(0, 0, 0) scale(.97); transition-duration: .08s;
}
.stButton > button:focus-visible { outline: 2px solid var(--dn-accent); outline-offset: 3px; }
.stButton > button:disabled { opacity: .45; }

/* the primary button wears a slowly turning ring of light */
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
  border: 0; background: transparent; color: #fff; font-weight: 600;
}
.stButton > button[kind="primary"]::before, [data-testid="stBaseButton-primary"]::before {
  content: ""; position: absolute; z-index: -2; left: 50%; top: 50%;
  width: 220%; aspect-ratio: 1; margin: -110% 0 0 -110%;
  background: conic-gradient(from 0deg, var(--dn-accent), var(--dn-accent-2), #5eead4, var(--dn-accent));
  animation: dn-spin 5s linear infinite; will-change: transform;
}
@keyframes dn-spin { to { transform: rotate(360deg); } }
.stButton > button[kind="primary"]::after, [data-testid="stBaseButton-primary"]::after {
  inset: 1.5px; border-radius: 10.5px; z-index: -1;
  background: linear-gradient(180deg, #1a2030, #11151f); transform: none; transition: opacity .3s;
}
.stButton > button[kind="primary"]:hover:not(:disabled)::after { transform: none; opacity: .86; }
.stButton > button[kind="primary"]:disabled::before { animation-play-state: paused; opacity: .35; }

/* ---- sidebar steps ---- */
[data-testid="stSidebar"] { border-right: 1px solid var(--dn-line); }
[data-testid="stSidebar"] .stButton > button {
  justify-content: flex-start; background: transparent; border-color: transparent; text-align: left;
}
[data-testid="stSidebar"] .stButton > button > div, [data-testid="stSidebar"] .stButton > button p { justify-content: flex-start; text-align: left; width: 100%; }
[data-testid="stSidebar"] .stButton > button:hover:not(:disabled) {
  background: var(--dn-panel-hi); transform: translate3d(4px, 0, 0); box-shadow: none;
}

/* ---- tabs: a pill pops in under the active tab ---- */
[role="tablist"] {
  gap: 4px !important; padding: 4px; border-radius: 14px; background: var(--dn-panel);
  border: 1px solid var(--dn-line); width: fit-content; max-width: 100%; overflow-x: auto;
}
[data-testid="stTab"] {
  position: relative; isolation: isolate; border-radius: 10px; padding: 7px 14px !important;
  color: var(--dn-dim); transition: color .25s var(--dn-ease), transform .25s var(--dn-ease);
}
[data-testid="stTab"] p { margin: 0; }
[data-testid="stTab"]:hover { color: var(--dn-text); transform: translate3d(0, -1px, 0); }
[data-testid="stTab"][aria-selected="true"] { color: #fff; }
.react-aria-SelectionIndicator {
  inset: 0 !important; width: auto !important; height: auto !important; z-index: -1; border-radius: 10px;
  background: linear-gradient(135deg, rgba(124,156,255,.30), rgba(178,140,255,.24)) !important;
  box-shadow: inset 0 0 0 1px rgba(124,156,255,.5), 0 6px 22px -8px rgba(124,156,255,.65);
  animation: dn-pop .5s var(--dn-ease) both; will-change: transform, opacity;
}
@keyframes dn-pop { from { opacity: 0; transform: scale(.72); } to { opacity: 1; transform: none; } }
[role="tabpanel"] { animation: dn-rise .5s var(--dn-ease) both; }

/* ---- cards: metrics, bordered blocks, expanders, inputs ---- */
[data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {
  border-radius: 16px;
}
[data-testid="stMetric"] {
  background: var(--dn-panel); border: 1px solid var(--dn-line); padding: 14px 16px;
  transition: transform .3s var(--dn-ease), border-color .3s var(--dn-ease), box-shadow .3s var(--dn-ease);
}
[data-testid="stMetric"]:hover {
  transform: translate3d(0, -3px, 0); border-color: rgba(124,156,255,.5);
  box-shadow: 0 14px 34px -16px rgba(124,156,255,.6);
}
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; letter-spacing: -.02em; animation: dn-rise .7s var(--dn-ease) both; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div { white-space: normal; overflow: visible; text-overflow: clip; overflow-wrap: anywhere; line-height: 1.2; font-size: 1.7rem; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { white-space: normal; overflow: visible; text-overflow: clip; }
[data-testid="stCheckbox"] label, [data-testid="stCheckbox"] label p, [data-testid="stCheckbox"] label > div { white-space: normal; overflow: visible; text-overflow: clip; overflow-wrap: anywhere; }
[data-testid="stCheckbox"] label { align-items: flex-start; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--dn-line) !important; border-radius: 16px;
  background: var(--dn-panel); transition: border-color .3s var(--dn-ease), box-shadow .3s var(--dn-ease); }
[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: rgba(124,156,255,.35) !important; }
[data-testid="stExpander"] { border: 1px solid var(--dn-line); border-radius: 14px; background: var(--dn-panel); overflow: hidden; }
[data-testid="stExpander"] summary { transition: background-color .25s var(--dn-ease); }
[data-testid="stExpander"] summary:hover { background: var(--dn-panel-hi); }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] { border-radius: 10px !important; transition: box-shadow .25s var(--dn-ease), border-color .25s var(--dn-ease); }
[data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within > div { box-shadow: 0 0 0 3px rgba(124,156,255,.28); }
[data-testid="stAlert"] { border-radius: 14px; animation: dn-rise .5s var(--dn-ease) both; }
[data-testid="stPlotlyChart"] { border-radius: 16px; border: 1px solid var(--dn-line); background: var(--dn-panel); padding: 6px; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--dn-line); }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,.12); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,156,255,.45); }

/* ---- the animated dashboard ---- */
.dn-board { display: grid; grid-template-columns: repeat(auto-fit, minmax(138px, 1fr)); gap: 14px; margin: 4px 0 18px; }
.dn-card {
  position: relative; padding: 16px 18px 14px; border-radius: 18px; overflow: hidden;
  background: linear-gradient(160deg, var(--dn-panel-hi), var(--dn-panel));
  border: 1px solid var(--dn-line); min-height: 118px;
  animation: dn-rise .7s var(--dn-ease) both; animation-delay: calc(var(--i, 0) * 70ms);
  transition: transform .35s var(--dn-ease), border-color .35s var(--dn-ease), box-shadow .35s var(--dn-ease);
}
.dn-card:hover { transform: translate3d(0, -4px, 0); border-color: rgba(124,156,255,.55);
  box-shadow: 0 18px 40px -18px rgba(124,156,255,.7); }
.dn-card::before { content: ""; position: absolute; inset: -40% -20% auto auto; width: 70%; aspect-ratio: 1;
  background: radial-gradient(closest-side, rgba(124,156,255,.18), transparent 70%); pointer-events: none; }
.dn-label { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; font-size: .7rem; text-transform: uppercase; letter-spacing: .06em; color: var(--dn-dim); }
.dn-value { font-size: 2.1rem; font-weight: 700; letter-spacing: -.03em; line-height: 1.15; margin-top: 6px;
  font-variant-numeric: tabular-nums; color: #fff; }
.dn-hint { font-size: .8rem; color: var(--dn-dim); margin-top: 2px; }
.dn-count { animation: dn-count 1.4s var(--dn-ease) both; animation-delay: calc(var(--i, 0) * 70ms + 120ms);
  counter-reset: a var(--a) b var(--b); }
.dn-count::before { content: counter(a); }
.dn-count.dn-dec::after { content: "." counter(b, decimal-leading-zero); }
@keyframes dn-count { from { --a: 0; --b: 0; } to { --a: var(--ta); --b: var(--tb, 0); } }
.dn-ring { position: absolute; right: 10px; bottom: 10px; width: 34px; height: 34px; transform: rotate(-90deg); }
.dn-card:has(.dn-ring) .dn-value { font-size: 1.7rem; padding-right: 36px; }
.dn-ring circle { fill: none; stroke-width: 3.4; stroke-linecap: round; }
.dn-ring .t { stroke: rgba(255,255,255,.08); }
.dn-ring .v { stroke-dasharray: 100; stroke-dashoffset: 100; animation: dn-draw 1.6s var(--dn-ease) forwards;
  animation-delay: calc(var(--i, 0) * 70ms + 200ms); }
.dn-spark { position: absolute; left: 0; right: 0; bottom: 0; width: 100%; height: 56%; }
.dn-spark .l { fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;
  stroke-dasharray: 100; stroke-dashoffset: 100; animation: dn-draw 2s var(--dn-ease) forwards; animation-delay: .25s; }
.dn-spark .a { opacity: 0; animation: dn-fade 1.2s var(--dn-ease) .9s forwards; }
@keyframes dn-draw { to { stroke-dashoffset: var(--off, 0); } }
@keyframes dn-fade { to { opacity: 1; } }
.dn-pulse { position: relative; display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: #4ade80; margin-right: 7px; }
.dn-pulse::after { content: ""; position: absolute; inset: 0; border-radius: 50%; background: #4ade80;
  animation: dn-ping 2.4s ease-out infinite; will-change: transform, opacity; }
@keyframes dn-ping { from { opacity: .55; transform: scale(1); } to { opacity: 0; transform: scale(3.4); } }
.dn-sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .001ms !important; animation-delay: 0ms !important;
    animation-iteration-count: 1 !important; transition-duration: .001ms !important; scroll-behavior: auto !important; }
}
"""


def style_tag() -> str:
    """The whole stylesheet, ready to place on the page."""
    return f"<style>{CSS}</style>"


# ------------------------------------------------------------------------------- dashboard


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def count_card(label: str, value: float, hint: str = "", index: int = 0) -> str:
    """A number that counts up from zero when the card appears (whole numbers or 2 decimals)."""
    whole = float(value).is_integer()
    if whole:
        a, b = int(value), 0
    else:
        scaled = int(round(abs(value) * 100))
        a, b = scaled // 100, scaled % 100
        a = -a if value < 0 else a
    dec = "" if whole else " dn-dec"
    shown = f"{int(value):,}" if whole else f"{value:,.2f}"
    return (
        f'<div class="dn-card" style="--i:{index}">'
        f'<div class="dn-label">{_esc(label)}</div>'
        f'<div class="dn-value"><span class="dn-count{dec}" style="--ta:{a};--tb:{b}" '
        f'aria-hidden="true"></span><span class="dn-sr">{_esc(shown)}</span></div>'
        f'<div class="dn-hint">{_esc(hint)}</div></div>'
    )


def ring_card(
    label: str, percent: float, hint: str = "", index: int = 0, good_low: bool = True
) -> str:
    """A percentage as a ring that draws itself, with the number counting up beside it."""
    pct = float(min(100.0, max(0.0, percent)))
    colour = GOOD if (pct < 5) == good_low else WARN
    ring = (
        '<svg class="dn-ring" viewBox="0 0 36 36" aria-hidden="true">'
        '<circle class="t" cx="18" cy="18" r="15.9155" pathLength="100"/>'
        f'<circle class="v" cx="18" cy="18" r="15.9155" pathLength="100" '
        f'style="stroke:{colour};--off:{100 - pct:.2f}"/></svg>'
    )
    scaled = int(round(pct * 100))
    return (
        f'<div class="dn-card" style="--i:{index}">{ring}'
        f'<div class="dn-label">{_esc(label)}</div>'
        f'<div class="dn-value"><span class="dn-count dn-dec" style="--ta:{scaled // 100};'
        f'--tb:{scaled % 100}" aria-hidden="true"></span>%'
        f'<span class="dn-sr">{pct:.2f} percent</span></div>'
        f'<div class="dn-hint">{_esc(hint)}</div></div>'
    )


def sparkline_points(
    values: list[float], width: float = 120.0, height: float = 40.0
) -> list[tuple[float, float]]:
    """Scale values to a small drawing area (flat data draws a centre line)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return []
    lo, hi = float(arr.min()), float(arr.max())
    span = (hi - lo) or 1.0
    xs = np.linspace(0, width, len(arr))
    ys = (
        height - 4 - (arr - lo) / span * (height - 8) if hi != lo else np.full(len(arr), height / 2)
    )
    return [(float(x), float(y)) for x, y in zip(xs, ys, strict=True)]


def spark_card(label: str, values: list[float], hint: str = "", index: int = 0) -> str:
    """A card whose background line draws itself from left to right."""
    pts = sparkline_points(values)
    if not pts:
        return (
            f'<div class="dn-card" style="--i:{index}"><div class="dn-label">{_esc(label)}</div>'
            f'<div class="dn-hint">{_esc(hint)}</div></div>'
        )
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,40 {line} 120,40"
    return (
        f'<div class="dn-card" style="--i:{index}">'
        f'<svg class="dn-spark" viewBox="0 0 120 40" preserveAspectRatio="none" aria-hidden="true">'
        '<defs><linearGradient id="dnfill" x1="0" x2="0" y1="0" y2="1">'
        '<stop offset="0" stop-color="#7c9cff" stop-opacity=".35"/>'
        '<stop offset="1" stop-color="#7c9cff" stop-opacity="0"/></linearGradient>'
        '<linearGradient id="dnline" x1="0" x2="1"><stop offset="0" stop-color="#7c9cff"/>'
        '<stop offset="1" stop-color="#b28cff"/></linearGradient></defs>'
        f'<polygon class="a" points="{area}" fill="url(#dnfill)"/>'
        f'<polyline class="l" pathLength="100" points="{line}" stroke="url(#dnline)"/></svg>'
        f'<div class="dn-label"><span class="dn-pulse"></span>{_esc(label)}</div>'
        f'<div class="dn-hint">{_esc(hint)}</div></div>'
    )


def board(cards: list[str]) -> str:
    """The row of dashboard cards."""
    return f'<div class="dn-board">{"".join(cards)}</div>'
