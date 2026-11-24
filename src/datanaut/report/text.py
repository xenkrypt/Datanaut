"""Text the report fonts can draw: DejaVu Sans, taken from matplotlib's own package data."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import matplotlib
from fontTools.ttLib import TTFont


def font_dir() -> str:
    """Where matplotlib keeps DejaVu Sans (no font file is bundled with Datanaut)."""
    return os.path.join(matplotlib.get_data_path(), "fonts", "ttf")


@lru_cache(maxsize=1)
def glyphs() -> frozenset[int]:
    """The characters DejaVu Sans can draw."""
    cmap = TTFont(os.path.join(font_dir(), "DejaVuSans.ttf"))["cmap"].getBestCmap()
    return frozenset(cmap)


def printable(text: Any) -> str:
    """Text the font can draw: other characters become '?' and control characters are dropped."""
    known = glyphs()
    out = []
    for ch in str(text):
        code = ord(ch)
        if ch == "\n":
            out.append(ch)
        elif ch in "\t\r":
            out.append(" ")
        elif code >= 32 and code in known:
            out.append(ch)
        elif code >= 32 and code != 127:
            out.append("?")
    return "".join(out)
