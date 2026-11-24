"""Cleaning of untrusted text (column names, cell values, file text) before it reaches a model.

Strings that originate from a file are length-limited, stripped of control characters and have
the data-block delimiter characters neutralized, so they cannot close a block or fake a new one.
"""

from __future__ import annotations

import unicodedata

OPEN = "<<DATA:"
CLOSE = "<<END:"


def sanitize_text(value: object, max_len: int) -> str:
    """One line of safe text, at most `max_len` characters long."""
    text = str(value)
    cleaned = "".join(
        " " if ch in "\r\n\t" else ch
        for ch in text
        if ch in "\r\n\t" or unicodedata.category(ch)[0] != "C"
    )
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.replace("<<", "‹‹").replace(">>", "››")
    cleaned = cleaned.replace("```", "'''")
    if len(cleaned) > max_len:
        cleaned = cleaned[: max(0, max_len - 1)].rstrip() + "…"
    return cleaned


def block(name: str, body: str) -> str:
    """A delimited data block. Anything inside is data, never instructions."""
    return f"{OPEN}{name}>>\n{body}\n{CLOSE}{name}>>"
