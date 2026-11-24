"""A check at start-up that optional pieces are in place, with a clear note for each that is not.

Nothing here stops the app. A missing piece turns off only the feature that needs it, and the
screen for that feature explains why. Keys are only looked for, never read into a message.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from datanaut.ai import config as ai_config


@dataclass(frozen=True)
class Notice:
    level: Literal["info", "warning"]
    feature: str
    message: str


def _fonts_present() -> bool:
    try:
        from datanaut.report.text import font_dir

        base = font_dir()
        return all(
            os.path.isfile(os.path.join(base, name))
            for name in (
                "DejaVuSans.ttf",
                "DejaVuSans-Bold.ttf",
                "DejaVuSans-Oblique.ttf",
                "DejaVuSansMono.ttf",
            )
        )
    except Exception:  # an unusable matplotlib install is the same as missing fonts
        return False


def report_available() -> bool:
    """Can the PDF report be built (its fonts are found)?"""
    return _fonts_present()


def check_environment(
    env: Mapping[str, str] | None = None, secrets: Mapping[str, str] | None = None
) -> list[Notice]:
    """What is missing, as notices. An empty list means everything is in place."""
    env = os.environ if env is None else env
    notices: list[Notice] = []
    if not report_available():
        notices.append(
            Notice(
                "warning",
                "Report",
                "The fonts needed for the PDF report were not found (they ship with matplotlib), "
                "so the report is turned off. The rest of the app works.",
            )
        )
    has_key = any(
        (secrets or {}).get(name) or env.get(name) for name in ai_config.KEY_NAMES.values()
    )
    if not has_key:
        notices.append(
            Notice(
                "info",
                "AI analyst",
                "No AI key is set, so the AI analyst is off. Add GEMINI_API_KEY or GROQ_API_KEY "
                "to .streamlit/secrets.toml or the environment to turn it on.",
            )
        )
    return notices
