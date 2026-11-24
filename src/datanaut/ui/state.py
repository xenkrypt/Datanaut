"""Wizard state: step gating, dependent invalidation, a bounded result cache and messages.

No streamlit import. One WizardState belongs to one browser session; nothing here is global.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from datanaut.ai.llm import AISession
from datanaut.ai.schemas import AnalystAnswer
from datanaut.ai.suggest import SuggestionOutcome, SummaryOutcome
from datanaut.core.loader import LoadResult
from datanaut.core.log import PipelineLog
from datanaut.core.profiler import DataProfile
from datanaut.core.timeindex import TimeIndexResult
from datanaut.core.validation import AnyRule, ValidationReport
from datanaut.core.versioning import VersionStore

STEPS: tuple[str, ...] = (
    "Upload",
    "Profile",
    "Map columns",
    "Clean",
    "Analyze",
    "Insights",
    "Ask",
    "Report",
)

_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "Upload": (),
    "Profile": ("Upload",),
    "Map columns": ("Upload",),
    "Clean": ("Map columns",),
    "Analyze": ("Map columns",),
    "Insights": ("Map columns",),
    "Ask": ("Map columns",),
    "Report": ("Map columns",),
}
_MAPPING_DEPENDENTS = ("Map columns", "Clean", "Analyze", "Insights", "Ask", "Report")
CACHE_SIZE = 24

Level = Literal["info", "success", "warning", "error"]


@dataclass(frozen=True)
class Message:
    """A message for the user, shown once."""

    level: Level
    text: str


def cache_key(fingerprint: str, name: str, **params: Any) -> tuple:
    """A hashable key: the data fingerprint, the function name and its parameters."""
    return (fingerprint, name, tuple(sorted((k, repr(v)) for k, v in params.items())))


class ResultCache:
    """A small least-recently-used cache, so results are not recomputed on every rerun."""

    def __init__(self, size: int = CACHE_SIZE) -> None:
        self._size = max(1, size)
        self._items: OrderedDict[tuple, Any] = OrderedDict()

    def get_or_compute(self, key: tuple, fn: Callable[[], Any]) -> Any:
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        value = fn()
        self._items[key] = value
        while len(self._items) > self._size:
            self._items.popitem(last=False)
        return value

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: tuple) -> bool:
        return key in self._items


@dataclass
class WizardState:
    """Everything the wizard remembers for one session."""

    step: str = "Upload"
    completed: set[str] = field(default_factory=set)
    loaded: LoadResult | None = None
    source_bytes: bytes | None = None
    source_digest: str = ""
    sheet: str | None = None
    profile: DataProfile | None = None
    profile_logged: bool = False
    indexed: TimeIndexResult | None = None
    time_columns: list[str] = field(default_factory=list)
    dayfirst: bool | None = None
    value_columns: list[str] = field(default_factory=list)
    consumption: dict[str, bool] = field(default_factory=dict)
    store: VersionStore | None = None
    log: PipelineLog = field(default_factory=PipelineLog)
    cache: ResultCache = field(default_factory=ResultCache)
    messages: list[Message] = field(default_factory=list)
    rules: list[AnyRule] = field(default_factory=list)  # the user's validation rules
    suggestions: list[AnyRule] | None = None  # suggested rules not yet accepted or deleted
    report: ValidationReport | None = None
    report_fingerprint: str = ""  # the data the report was computed on
    analysis_column: str | None = None
    weather: pd.DataFrame | None = None  # time-indexed weather the user supplied or fetched
    weather_id: str = ""  # identifies the weather in cache keys
    ai_enabled: bool = False  # the AI analyst is opt-in for each session
    ai_sample_rows: bool = False  # a second opt-in: send a few sample rows
    ai_mode: str = "beginner"
    ai_session: AISession = field(default_factory=AISession)
    ai_history: list[tuple[str, AnalystAnswer]] = field(default_factory=list)
    ai_suggestions: SuggestionOutcome | None = None
    ai_summaries: dict[str, SummaryOutcome] = field(default_factory=dict)
    latest: dict[tuple[str, str], Any] = field(default_factory=dict)
    """The most recent analysis result of each kind, by (dataset fingerprint, name)."""
    ai_suggestions_fingerprint: str = ""  # the data the AI suggestions were made for
    report_pdf: tuple[str, bytes] | None = None  # the last PDF, in memory only
    base_log_length: int = 0  # log entries that exist before any time index was built

    # --- steps ---------------------------------------------------------------------------

    def is_reachable(self, step: str) -> bool:
        """A step is reachable when its prerequisites are done."""
        return all(p in self.completed for p in _PREREQUISITES[step])

    def go(self, step: str) -> bool:
        """Move to a step. Going back is always allowed; going forward needs the prerequisites."""
        if step not in STEPS:
            return False
        going_back = STEPS.index(step) < STEPS.index(self.step)
        if not going_back and not self.is_reachable(step):
            return False
        self.step = step
        return True

    def mark_done(self, step: str) -> None:
        self.completed.add(step)

    def status(self, step: str) -> Literal["current", "done", "available", "locked"]:
        if step == self.step:
            return "current"
        if step in self.completed:
            return "done"
        return "available" if self.is_reachable(step) else "locked"

    # --- messages ------------------------------------------------------------------------

    def add_message(self, level: Level, text: str) -> None:
        self.messages.append(Message(level, text))

    def take_messages(self) -> list[Message]:
        taken, self.messages = self.messages, []
        return taken

    # --- data and invalidation -----------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        """Identifies the data the results depend on: the head version once there is one."""
        if self.store is not None:
            return self.store.version(self.store.head).fingerprint
        return self.source_digest

    def get_or_compute(self, key: tuple, fn: Callable[[], Any]) -> Any:
        return self.cache.get_or_compute(key, fn)

    def set_loaded(
        self, loaded: LoadResult, data: bytes, digest: str, sheet: str | None = None
    ) -> None:
        """A new file (or sheet): everything that depended on the old one is discarded."""
        had_work = self.store is not None
        self.completed = {"Upload"}
        self.loaded, self.source_bytes, self.source_digest, self.sheet = loaded, data, digest, sheet
        self.profile, self.profile_logged = None, False
        self.rules = []
        self.weather, self.weather_id, self.analysis_column = None, "", None
        self._drop_ai()
        self._drop_mapping()
        self.log = PipelineLog()
        self.log.extend(loaded.log)
        self.base_log_length = len(self.log)
        self.cache.clear()
        self.step = "Upload"
        if had_work:
            self.add_message("warning", "A new file was loaded, so the earlier work was discarded.")

    def _drop_ai(self) -> None:
        """Forget AI answers about the old data (the question count stays with the session)."""
        self.ai_history, self.ai_suggestions, self.ai_summaries = [], None, {}
        self.ai_session.cache = type(self.ai_session.cache)()

    def _drop_mapping(self) -> None:
        self.indexed, self.store = None, None
        self.time_columns, self.dayfirst = [], None
        self.value_columns, self.consumption = [], {}
        self.suggestions, self.report, self.report_fingerprint = None, None, ""
        self.latest, self.report_pdf = {}, None
        self.completed -= set(_MAPPING_DEPENDENTS)

    def invalidate_mapping(self, reason: str) -> None:
        """An earlier choice changed: drop the time index, versions, log entries and results
        that depend on it, and keep the file and its profile."""
        had_versions = self.store is not None and len(self.store.history()) > 1
        self._drop_mapping()
        self._drop_ai()
        self.log = PipelineLog(self.log.entries[: self.base_log_length])
        self.cache.clear()
        if self.step in _MAPPING_DEPENDENTS[1:]:
            self.step = "Map columns"
        extra = " Your cleaning steps were discarded." if had_versions else ""
        self.add_message("warning", f"{reason}{extra}")
