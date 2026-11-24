"""What the model is told about the data, built and cleaned in one place.

The model receives column names and types, summary statistics, a summary of the pipeline log,
the ranked findings and (only if the user opts in) a few sample rows. It never receives the
dataset. Everything that came from the file is cleaned by `text.sanitize_text` and placed in a
delimited data block. `ContextPayload.to_prompt()` is exactly what is sent, so the UI can show it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from datanaut.ai import config
from datanaut.ai.executor import DataSchema, schema_from_frame
from datanaut.ai.schemas import FIXED_GROUPS, VIRTUAL_COLUMNS
from datanaut.ai.text import block, sanitize_text
from datanaut.core.log import PipelineLog

TOOLS_DESCRIPTION = (
    "You can only call two tools. query_data asks for a computed table: choose value_columns, "
    "optional filters (operators ==, !=, <, <=, >, >=, between, in), an optional time_scope, "
    "group_by (one of "
    + ", ".join(FIXED_GROUPS)
    + " or a categorical column), an aggregation (mean, sum, min, max, count, median, std, "
    "quantile) and a limit. Calendar fields hour, weekday (Monday is 0) and month can be used in "
    "filters. To compare two groups use compare. refuse declines a request that is not a "
    "question about this dataset."
)


def _round(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(f"{value:.4g}")


def _column_summary(df: pd.DataFrame, schema: DataSchema) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for info in schema.columns.values():
        s = df[info.real]
        entry: dict[str, Any] = {
            "name": info.name,
            "type": info.kind,
            "missing_pct": _round(100 * float(s.isna().mean())),
        }
        if info.kind == "numeric":
            present = s.dropna().astype(float)
            entry.update(
                count=int(len(present)),
                min=_round(present.min()) if len(present) else None,
                mean=_round(present.mean()) if len(present) else None,
                max=_round(present.max()) if len(present) else None,
                std=_round(present.std()) if len(present) > 1 else None,
            )
        else:
            counts = s.dropna().astype(str).value_counts()
            entry["unique"] = int(len(counts))
            if len(counts) <= config.MAX_CATEGORY_VALUES:
                entry["values"] = [sanitize_text(v, config.MAX_STRING) for v in counts.index]
        out.append(entry)
    return out


def _log_summary(log: PipelineLog | None) -> list[dict[str, Any]]:
    if log is None:
        return []
    entries = log.entries[-config.MAX_LOG_ENTRIES :]
    return [
        {
            "step": e.step_id,
            "operation": sanitize_text(e.operation, 40),
            "counts": {sanitize_text(k, 40): _round(float(v)) for k, v in e.counts.items()},
        }
        for e in entries
    ]


def _finding_summary(findings: Iterable[Any]) -> list[dict[str, str]]:
    out = []
    for f in findings:
        get = f.get if isinstance(f, dict) else lambda k, f=f: getattr(f, k, "")
        out.append(
            {
                "id": sanitize_text(get("id"), 60),
                "title": sanitize_text(get("title"), config.MAX_STRING),
                "statement": sanitize_text(get("statement"), config.MAX_FINDING_TEXT),
            }
        )
    return out


def _sample(df: pd.DataFrame, schema: DataSchema, n: int) -> list[dict[str, str]]:
    n = max(0, min(int(n), config.MAX_SAMPLE_ROWS))
    if n == 0 or df.empty:
        return []
    positions = np.unique(np.linspace(0, len(df) - 1, min(n, len(df))).astype(int))
    rows = []
    for pos in positions:
        row = {"time": str(df.index[pos])}
        for info in schema.columns.values():
            value = df.iloc[pos][info.real]
            row[info.name] = "" if pd.isna(value) else sanitize_text(value, config.MAX_STRING)
        rows.append(row)
    return rows


@dataclass(frozen=True)
class ContextPayload:
    """Exactly what will be sent as background about the data."""

    columns: list[dict[str, Any]]
    time_range: dict[str, str]
    n_rows: int
    pipeline: list[dict[str, Any]]
    findings: list[dict[str, str]]
    sample_rows: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "time_range": self.time_range,
            "n_rows": self.n_rows,
            "pipeline": self.pipeline,
            "findings": self.findings,
            "sample_rows": self.sample_rows,
        }

    def to_prompt(self) -> str:
        """The text sent to the model: instructions-free background, in data blocks."""

        def dump(obj: Any) -> str:
            return json.dumps(obj, ensure_ascii=False, indent=1)

        parts = [
            block(
                "schema",
                dump(
                    {"columns": self.columns, "n_rows": self.n_rows, "time_range": self.time_range}
                ),
            ),
            block("pipeline_log", dump(self.pipeline)),
            block("findings", dump(self.findings)),
        ]
        if self.sample_rows:
            parts.append(block("sample_rows", dump(self.sample_rows)))
        parts.append("Tools (instructions from the application):\n" + TOOLS_DESCRIPTION)
        return "\n\n".join(parts)

    def digest(self) -> str:
        return hashlib.sha256(self.to_prompt().encode("utf-8")).hexdigest()[:16]

    @property
    def virtual_columns(self) -> tuple[str, ...]:
        return VIRTUAL_COLUMNS


def build_context(
    df: pd.DataFrame,
    *,
    findings: Iterable[Any] = (),
    log: PipelineLog | None = None,
    sample_rows: int = 0,
    schema: DataSchema | None = None,
) -> ContextPayload:
    """Build the background for the model. `sample_rows` is 0 unless the user opted in."""
    schema = schema or schema_from_frame(df)
    return ContextPayload(
        columns=_column_summary(df, schema),
        time_range={"start": str(schema.start), "end": str(schema.end)},
        n_rows=schema.n_rows,
        pipeline=_log_summary(log),
        findings=_finding_summary(findings),
        sample_rows=_sample(df, schema, sample_rows),
    )
