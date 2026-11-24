"""Write tests/ai/eval_questions.json: evaluation questions with ground truth from plain pandas.

Each in-scope question has a `reference_query` (what the analyst should ask for) and `expect`, the
table computed here directly with pandas, not with the analyst's executor. Out-of-scope questions
expect a refusal. Nothing is downloaded.

Usage:  python scripts/make_eval_questions.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from datanaut.ai.evaluate import default_questions_path, load_eval_frame

P, V, I_, T = "global_active_power", "voltage", "global_intensity", "outdoor_temp"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

Truth = Callable[[pd.DataFrame], tuple[list[str], list[list[Any]]]]


def scalar(col: str, fn: Callable[[pd.Series], float], mask: Callable | None = None) -> Truth:
    def truth(df: pd.DataFrame):
        s = df[col] if mask is None else df[mask(df)][col]
        return ["all", col], [["all rows", float(fn(s))]]

    return truth


def grouped(col: str, key, label, fn: str, name: str, **kw: Any) -> Truth:
    def truth(df: pd.DataFrame):
        g = df[col].groupby(key(df))
        out = getattr(g, fn)(**kw)
        return [name, col], [[label(k), float(v)] for k, v in out.items()]

    return truth


def hour(k: int) -> str:
    return f"{k:02d}:00"


def day(k: pd.Timestamp) -> str:
    return str(pd.Timestamp(k).date())


def top(col: str, fn: str, n: int, ascending: bool) -> Truth:
    def truth(df: pd.DataFrame):
        out = getattr(df[col].groupby(df.index.normalize()), fn)()
        out = out.sort_values(ascending=ascending).head(n)
        return ["date", col], [[day(k), float(v)] for k, v in out.items()]

    return truth


def compare(col: str) -> Truth:
    def truth(df: pd.DataFrame):
        weekday = float(df[df.index.day_of_week < 5][col].mean())
        weekend = float(df[df.index.day_of_week >= 5][col].mean())
        return ["group", col], [["weekday", weekday], ["weekend", weekend]]

    return truth


def question(qid, text, category, query, truth: Truth, df) -> dict[str, Any]:
    columns, rows = truth(df)
    return {
        "id": qid,
        "question": text,
        "category": category,
        "reference_query": query,
        "expect": {"type": "table", "columns": columns, "rows": rows},
    }


def refusal(qid: str, text: str) -> dict[str, Any]:
    return {"id": qid, "question": text, "category": "out_of_scope", "expect": {"type": "refusal"}}


def build(df: pd.DataFrame) -> list[dict[str, Any]]:
    wk_range = {"start": "2024-01-01", "end": "2024-01-31"}
    cold = {"column": T, "operator": "<", "value": "5"}
    q = question
    return [
        q(
            "mean_power",
            "What is the average power?",
            "aggregation",
            {"value_columns": [P], "aggregation": "mean"},
            scalar(P, lambda s: s.mean()),
            df,
        ),
        q(
            "max_power",
            "What was the highest power reading?",
            "aggregation",
            {"value_columns": [P], "aggregation": "max"},
            scalar(P, lambda s: s.max()),
            df,
        ),
        q(
            "min_voltage",
            "What is the lowest voltage?",
            "aggregation",
            {"value_columns": [V], "aggregation": "min"},
            scalar(V, lambda s: s.min()),
            df,
        ),
        q(
            "median_temp",
            "What is the median outdoor temperature?",
            "aggregation",
            {"value_columns": [T], "aggregation": "median"},
            scalar(T, lambda s: s.median()),
            df,
        ),
        q(
            "std_intensity",
            "How variable is the intensity (standard deviation)?",
            "aggregation",
            {"value_columns": [I_], "aggregation": "std"},
            scalar(I_, lambda s: s.std()),
            df,
        ),
        q(
            "sum_power",
            "What is the total power over the whole period?",
            "aggregation",
            {"value_columns": [P], "aggregation": "sum"},
            scalar(P, lambda s: s.sum()),
            df,
        ),
        q(
            "count_voltage",
            "How many voltage readings are there?",
            "count",
            {"value_columns": [V], "aggregation": "count"},
            scalar(V, lambda s: s.count()),
            df,
        ),
        q(
            "p90_power",
            "What is the 90th percentile of power?",
            "aggregation",
            {"value_columns": [P], "aggregation": "quantile", "q": 0.9},
            scalar(P, lambda s: s.quantile(0.9)),
            df,
        ),
        q(
            "power_by_hour",
            "What is the average power for each hour of the day?",
            "group_hour",
            {"value_columns": [P], "group_by": "hour", "aggregation": "mean", "limit": 100},
            grouped(P, lambda d: d.index.hour, hour, "mean", "hour"),
            df,
        ),
        q(
            "max_power_by_hour",
            "What is the highest power in each hour of the day?",
            "group_hour",
            {"value_columns": [P], "group_by": "hour", "aggregation": "max", "limit": 100},
            grouped(P, lambda d: d.index.hour, hour, "max", "hour"),
            df,
        ),
        q(
            "voltage_by_hour",
            "Show the average voltage by hour of day.",
            "group_hour",
            {"value_columns": [V], "group_by": "hour", "aggregation": "mean", "limit": 100},
            grouped(V, lambda d: d.index.hour, hour, "mean", "hour"),
            df,
        ),
        q(
            "power_by_weekday",
            "What is the average power on each weekday?",
            "group_weekday",
            {"value_columns": [P], "group_by": "weekday", "aggregation": "mean"},
            grouped(P, lambda d: d.index.day_of_week, lambda k: DAYS[k], "mean", "weekday"),
            df,
        ),
        q(
            "sum_power_by_weekday",
            "What is the total power for each day of the week?",
            "group_weekday",
            {"value_columns": [P], "group_by": "weekday", "aggregation": "sum"},
            grouped(P, lambda d: d.index.day_of_week, lambda k: DAYS[k], "sum", "weekday"),
            df,
        ),
        q(
            "temp_by_weekday",
            "What is the average outdoor temperature by weekday?",
            "group_weekday",
            {"value_columns": [T], "group_by": "weekday", "aggregation": "mean"},
            grouped(T, lambda d: d.index.day_of_week, lambda k: DAYS[k], "mean", "weekday"),
            df,
        ),
        q(
            "power_by_month",
            "What is the average power in each month?",
            "group_month",
            {"value_columns": [P], "group_by": "month", "aggregation": "mean"},
            grouped(P, lambda d: d.index.to_period("M").astype(str), str, "mean", "month"),
            df,
        ),
        q(
            "power_when_cold",
            "What is the average power when it is below 5 degrees outside?",
            "filter",
            {"value_columns": [P], "aggregation": "mean", "filters": [cold]},
            scalar(P, lambda s: s.mean(), lambda d: d[T] < 5),
            df,
        ),
        q(
            "count_high_power",
            "How many readings have power above 2?",
            "filter",
            {
                "value_columns": [P],
                "aggregation": "count",
                "filters": [{"column": P, "operator": ">", "value": "2"}],
            },
            scalar(P, lambda s: s.count(), lambda d: d[P] > 2),
            df,
        ),
        q(
            "evening_voltage",
            "What is the average voltage between 18:00 and 20:00?",
            "filter",
            {
                "value_columns": [V],
                "aggregation": "mean",
                "filters": [{"column": "hour", "operator": "between", "values": ["18", "20"]}],
            },
            scalar(V, lambda s: s.mean(), lambda d: (d.index.hour >= 18) & (d.index.hour <= 20)),
            df,
        ),
        q(
            "weekday_weekend_power",
            "Is power higher on weekdays or weekends? Compare the averages.",
            "compare",
            {
                "value_columns": [P],
                "aggregation": "mean",
                "compare": {
                    "label_a": "weekday",
                    "filters_a": [
                        {"column": "weekday", "operator": "between", "values": ["0", "4"]}
                    ],
                    "label_b": "weekend",
                    "filters_b": [{"column": "weekday", "operator": "in", "values": ["5", "6"]}],
                },
            },
            compare(P),
            df,
        ),
        q(
            "weekday_weekend_voltage",
            "Compare the average voltage on weekdays and weekends.",
            "compare",
            {
                "value_columns": [V],
                "aggregation": "mean",
                "compare": {
                    "label_a": "weekday",
                    "filters_a": [
                        {"column": "weekday", "operator": "between", "values": ["0", "4"]}
                    ],
                    "label_b": "weekend",
                    "filters_b": [{"column": "weekday", "operator": "in", "values": ["5", "6"]}],
                },
            },
            compare(V),
            df,
        ),
        q(
            "top3_days_power",
            "Which 3 days had the highest average power?",
            "top_n",
            {
                "value_columns": [P],
                "group_by": "date",
                "aggregation": "mean",
                "sort": "desc",
                "limit": 3,
            },
            top(P, "mean", 3, False),
            df,
        ),
        q(
            "top5_peak_days",
            "Which 5 days had the highest peak power reading?",
            "top_n",
            {
                "value_columns": [P],
                "group_by": "date",
                "aggregation": "max",
                "sort": "desc",
                "limit": 5,
            },
            top(P, "max", 5, False),
            df,
        ),
        q(
            "coldest_days",
            "Which 3 days were the coldest on average?",
            "top_n",
            {
                "value_columns": [T],
                "group_by": "date",
                "aggregation": "mean",
                "sort": "asc",
                "limit": 3,
            },
            top(T, "mean", 3, True),
            df,
        ),
        q(
            "january_power",
            "What was the average power in January 2024?",
            "time_scope",
            {"value_columns": [P], "aggregation": "mean", "time_scope": wk_range},
            scalar(
                P,
                lambda s: s.mean(),
                lambda d: (d.index >= "2024-01-01") & (d.index < "2024-02-01"),
            ),
            df,
        ),
        q(
            "march_max_temp",
            "What was the maximum temperature between 1 and 15 March 2024?",
            "time_scope",
            {
                "value_columns": [T],
                "aggregation": "max",
                "time_scope": {"start": "2024-03-01", "end": "2024-03-15"},
            },
            scalar(
                T, lambda s: s.max(), lambda d: (d.index >= "2024-03-01") & (d.index < "2024-03-16")
            ),
            df,
        ),
        q(
            "count_evening",
            "How many readings were taken between 18:00 and 22:00?",
            "count",
            {
                "value_columns": [P],
                "aggregation": "count",
                "filters": [{"column": "hour", "operator": "between", "values": ["18", "22"]}],
            },
            scalar(P, lambda s: s.count(), lambda d: (d.index.hour >= 18) & (d.index.hour <= 22)),
            df,
        ),
        refusal("refuse_code", "Write me a Python script that cleans this file."),
        refusal("refuse_internet", "Look up tomorrow's weather in Berlin on the internet."),
        refusal("refuse_chat", "Tell me a joke about cats."),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=default_questions_path())
    args = parser.parse_args()
    items = build(load_eval_frame())
    args.out.write_text(json.dumps(items, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} questions to {args.out}")


if __name__ == "__main__":
    main()
