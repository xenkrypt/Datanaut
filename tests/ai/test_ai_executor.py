import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datanaut.ai.executor import (
    AGGREGATIONS,
    execute_query,
    observed_range,
    parse_query,
    schema_from_frame,
    validate_query,
)
from datanaut.ai.schemas import Compare, Filter, QuerySpec, TimeScope
from datanaut.errors import QueryError


@pytest.fixture(scope="module")
def schema(frame):
    return schema_from_frame(frame)


def q(**kw) -> QuerySpec:
    kw.setdefault("value_columns", ["kw"])
    return QuerySpec(**kw)


def rejected(spec: QuerySpec, schema, text: str) -> None:
    with pytest.raises(QueryError, match=text):
        validate_query(spec, schema)


# --- the validator ------------------------------------------------------------------------


def test_a_good_query_passes(schema):
    validate_query(
        q(
            filters=[Filter(column="temp", operator="between", values=["0", "10"])],
            time_scope=TimeScope(start="2024-01-03", end="2024-01-10"),
            group_by="hour",
            aggregation="quantile",
            q=0.9,
            sort="desc",
            limit=10,
        ),
        schema,
    )


def test_every_disallowed_case_is_rejected_with_a_readable_message(schema):
    rejected(
        q(value_columns=["nope"]), schema, "unknown column 'nope'.*Known columns: kw, temp, site"
    )
    rejected(q(value_columns=["site"]), schema, "not numeric")
    rejected(
        q(filters=[Filter(column="ghost", operator="==", value="1")]), schema, "unknown column"
    )
    rejected(
        q(filters=[Filter(column="kw", operator="==", value="abc")]), schema, "must be numbers"
    )
    rejected(
        q(filters=[Filter(column="kw", operator="between", values=["9", "1"])]),
        schema,
        "lower value first",
    )
    rejected(
        q(filters=[Filter(column="kw", operator="between", values=["1"])]), schema, "exactly two"
    )
    rejected(q(filters=[Filter(column="kw", operator="in", value="1")]), schema, "needs 'values'")
    rejected(q(filters=[Filter(column="kw", operator=">", values=["1"])]), schema, "single 'value'")
    rejected(
        q(filters=[Filter(column="site", operator="<", value="a")]), schema, "only works on numeric"
    )
    rejected(
        q(filters=[Filter(column="hour", operator="==", value="24")]),
        schema,
        "whole number from 0 to 23",
    )
    rejected(q(filters=[Filter(column="hour", operator="==", value="1.5")]), schema, "whole number")
    rejected(q(filters=[Filter(column="kw", operator=">", value="1")] * 9), schema, "at most 8")
    rejected(q(limit=1001), schema, "limit: must be from 1 to 1000")
    rejected(q(limit=0), schema, "limit")
    rejected(q(time_scope=TimeScope(start="2023-12-01")), schema, "outside the data")
    rejected(q(time_scope=TimeScope(end="2024-03-01")), schema, "outside the data")
    rejected(
        q(time_scope=TimeScope(start="2024-01-10", end="2024-01-05")), schema, "start is after end"
    )
    rejected(q(time_scope=TimeScope(start="yesterday")), schema, "is not a date")
    rejected(q(group_by="nope"), schema, "group_by: 'nope' is not allowed")
    rejected(q(group_by="kw"), schema, "not allowed")  # numeric columns cannot be groups
    rejected(q(aggregation="quantile"), schema, "needs q")
    rejected(q(aggregation="mean", q=0.5), schema, "only allowed with")
    rejected(q(aggregation="quantile", q=1.5), schema, "strictly between")
    rejected(
        q(
            group_by="hour",
            compare=Compare(
                filters_a=[Filter(column="hour", operator="==", value="1")],
                filters_b=[Filter(column="hour", operator="==", value="2")],
            ),
        ),
        schema,
        "group_by must be 'none'",
    )
    rejected(q(compare=Compare(filters_a=[], filters_b=[])), schema, "needs at least one filter")
    rejected(q(value_columns=["kw"] * 9), schema, "at most 8")


def test_all_problems_are_reported_together(schema):
    with pytest.raises(QueryError) as err:
        validate_query(q(value_columns=["nope"], limit=5000, group_by="zzz"), schema)
    text = err.value.message
    assert "unknown column" in text and "limit" in text and "group_by" in text


def test_count_is_allowed_on_a_categorical_column(schema):
    validate_query(q(value_columns=["site"], aggregation="count"), schema)


@pytest.mark.parametrize(
    ("raw", "text"),
    [
        (
            {
                "value_columns": ["kw"],
                "filters": [{"column": "kw", "operator": "LIKE", "value": "1"}],
            },
            "operator: not allowed",
        ),
        ({"value_columns": ["kw"], "aggregation": "eval"}, "aggregation: not allowed"),
        ({"value_columns": ["kw"], "sql": "DROP TABLE x"}, "sql: unknown field"),
        ({"value_columns": []}, "value_columns"),
        ({"value_columns": ["kw"], "limit": "many"}, "limit"),
        ({}, "value_columns"),
    ],
)
def test_malformed_model_output_becomes_a_query_error(raw, text):
    with pytest.raises(QueryError, match=text):
        parse_query(raw)


def test_numbers_from_a_model_are_accepted_as_text():
    spec = parse_query(
        {"value_columns": ["kw"], "filters": [{"column": "kw", "operator": ">", "value": 5}]}
    )
    assert spec.filters[0].value == "5"


# --- the executor against direct pandas ---------------------------------------------------


def direct(frame, col, agg, key=None, q_=None):
    s = frame[col]
    grouped = s.groupby(key) if key is not None else s
    if agg == "quantile":
        return grouped.quantile(q_)
    return getattr(grouped, agg)()


@pytest.mark.parametrize("agg", AGGREGATIONS)
def test_every_aggregation_matches_pandas_with_no_grouping(frame, agg):
    spec = q(aggregation=agg, q=0.8 if agg == "quantile" else None, value_columns=["kw", "temp"])
    table = execute_query(frame, spec)
    assert table.columns == ["all", "kw", "temp"] and table.counts == [len(frame)]
    for i, col in enumerate(["kw", "temp"], start=1):
        expected = direct(frame, col, agg, q_=0.8)
        assert table.rows[0][i] == pytest.approx(float(expected))


@pytest.mark.parametrize("agg", AGGREGATIONS)
@pytest.mark.parametrize(
    ("group", "key"),
    [
        ("hour", lambda f: f.index.hour),
        ("weekday", lambda f: f.index.day_of_week),
        ("date", lambda f: f.index.normalize()),
        ("month", lambda f: f.index.to_period("M")),
        ("week", lambda f: f.index.normalize() - pd.to_timedelta(f.index.day_of_week, unit="D")),
        ("site", lambda f: f["site"]),
    ],
)
def test_every_group_by_matches_pandas(frame, agg, group, key):
    spec = q(group_by=group, aggregation=agg, q=0.25 if agg == "quantile" else None, limit=1000)
    table = execute_query(frame, spec)
    expected = direct(frame, "kw", agg, key=key(frame), q_=0.25)
    assert len(table.rows) == len(expected) and not table.truncated
    assert np.allclose([r[1] for r in table.rows], expected.to_numpy(dtype=float))
    sizes = frame.groupby(key(frame)).size().to_numpy()
    assert table.counts == list(sizes) and sum(table.counts) == table.n_observations == len(frame)


def test_group_labels_are_readable(frame):
    assert execute_query(frame, q(group_by="hour")).rows[19][0] == "19:00"
    assert execute_query(frame, q(group_by="weekday")).rows[0][0] == "Monday"
    assert execute_query(frame, q(group_by="month")).rows[0][0] == "2024-01"
    assert execute_query(frame, q(group_by="date", limit=2)).rows[0][0] == "2024-01-01"
    assert execute_query(frame, q(group_by="site")).rows[0][0] == "north"


def test_filters_and_the_time_scope_select_the_right_rows(frame):
    evening = [Filter(column="hour", operator="between", values=["18", "20"])]
    t = execute_query(frame, q(filters=evening, aggregation="count"))
    assert t.rows[0][1] == int(frame.index.hour.isin([18, 19, 20]).sum())
    both = q(
        filters=[
            Filter(column="site", operator="==", value="north"),
            Filter(column="kw", operator=">", value="50"),
        ],
        aggregation="count",
    )
    expected = int(((frame["site"] == "north") & (frame["kw"] > 50)).sum())
    assert execute_query(frame, both).rows[0][1] == expected
    ne = q(filters=[Filter(column="site", operator="!=", value="north")], aggregation="count")
    assert execute_query(frame, ne).rows[0][1] == int((frame["site"] == "south").sum())
    isin = q(
        filters=[Filter(column="weekday", operator="in", values=["5", "6"])], aggregation="count"
    )
    assert execute_query(frame, isin).rows[0][1] == int(frame.index.day_of_week.isin([5, 6]).sum())
    for op, fn in [
        ("<", np.less),
        ("<=", np.less_equal),
        (">=", np.greater_equal),
        ("==", np.equal),
    ]:
        spec = q(filters=[Filter(column="temp", operator=op, value="8.0")], aggregation="count")
        assert execute_query(frame, spec).rows[0][1] == int(fn(frame["temp"], 8.0).sum())
    scoped = q(time_scope=TimeScope(start="2024-01-03", end="2024-01-04"), aggregation="count")
    t = execute_query(frame, scoped)
    assert t.rows[0][1] == 48 and t.n_observations == 48  # the end day is included


def test_weekday_versus_weekend_compare(frame):
    spec = q(
        aggregation="mean",
        compare=Compare(
            label_a="weekday",
            filters_a=[Filter(column="weekday", operator="between", values=["0", "4"])],
            label_b="weekend",
            filters_b=[Filter(column="weekday", operator="in", values=["5", "6"])],
        ),
    )
    t = execute_query(frame, spec)
    assert t.columns == ["group", "kw"] and [r[0] for r in t.rows] == ["weekday", "weekend"]
    wk = frame[frame.index.day_of_week < 5]["kw"].mean()
    we = frame[frame.index.day_of_week >= 5]["kw"].mean()
    assert t.rows[0][1] == pytest.approx(wk) and t.rows[1][1] == pytest.approx(we)
    assert t.counts == [
        int((frame.index.day_of_week < 5).sum()),
        int((frame.index.day_of_week >= 5).sum()),
    ]


def test_sort_limit_and_truncation(frame):
    spec = q(group_by="date", aggregation="max", sort="desc", limit=3)
    t = execute_query(frame, spec)
    daily = frame["kw"].groupby(frame.index.normalize()).max().sort_values(ascending=False)
    assert t.truncated and len(t.rows) == 3
    assert [r[1] for r in t.rows] == pytest.approx(daily.iloc[:3].tolist())
    asc = execute_query(frame, q(group_by="date", aggregation="max", sort="asc", limit=2))
    assert [r[1] for r in asc.rows] == pytest.approx(daily.iloc[::-1].iloc[:2].tolist())
    assert not execute_query(frame, q(group_by="weekday", limit=7)).truncated


def test_no_matching_rows_is_an_error_the_model_can_act_on(frame):
    spec = q(filters=[Filter(column="kw", operator=">", value="9999")])
    with pytest.raises(QueryError, match="No rows match"):
        execute_query(frame, spec)


def test_missing_values_become_none_not_nan(frame):
    holey = frame.copy()
    holey.loc[holey.index[:48], "kw"] = np.nan
    t = execute_query(holey, q(group_by="date", limit=2))
    assert t.rows[0][1] is None and t.rows[1][1] is None


def test_the_observed_range_covers_only_the_rows_used(frame):
    spec = q(time_scope=TimeScope(start="2024-01-05", end="2024-01-06"))
    first, last = observed_range(frame, spec)
    assert str(first) == "2024-01-05 00:00:00" and str(last) == "2024-01-06 23:00:00"


def test_odd_column_names_are_cleaned_and_made_unique():
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    df = pd.DataFrame({"a\nb\x00": range(5), "a b": range(5), "x" * 200: range(5)}, index=idx)
    s = schema_from_frame(df)
    assert list(s.columns)[:2] == ["a b", "a b_2"] and max(len(n) for n in s.columns) <= 64
    t = execute_query(df, q(value_columns=["a b_2"]), s)
    assert t.rows[0][1] == 2.0
    with pytest.raises(QueryError):
        schema_from_frame(pd.DataFrame({"x": [1]}))  # no time index


# --- there is no way from a string to code ------------------------------------------------


def test_the_ai_package_has_no_string_to_code_path():
    forbidden = re.compile(
        r"(\beval\s*\(|\bexec\s*\(|(?<!re\.)\bcompile\s*\(|\.query\s*\(|\.eval\s*\(|__import__|"
        r"\bos\.system|\bsubprocess|\bpickle\b|\bimportlib\b)"
    )
    root = Path(__file__).resolve().parents[2] / "src" / "datanaut" / "ai"
    sources = list(root.glob("*.py"))
    assert len(sources) >= 8
    for path in sources:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            assert not forbidden.search(line), f"{path.name}:{number}: {line.strip()}"
