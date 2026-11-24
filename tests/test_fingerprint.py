import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from datanaut.core.fingerprint import fingerprint


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {"a": [1, 2, 3], "b": [1.5, np.nan, 2.5], "c": ["x", "y", "z"]},
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )


def test_stable_across_calls_and_copies():
    df = _df()
    assert fingerprint(df) == fingerprint(df) == fingerprint(df.copy())
    assert len(fingerprint(df)) == 64


def test_does_not_mutate_input():
    df = _df()
    before = df.copy()
    fingerprint(df)
    pd.testing.assert_frame_equal(df, before)


def test_changes_with_cell_column_dtype_and_index():
    df = _df()
    base = fingerprint(df)
    changed = df.copy()
    changed.loc[changed.index[0], "a"] = 99
    assert fingerprint(changed) != base
    assert fingerprint(df.rename(columns={"a": "aa"})) != base
    assert fingerprint(df.astype({"a": "float64"})) != base
    shifted = df.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=1)
    assert fingerprint(shifted) != base


@settings(max_examples=50, deadline=None)
@given(
    values=st.lists(st.integers(-1000, 1000), min_size=1, max_size=30),
    data=st.data(),
)
def test_any_single_cell_change_changes_hash(values, data):
    df = pd.DataFrame({"v": values})
    pos = data.draw(st.integers(0, len(values) - 1))
    new = data.draw(st.integers(-1000, 1000).filter(lambda x: x != values[pos]))
    other = df.copy()
    other.iloc[pos, 0] = new
    assert fingerprint(other) != fingerprint(df)
    assert fingerprint(df) == fingerprint(pd.DataFrame({"v": values}))
