"""Stable content fingerprint for DataFrames."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
from pandas.util import hash_pandas_object


def fingerprint(df: pd.DataFrame) -> str:
    """Return a sha256 hex digest over column names, dtypes, cell values and the index."""
    h = hashlib.sha256()
    header = json.dumps([[str(c) for c in df.columns], [str(t) for t in df.dtypes]])
    h.update(header.encode("utf-8"))
    h.update(hash_pandas_object(df, index=True).to_numpy(dtype="<u8").tobytes())
    return h.hexdigest()
