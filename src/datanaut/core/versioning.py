"""Dataset versioning: store recipes, not copies.

The raw dataset is kept once, as a private copy. Every version is the raw data plus a chain of
`OpSpec`s, rebuilt by replaying that chain; each rebuild is checked against the fingerprint
recorded when the version was created. Restoring an old version moves the head pointer, so later
operations branch and no history is ever deleted.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from datanaut.core.cleaner import OpSpec, apply_op
from datanaut.core.fingerprint import fingerprint
from datanaut.core.log import PipelineLog, library_versions
from datanaut.core.validation import inline_rule_reference
from datanaut.errors import DatanautError, RecipeMismatchError, VersioningError

RECIPE_FORMAT = 1
EXAMPLE_LIMIT = 20
"""At most this many changed cells (and added or removed row labels) are listed in a diff."""


# ----------------------------------------------------------------------------------- models


class DatasetVersion(BaseModel):
    """One node in the version tree."""

    version_id: str
    parent_id: str | None = None
    op: OpSpec | None = None  # None for V1, the raw data
    fingerprint: str
    rows: int
    columns: int
    created_at: str
    environment: dict[str, str]


class VersionNode(BaseModel):
    """A version with its place in the tree, for display."""

    version: DatasetVersion
    children: list[str]
    depth: int
    is_head: bool


class CellChange(BaseModel):
    """One cell that differs between two versions."""

    row: str
    column: str
    before: Any = None
    after: Any = None


class VersionDiff(BaseModel):
    """What differs between two datasets. Rows are matched by index label, duplicates by order."""

    a: str
    b: str
    identical: bool
    columns_added: list[str] = Field(default_factory=list)
    columns_removed: list[str] = Field(default_factory=list)
    dtype_changes: dict[str, str] = Field(default_factory=dict)
    rows_added: int = 0
    rows_removed: int = 0
    added_sample: list[str] = Field(default_factory=list)
    removed_sample: list[str] = Field(default_factory=list)
    cells_changed: dict[str, int] = Field(default_factory=dict)
    cells_changed_total: int = 0
    examples: list[CellChange] = Field(default_factory=list)


class RecipeStep(BaseModel):
    """One operation in a recipe and the data fingerprint it must reproduce."""

    version_id: str
    op: OpSpec
    fingerprint: str
    rows: int
    columns: int


class Recipe(BaseModel):
    """The operation chain from the raw data to one version, with fingerprints to verify it."""

    format_version: int = RECIPE_FORMAT
    raw_fingerprint: str
    raw_rows: int
    raw_columns: int
    steps: list[RecipeStep]
    head_fingerprint: str
    environment: dict[str, str]
    created_at: str
    source: dict[str, Any] = Field(default_factory=dict)
    """Free-form notes on how V1 was built from the file (sheet, time columns, day order)."""

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize the recipe."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> Recipe:
        """Parse a recipe, raising VersioningError if it is malformed."""
        try:
            return cls.model_validate_json(text.lstrip("\ufeff"))
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"]) or "recipe"
            raise VersioningError(f"The recipe is not valid ({where}: {first['msg']}).") from exc


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ------------------------------------------------------------------------------------ store


class VersionStore:
    """The raw data plus a tree of versions, each defined by an operation on its parent."""

    def __init__(
        self,
        raw: pd.DataFrame,
        log: PipelineLog | None = None,
        *,
        cache_size: int = 3,
    ) -> None:
        self._raw = raw.copy(deep=True)  # private: callers only ever receive copies
        self._log = log if log is not None else PipelineLog()
        self._cache_size = max(0, cache_size)
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        first = DatasetVersion(
            version_id="V1",
            parent_id=None,
            op=None,
            fingerprint=fingerprint(self._raw),
            rows=len(self._raw),
            columns=self._raw.shape[1],
            created_at=_now(),
            environment=library_versions(),
        )
        self._versions: dict[str, DatasetVersion] = {"V1": first}
        self._children: dict[str, list[str]] = {"V1": []}
        self._head = "V1"

    # --- access -----------------------------------------------------------------------

    @property
    def log(self) -> PipelineLog:
        """The pipeline log this store appends to."""
        return self._log

    @property
    def head(self) -> str:
        """The id of the working version."""
        return self._head

    def _meta(self, version_id: str) -> DatasetVersion:
        if version_id not in self._versions:
            known = ", ".join(self._versions)
            raise VersioningError(f"There is no version '{version_id}'. Versions: {known}.")
        return self._versions[version_id]

    def version(self, version_id: str) -> DatasetVersion:
        """A copy of one version's metadata (changing it does not change the store)."""
        return self._meta(version_id).model_copy(deep=True)

    def raw(self) -> pd.DataFrame:
        """A copy of the raw data (V1)."""
        return self._raw.copy(deep=True)

    def current(self) -> pd.DataFrame:
        """A copy of the head version's data."""
        return self.get(self._head)

    def cached_ids(self) -> list[str]:
        """Versions whose frames are cached, least recently used first."""
        return list(self._cache)

    def get(self, version_id: str) -> pd.DataFrame:
        """Rebuild a version by replaying its operations. Always returns a fresh copy."""
        self._meta(version_id)
        hit = self._cache_get(version_id)
        if hit is not None:
            return hit.copy(deep=True)
        pending: list[DatasetVersion] = []
        node = version_id
        while True:
            cached = self._cache_get(node)
            if cached is not None:
                frame = cached.copy(deep=True)
                break
            meta = self._versions[node]
            if meta.parent_id is None:
                frame = self._raw.copy(deep=True)
                break
            pending.append(meta)
            node = meta.parent_id
        for meta in reversed(pending):
            frame = apply_op(frame, meta.op).df  # type: ignore[arg-type]
            if fingerprint(frame) != meta.fingerprint:
                raise VersioningError(
                    f"Rebuilding {meta.version_id} did not reproduce its recorded fingerprint. "
                    "The environment may have changed since it was created."
                )
        self._cache_put(version_id, frame)
        return frame.copy(deep=True)

    def _cache_get(self, version_id: str) -> pd.DataFrame | None:
        if version_id not in self._cache:
            return None
        self._cache.move_to_end(version_id)
        return self._cache[version_id]

    def _cache_put(self, version_id: str, frame: pd.DataFrame) -> None:
        if self._cache_size == 0:
            return
        self._cache[version_id] = frame.copy(deep=True)
        self._cache.move_to_end(version_id)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    # --- changing ---------------------------------------------------------------------

    def apply(self, spec: OpSpec) -> DatasetVersion:
        """Apply an operation to the head version, creating a child version that becomes head.

        If the operation raises, nothing is created and the store is unchanged.
        """
        spec = _frozen(spec)  # the store keeps its own copy, with any rule file contents inlined
        parent = self._head
        result = apply_op(self.get(parent), spec)
        version_id = f"V{len(self._versions) + 1}"
        new = DatasetVersion(
            version_id=version_id,
            parent_id=parent,
            op=spec,
            fingerprint=fingerprint(result.df),
            rows=len(result.df),
            columns=result.df.shape[1],
            created_at=_now(),
            environment=library_versions(),
        )
        self._versions[version_id] = new
        self._children[version_id] = []
        self._children[parent].append(version_id)
        self._head = version_id
        self._cache_put(version_id, result.df)
        entry = result.log_entry.model_copy(
            update={
                "params": {
                    **result.log_entry.params,
                    "dataset_version": {
                        "version_id": version_id,
                        "parent_id": parent,
                        "fingerprint": new.fingerprint,
                    },
                }
            }
        )
        self._log.extend([entry])
        return new.model_copy(deep=True)

    def restore(self, version_id: str) -> DatasetVersion:
        """Make an earlier version the head. Nothing is deleted; later changes branch from it."""
        target = self._meta(version_id)
        previous = self._versions[self._head]
        self._head = version_id
        self._log.append(
            "restore_version",
            f"Restored {version_id} as the working version (was {previous.version_id}); "
            f"later changes will branch from {version_id}. No version was deleted.",
            params={"from_version": previous.version_id, "to_version": version_id},
            counts={"rows": target.rows, "columns": target.columns},
            rows_before=previous.rows,
            rows_after=target.rows,
        )
        return target.model_copy(deep=True)

    # --- reading the tree -------------------------------------------------------------

    def history(self) -> list[VersionNode]:
        """All versions in tree order (parents before children), with depth for indenting."""
        out: list[VersionNode] = []
        stack: list[tuple[str, int]] = [("V1", 0)]
        while stack:
            version_id, depth = stack.pop()
            out.append(
                VersionNode(
                    version=self._versions[version_id].model_copy(deep=True),
                    children=list(self._children[version_id]),
                    depth=depth,
                    is_head=version_id == self._head,
                )
            )
            stack.extend((child, depth + 1) for child in reversed(self._children[version_id]))
        return out

    def chain(self, version_id: str | None = None) -> list[DatasetVersion]:
        """The versions from V1 to `version_id` (default: head), oldest first."""
        node: str | None = version_id or self._head
        self._meta(node)
        path: list[DatasetVersion] = []
        while node is not None:
            meta = self._versions[node]
            path.append(meta.model_copy(deep=True))
            node = meta.parent_id
        return path[::-1]

    def compare(self, a: str, b: str) -> VersionDiff:
        """What changed between two versions."""
        return compare_frames(self.get(a), self.get(b), a, b)

    # --- recipes ----------------------------------------------------------------------

    def to_recipe(
        self, version_id: str | None = None, *, source: dict[str, Any] | None = None
    ) -> Recipe:
        """The operation chain that rebuilds a version (default: head) from the raw data.

        `source` records how the raw data was produced from its file, so a replay can redo it.
        """
        chain = self.chain(version_id)
        return Recipe(
            raw_fingerprint=self._versions["V1"].fingerprint,
            raw_rows=self._versions["V1"].rows,
            raw_columns=self._versions["V1"].columns,
            steps=[
                RecipeStep(
                    version_id=v.version_id,
                    op=v.op,  # type: ignore[arg-type]
                    fingerprint=v.fingerprint,
                    rows=v.rows,
                    columns=v.columns,
                )
                for v in chain[1:]
            ],
            head_fingerprint=chain[-1].fingerprint,
            environment=library_versions(),
            created_at=_now(),
            source=dict(source or {}),
        )


def replay_recipe(
    raw_df: pd.DataFrame,
    recipe: Recipe | dict[str, Any] | str,
    *,
    log: PipelineLog | None = None,
    cache_size: int = 3,
) -> VersionStore:
    """Rebuild a dataset from raw data and a recipe, verifying every fingerprint.

    Returns a VersionStore whose head holds the rebuilt data. Raises RecipeMismatchError, naming
    the first step that differs, if the raw data or any result does not match the recipe.
    """
    if isinstance(recipe, str):
        recipe = Recipe.from_json(recipe)
    elif isinstance(recipe, dict):
        try:
            recipe = Recipe.model_validate(recipe)
        except ValidationError as exc:
            raise VersioningError(f"The recipe is not valid: {exc.errors()[0]['msg']}.") from exc
    if recipe.format_version != RECIPE_FORMAT:
        raise VersioningError(
            f"This recipe has format {recipe.format_version}; only format {RECIPE_FORMAT} "
            "is supported."
        )
    store = VersionStore(raw_df, log, cache_size=cache_size)
    here = library_versions()
    drift = (
        f" It was made with pandas {recipe.environment.get('pandas')} and numpy "
        f"{recipe.environment.get('numpy')}; this is pandas {here['pandas']} and numpy "
        f"{here['numpy']}."
        if recipe.environment != here
        else ""
    )
    if store.version("V1").fingerprint != recipe.raw_fingerprint:
        raise RecipeMismatchError(
            "The raw data does not match the data this recipe was made from "
            f"(fingerprint {store.version('V1').fingerprint[:12]}, expected "
            f"{recipe.raw_fingerprint[:12]}). Use the same file.{drift}"
        )
    for step in recipe.steps:
        try:
            version = store.apply(step.op)
        except DatanautError as exc:
            raise RecipeMismatchError(
                f"Step {step.version_id} ({step.op.operation}) could not be applied: {exc}{drift}"
            ) from exc
        if version.fingerprint != step.fingerprint:
            raise RecipeMismatchError(
                f"Step {step.version_id} ({step.op.operation}) produced fingerprint "
                f"{version.fingerprint[:12]} but the recipe expects {step.fingerprint[:12]}.{drift}"
            )
    return store


def _frozen(spec: OpSpec) -> OpSpec:
    """An independent copy of a spec, with a rule looked up in a file turned into an inline rule."""
    return inline_rule_reference(spec.model_copy(deep=True))


# --------------------------------------------------------------------------------- compare


def _row_keys(index_codes: np.ndarray, spread: int) -> np.ndarray:
    """Make labels unique: label code plus the running count of repeats of that label."""
    seen = pd.Series(index_codes).groupby(index_codes).cumcount().to_numpy()
    return index_codes.astype(np.int64) * spread + seen


def _jsonable(value: Any) -> Any:
    if value is None or (not isinstance(value, str | list | dict) and pd.isna(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bool | int | float | str):
        return value
    return str(value)


def _differs(before: pd.Series, after: pd.Series) -> np.ndarray:
    try:
        same = before.eq(after) | (before.isna() & after.isna())
    except TypeError:
        same = before.astype(str).eq(after.astype(str)) | (before.isna() & after.isna())
    return ~same.to_numpy(dtype=bool)


def compare_frames(
    a: pd.DataFrame, b: pd.DataFrame, a_name: str = "a", b_name: str = "b"
) -> VersionDiff:
    """Compare two frames: columns, rows (by index label) and cell values."""
    cols_a, cols_b = list(a.columns), list(b.columns)
    added = [c for c in cols_b if c not in cols_a]
    removed = [c for c in cols_a if c not in cols_b]
    common_cols = [c for c in cols_a if c in cols_b]
    dtype_changes = {
        str(c): f"{a[c].dtype} -> {b[c].dtype}" for c in common_cols if a[c].dtype != b[c].dtype
    }

    codes = pd.factorize(a.index.append(b.index))[0]
    codes_a, codes_b = codes[: len(a)], codes[len(a) :]
    spread = len(a) + len(b) + 1
    keys_a, keys_b = _row_keys(codes_a, spread), _row_keys(codes_b, spread)
    in_a_for_b = pd.Index(keys_a).get_indexer(keys_b)
    in_b_for_a = pd.Index(keys_b).get_indexer(keys_a)
    only_b = np.flatnonzero(in_a_for_b < 0)
    only_a = np.flatnonzero(in_b_for_a < 0)
    both_a = np.flatnonzero(in_b_for_a >= 0)
    both_b = in_b_for_a[both_a]

    changed_by_column: dict[str, int] = {}
    found: list[tuple[int, int, str]] = []
    for ci, col in enumerate(common_cols):
        before = a[col].iloc[both_a].reset_index(drop=True)
        after = b[col].iloc[both_b].reset_index(drop=True)
        bad = _differs(before, after)
        if bad.any():
            changed_by_column[str(col)] = int(bad.sum())
            found += [(int(r), ci, str(col)) for r in np.flatnonzero(bad)[:EXAMPLE_LIMIT]]
    found.sort()
    examples = [
        CellChange(
            row=str(a.index[both_a[r]]),
            column=col,
            before=_jsonable(a[common_cols[ci]].iloc[both_a[r]]),
            after=_jsonable(b[common_cols[ci]].iloc[both_b[r]]),
        )
        for r, ci, col in found[:EXAMPLE_LIMIT]
    ]
    total = sum(changed_by_column.values())
    return VersionDiff(
        a=a_name,
        b=b_name,
        identical=not (added or removed or dtype_changes or len(only_a) or len(only_b) or total),
        columns_added=[str(c) for c in added],
        columns_removed=[str(c) for c in removed],
        dtype_changes=dtype_changes,
        rows_added=len(only_b),
        rows_removed=len(only_a),
        added_sample=[str(v) for v in b.index[only_b[:EXAMPLE_LIMIT]]],
        removed_sample=[str(v) for v in a.index[only_a[:EXAMPLE_LIMIT]]],
        cells_changed=changed_by_column,
        cells_changed_total=total,
        examples=examples,
    )
