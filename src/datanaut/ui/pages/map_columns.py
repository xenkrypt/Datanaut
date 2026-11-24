"""Map columns screen: choose the time and value columns, then build the time index (V1)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from datanaut.ui import actions
from datanaut.ui.state import WizardState


def render(state: WizardState) -> None:
    st.header("Map columns")
    st.write("Tell Datanaut which column holds the time and which hold the readings to analyze.")
    assert state.loaded is not None
    raw = state.loaded.df
    suggested_time, suggested_values = actions.suggest_mapping(raw)
    default_time = state.time_columns or suggested_time
    default_values = state.value_columns or suggested_values

    time_cols = st.multiselect(
        "Timestamp column (or a date column and a time column)",
        list(raw.columns),
        default=default_time,
        max_selections=2,
        key="map_time",
    )
    if not time_cols:
        st.info("Choose the column that holds the time of each reading.")
        return
    if suggested_time and time_cols != suggested_time:
        st.caption(f"Suggested: {' + '.join(suggested_time)}")

    dayfirst = _dayfirst_choice(state, time_cols)
    preview = actions.guard(
        state, "Map columns", actions.preview_time_index, state, time_cols, dayfirst
    )
    if preview is None:
        return
    flags = preview.flags
    if not flags.ambiguous:
        _time_summary(preview)

    value_options = [c for c in raw.columns if c not in time_cols and _numeric(raw[c])]
    values = st.multiselect(
        "Value columns to analyze",
        value_options,
        default=[c for c in default_values if c in value_options],
        key="map_values",
    )
    consumption = {}
    if values:
        st.caption(
            "Mark the columns that measure energy or consumption (this enables energy metrics)."
        )
        for col, name in zip(st.columns(min(len(values), 4)), values[:4], strict=False):
            consumption[name] = col.checkbox(
                f"{name} is consumption",
                value=state.consumption.get(name, False),
                key=f"cons_{name}",
            )
        for name in values[4:]:
            consumption[name] = st.checkbox(
                f"{name} is consumption",
                value=state.consumption.get(name, False),
                key=f"cons_{name}",
            )

    blocked = flags.ambiguous or not values
    if st.button("Build time index and create V1", type="primary", disabled=blocked):
        created = actions.guard(
            state,
            "Map columns",
            actions.confirm_mapping,
            state,
            time_cols,
            dayfirst,
            values,
            consumption,
        )
        if created is not None:
            state.add_message(
                "success",
                "Time index built. V1 (the raw data) is ready."
                if created
                else "Value columns updated.",
            )
            state.go("Clean")
            st.rerun()
    if state.store is not None:
        st.success("Columns are mapped. You can continue, or change the choices above.")
        st.button("Continue to Clean", on_click=state.go, args=("Clean",), key="map_next")


def _numeric(series: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series)) and not pd.api.types.is_bool_dtype(series)


def _dayfirst_choice(state: WizardState, time_cols: list[str]) -> bool | None:
    """Ask day-first or month-first only when the core says both readings are possible."""
    auto = actions.guard(state, "Map columns", actions.preview_time_index, state, time_cols, None)
    if auto is None or not auto.flags.ambiguous:
        return None
    st.warning(
        "These dates can be read two ways. Pick the one that matches your data. "
        f"{auto.flags.ambiguous_values:,} value(s) are affected."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"Value in file": s.raw, "Day first": s.dayfirst, "Month first": s.monthfirst}
                for s in auto.flags.ambiguous_preview
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    choice = st.radio(
        "How are dates written?",
        ["Not chosen yet", "Day first (31/12/2024)", "Month first (12/31/2024)"],
        key="map_dayfirst",
    )
    return {"Day first (31/12/2024)": True, "Month first (12/31/2024)": False}.get(choice)


def _time_summary(preview) -> None:
    flags, df = preview.flags, preview.df
    if len(df):
        st.write(
            f"Readings run from **{df.index.min()}** to **{df.index.max()}** ({len(df):,} rows)."
        )
    if flags.unparseable_rows or flags.missing_timestamp_rows:
        st.warning(
            f"{flags.unparseable_rows:,} row(s) have a timestamp that could not be read and "
            f"{flags.missing_timestamp_rows:,} have none; they are set aside, not deleted."
        )
