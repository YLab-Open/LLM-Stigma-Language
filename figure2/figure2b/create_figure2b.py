"""Shared loaders for stigma `rate` + benchmark `Task-all` joins.

Used by `figure2_highlight_four_107/create_figure2cd_dotplots.py`. To render
Figure 2B, run ``run_figure2b_107.py`` in this folder (see README.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def normalize_task_key(header: str) -> str:
    t = str(header).strip()
    if t.endswith(".result.json"):
        t = t[: -len(".result.json")]
    t = t.split("-cot-")[0]
    return t


def build_taxonomy_lookups(df_tasks: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    col = "Task-Original"
    if col not in df_tasks.columns:
        raise ValueError(f"Task-all sheet missing column {col!r}")
    by_exact: dict[str, pd.Series] = {}
    by_lower: dict[str, pd.Series] = {}
    for _, row in df_tasks.iterrows():
        key = str(row.get(col, "")).strip()
        if not key or key.lower() == "nan":
            continue
        by_exact[key] = row
        by_lower[key.lower()] = row
    return by_exact, by_lower


def resolve_row(
    normalized_key: str,
    by_exact: dict[str, pd.Series],
    by_lower: dict[str, pd.Series],
) -> pd.Series | None:
    if normalized_key in by_exact:
        return by_exact[normalized_key]
    low = normalized_key.lower()
    if low in by_lower:
        return by_lower[low]
    return None


def load_fraction_matrix(stigma_xlsx: Path) -> tuple[list[str], np.ndarray, list[str]]:
    rate = pd.read_excel(stigma_xlsx, sheet_name="rate", header=None)
    task_headers = [str(x).strip() for x in rate.iloc[1, 1:-1].tolist()]
    n_task = len(task_headers)
    rows: list[np.ndarray] = []
    names: list[str] = []
    for i in range(2, len(rate)):
        name = str(rate.iloc[i, 0]).strip()
        if not name or name.lower() == "nan" or name == "Average":
            continue
        if name == "Input":
            continue
        vals = pd.to_numeric(rate.iloc[i, 1 : 1 + n_task], errors="coerce").to_numpy(dtype=float)
        if vals.shape[0] != n_task:
            raise ValueError(f"Row {name}: expected {n_task} task columns")
        rows.append(vals)
        names.append(name)
    if not rows:
        raise ValueError(f"No model rows in {stigma_xlsx}")
    frac_block = np.vstack(rows)
    return task_headers, frac_block, names
