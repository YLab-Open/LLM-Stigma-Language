from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from figure2.figure2b.create_figure2b import (
    build_taxonomy_lookups,
    normalize_task_key,
    resolve_row,
)

STIGMA_XLSX = REPO / "Data" / "stigma_info_107models.xlsx"
BENCHMARK_XLSX = REPO / "Data" / "Clinical Benchmark and LLM 107.xlsx"
N_TASKS = 35

PANEL_COLORS = {
    "a": "#F59B7B",
    "b": "#EDC66A",
    "c": "#A8D3A0",
    "d": "#8FB4DC",
}

# Footnote lines appended below the caption
FOOTNOTES = (
    "* Tasks with multiple labels in a category are counted in each relevant label (e.g., a task with clinical specialty \"Cardiology, Endocrinology\" counts once in both \"Cardiology\" and \"Endocrinology\"). "
)

LOWERCASE_TITLE_WORDS = {
    "a", "an", "the",
    "and", "but", "for", "nor", "or", "so", "yet",
    "as", "at", "by", "from", "in", "into", "of", "on", "onto", "per", "to", "via", "vs", "with",
}


def _clean_application(s: str) -> str:
    t = str(s).strip()
    if t == "Procudure information":
        return "Procedure information"
    return t


def _format_title_word(word: str) -> str:
    lower = word.lower()
    if lower in LOWERCASE_TITLE_WORDS:
        return lower
    if len(word) > 1 and word.isupper():
        return word
    return word[:1].upper() + word[1:].lower()


def _title_case_label(label: str) -> str:
    return re.sub(r"[A-Za-z]+", lambda m: _format_title_word(m.group(0)), label)


def _split_labels(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [_title_case_label(part.strip()) for part in str(value).split(",") if part.strip()]


def load_subframe() -> pd.DataFrame:
    rate = pd.read_excel(STIGMA_XLSX, sheet_name="rate", header=None)
    raw_headers = [str(x).strip() for x in rate.iloc[1, 1:-1].tolist()]
    if len(raw_headers) != N_TASKS:
        raise ValueError(f"Expected {N_TASKS} task columns, got {len(raw_headers)}")

    tasks = pd.read_excel(BENCHMARK_XLSX, sheet_name="Task-all")
    by_exact, by_lower = build_taxonomy_lookups(tasks)

    rows: list[pd.Series] = []
    for h in raw_headers:
        nk = normalize_task_key(h)
        row = resolve_row(nk, by_exact, by_lower)
        if row is None:
            raise KeyError(f"No Task-all row for {h!r} (normalized {nk!r})")
        rows.append(row)

    out = pd.DataFrame(rows)
    out["_a"] = out["Task Type"].astype(str).str.strip()
    out["_b"] = out["Clinical Application"].map(_clean_application)
    out["_c"] = out["Source Document Type"].astype(str).str.strip()
    out["_d"] = out["Clinical context"].astype(str).str.strip()
    return out


def ordered_counts(
    series: pd.Series, *, ascending: bool, tie_label_order: bool = True
) -> tuple[np.ndarray, list[str]]:
    split = series.map(_split_labels).explode()
    split = split[split.notna()]
    vc = split.value_counts().reset_index()
    vc.columns = ["label", "n"]
    if tie_label_order:
        vc = vc.sort_values(["n", "label"], ascending=[ascending, ascending])
    else:
        vc = vc.sort_values("n", ascending=ascending)
    return vc["n"].to_numpy(dtype=float), vc["label"].astype(str).tolist()


def _panel_barh(
    ax: plt.Axes,
    counts: np.ndarray,
    labels: list[str],
    color: str,
    panel: str,
    subtitle: str,
    x_hi: float,
    x_step: float,
    invert_y: bool,
) -> None:
    if len(counts) != len(labels):
        raise ValueError(f"Panel {panel}: {len(counts)} counts for {len(labels)} labels")

    y = np.arange(len(labels))
    max_count = float(counts.max()) if len(counts) else 0.0
    if max_count >= x_hi:
        x_hi = float(np.ceil((max_count * 1.15) / x_step) * x_step)

    ax.barh(
        y,
        counts,
        height=0.58,
        color=color,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.88,
        zorder=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color="#2b2b2b")
    ax.set_xlabel("Number of tasks", fontsize=9.5, color="#444444", labelpad=5)
    ax.set_xlim(0, x_hi)
    ax.xaxis.set_major_locator(plt.MultipleLocator(x_step))
    ax.tick_params(axis="x", colors="#555555", length=3, width=0.6, labelsize=8)
    ax.tick_params(axis="y", length=0)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d0d0d0")
        ax.spines[s].set_linewidth(0.8)

    ax.set_facecolor("#f8f8f8")
    ax.grid(axis="x", linestyle=(0, (1, 3)), color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Panel letter badge
    ax.text(
        -0.06, 1.04,
        f"({panel})",
        transform=ax.transAxes,
        fontsize=11, fontweight="bold", color="#1a1a1a",
        va="bottom", ha="right",
    )
    ax.set_title(subtitle, loc="left", fontsize=10.5, fontweight="bold",
                 color="#1a1a1a", pad=6)

    if invert_y:
        ax.invert_yaxis()

    # Count labels on bars
    xr = ax.get_xlim()[1]
    for yi, c in zip(y, counts):
        pct = 100.0 * float(c) / float(N_TASKS)
        ax.text(
            float(c) + 0.025 * xr,
            float(yi),
            f"{int(c)} ({pct:.0f}%)",
            va="center", fontsize=7.5, color="#444444", zorder=3,
        )


def main() -> tuple[Path, Path | None]:
    df = load_subframe()

    cnt_a, lab_a = ordered_counts(df["_a"], ascending=False)
    cnt_b, lab_b = ordered_counts(df["_b"], ascending=False)
    cnt_c, lab_c = ordered_counts(df["_c"], ascending=False)
    cnt_d, lab_d = ordered_counts(df["_d"], ascending=False)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f8f8f8",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.8), constrained_layout=False)
    fig.patch.set_facecolor("white")

    _panel_barh(axes[0, 0], cnt_a, lab_a,
                PANEL_COLORS["a"], "A", "NLP Task Type", 20, 4, invert_y=True)
    _panel_barh(axes[0, 1], cnt_c, lab_c,
                PANEL_COLORS["b"], "B", "Source Clinical Document Type", 16, 4, invert_y=True)
    _panel_barh(axes[1, 0], cnt_d, lab_d,
                PANEL_COLORS["c"], "C", "Clinical Specialty", 11, 2, invert_y=True)
    _panel_barh(axes[1, 1], cnt_b, lab_b,
                PANEL_COLORS["d"], "D", "Clinical Application", 7, 1, invert_y=True)

    fig.subplots_adjust(wspace=0.75)

    fig.suptitle(
        "Task Taxonomy of The 35 English Clinical Tasks",
        fontsize=13, fontweight="bold", color="#111111",
        x=0.5, y=0.95, ha="center"
    )

    # cap = (
    #     "Distribution of tasks by (A) NLP task type, (B) sourced clinical document type, "
    #     "(C) clinical specialty, and (D) clinical application."
    # )
    # fig.text(0.5, 0.075, cap,
    #          ha="center", va="top", fontsize=9.5, color="#333333")
    fig.text(0.5, 0.065, FOOTNOTES,
             ha="center", va="top", fontsize=8, color="#666666",
             style="italic")

    out_dir = Path(__file__).resolve().parent
    svg = out_dir / "figure_s1_task_taxonomy.svg"
    pdf = out_dir / "figure_s1_task_taxonomy.pdf"
    fig.savefig(svg, dpi=1200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, dpi=1200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return svg, pdf


if __name__ == "__main__":
    p, q = main()
    print(p)
    if q:
        print(q)
