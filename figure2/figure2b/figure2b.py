"""Figure 2B — mean stigma rates by clinical application (horizontal gradient bars).

Aggregation matches ``create_figure2cd_dotplots.aggregate_dot_df`` (macro mean over models
per category, descending order, zero-macro categories dropped). Plot style matches the
published panel: horizontal bars with value-graded color (high ≈ orange–red, low ≈ yellow–green),
``0–10`` % x-axis ticks, bar-end labels, dashed vertical grid, rectangular frame.

Writes ``figure2b.svg``, ``figure2b.pdf``, and ``figure2b_stats.txt`` in this folder.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
REBUILD = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
FIG2CD = REBUILD / "figure2cd" / "figure2cd.py"


def _load_fig2cd():
    spec = importlib.util.spec_from_file_location("figure2cd", str(FIG2CD))
    if spec is None or spec.loader is None:
        raise ImportError(FIG2CD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["figure2cd_reused"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_stats_txt(path: Path, df: pd.DataFrame, n_models: int) -> None:
    lines = [
        f"models_n={n_models}",
        f"categories_shown_n={len(df)}",
        "",
        "Mean stigma rate (macro): mean over models of each model's mean rate within category.",
        "Order: descending macro mean. Excludes categories with Task_Count=0 or macro mean 0.",
        "",
    ]
    for _, r in df.iterrows():
        cat = str(r["Category"]).replace("\t", " ")
        mm = float(r["Mean_all_models_macro_pct"])
        tc = int(r["Task_Count"])
        lines.append(f"{cat}\t{mm:.4f}%\ttasks={tc}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_horizontal_bars(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Match published Figure 2B: gradient bars, 0-10 % axis, labels at bar ends."""
    vals = df["Mean_all_models_macro_pct"].to_numpy(dtype=float)
    labels = [str(x) for x in df["Category"].tolist()]
    n = len(vals)
    y = np.arange(n)
    t = np.linspace(0.0, 1.0, n)
    cmap = plt.get_cmap("RdYlGn_r")
    color_positions = 0.85 - 0.7 * t
    colors = [cmap(0.2 + val / 15.0) for val in vals]

    ax.barh(y, vals, height=0.72, color=colors, edgecolor="black", linewidth=0.45, zorder=2)
    xmax_axis = 11.0
    for yi, v in zip(y, vals):
        tx = float(v) + 0.12
        if tx > xmax_axis - 0.25:
            tx = float(v) - 0.55
            ha = "right"
        else:
            ha = "left"
        ax.text(
            tx,
            float(yi),
            f"{v:.2f}%",
            va="center",
            ha=ha,
            fontsize=16,
            color="#1a1a1a",
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=16)
    ax.set_xlabel("Stigma Rate (%)", fontsize=22, fontweight="bold")
    ax.set_ylabel("Clinical Application", fontsize=22, fontweight="bold")
    ax.set_xlim(0, xmax_axis)
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.75, color="#b0b0b0", alpha=0.9, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.tick_params(axis="both", labelsize=16, width=0.9, length=4)
    # ax.set_title("Figure 2B. stigma rates by clinical application category", fontsize=16, fontweight="bold", pad=12)
    ax.invert_yaxis()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stigma",
        type=Path,
        default=BASE / "Data" / "stigma_info_107models.xlsx",
        help="Default matches create_figure2cd_dotplots.py.",
    )
    p.add_argument(
        "--bench",
        type=Path,
        default=BASE / "Data" / "Clinical Benchmark and LLM 107.xlsx",
    )
    p.add_argument("--out-dir", type=Path, default=HERE)
    args = p.parse_args()

    cd = _load_fig2cd()
    task_headers, frac_block, model_names = cd.fb.load_fraction_matrix(args.stigma)
    cd.assert_highlight_models(model_names)
    df_tasks = pd.read_excel(args.bench, sheet_name="Task-all")

    df_b = cd.aggregate_dot_df(
        frac_block=frac_block,
        model_names=model_names,
        task_headers=task_headers,
        df_tasks=df_tasks,
        column="Clinical Application",
    )

    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
    plt.rcParams["axes.unicode_minus"] = False
    n = len(df_b)
    fig_h = max(6.0, 0.52 * n + 2.0)
    fig_w = 9.5
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    _plot_horizontal_bars(ax, df_b)
    # fig.subplots_adjust(left=0.34, right=0.96, top=0.90, bottom=0.10)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    svg = out_dir / "figure2b.svg"
    pdf = out_dir / "figure2b.pdf"
    fig.savefig(svg, dpi=1200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, dpi=1200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats_path = out_dir / "figure2b_stats.txt"
    _write_stats_txt(stats_path, df_b, n_models=len(model_names))
    print(svg.resolve())
    print(pdf.resolve())
    print(stats_path.resolve())


if __name__ == "__main__":
    main()
