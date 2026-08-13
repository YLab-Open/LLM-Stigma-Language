"""Figure 4: input vs output stigma rate (task-level + model–task pairs).

Reads `stigma_info_107models.xlsx` in this same folder. Replace that file to rerun.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REBUILD = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
EXCEL_PATH = REBUILD / "Data" / "stigma_info_107models.xlsx"
OUT_BASE = HERE / "figure4b"
SHEET_RATE = "rate"
X_LIM = 16.0
Y_LIM_A = 16.0
Y_LIM_B = 35.0


def load_rate_from_excel(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    df = pd.read_excel(path, sheet_name=SHEET_RATE, header=None)
    if df.shape[1] < 3:
        raise ValueError("rate sheet needs at least 3 columns")

    r0, r1 = df.iloc[0, 0], df.iloc[1, 0]
    if str(r0).strip().lower() != "input" or str(r1).strip().lower() != "model":
        raise ValueError("expected row0 col0 'Input' and row1 col0 'Model'")

    block = df.iloc[:, 1:-1]
    input_row = pd.to_numeric(block.iloc[0], errors="coerce").to_numpy(dtype=float)
    input_row = np.where(~np.isfinite(input_row), np.nan, np.where(input_row <= 1.0, input_row, input_row / 100.0))
    out_block = df.iloc[2:, 1:-1].copy()
    if len(out_block) == 0:
        raise ValueError("no model rows below header")

    if str(df.iloc[-1, 0]).strip().lower() == "average":
        out_block = out_block.iloc[:-1]

    out_mat = out_block.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    out_mat = np.where(~np.isfinite(out_mat), np.nan, np.where(out_mat <= 1.0, out_mat, out_mat / 100.0))

    input_pct = input_row * 100.0
    out_pct = out_mat * 100.0
    n_models, n_tasks = out_pct.shape
    if input_pct.shape[0] != n_tasks:
        raise ValueError("input length must match number of task columns")
    return input_pct, out_pct, n_models, n_tasks


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan"), float("nan")
    r, p = pearsonr(x[m], y[m])
    return float(r), float(p)


def _diag_split(x: np.ndarray, y: np.ndarray) -> tuple[int, int, int]:
    atol = 1e-9
    m = np.isfinite(x) & np.isfinite(y)
    xd, yd = x[m], y[m]
    lt = int(np.sum(yd - xd < -atol))
    gt = int(np.sum(yd - xd > atol))
    eq = int(m.sum() - lt - gt)
    return lt, eq, gt


def plot_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    y_label: str,
    x_max: float,
    y_max: float,
) -> None:
    # Count how many points are below, on, and above the diagonal y=x line
    total = above = below = on = 0
    for i in range(len(x)):
        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            continue
        total += 1
        if x[i] > y[i] :
            color = "#8FB4DC"  # blue for below diagonal
            below += 1
        elif y[i] > x[i] :
            color = "#F59B7B"  # orange for above diagonal
            above += 1
        else:
            color = "#3CA02A"  # green for on diagonal
            on += 1
        ax.scatter(x[i], y[i], s=28, alpha=0.45, facecolors=color, edgecolors="#0d47a1", linewidths=0.35)
    print(f"{title} - total: {total}, below: {below} ({below/total*100:.2f}%), on: {on} ({on/total*100:.2f}%), above: {above} ({above/total*100:.2f}%)")
    ax.plot([0, max(x_max, y_max)], [0, max(x_max, y_max)], color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Input Text Stigma Rate (%)", fontsize=16, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=16, fontweight="bold")
    # Make x and y ticks larger
    ax.tick_params(axis="both", which="major", labelsize=16)
    # ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="-", linewidth=0.45, alpha=0.28)
    ax.set_axisbelow(True)

    r, p = _pearson(x, y)
    p_txt = "p < 0.001" if p < 0.001 else f"p = {p:.3g}"
    ax.text(
        0.97,
        0.97,
        f"Pearson r = $\\bf{{{r:.3f}}}$\n{p_txt}",
        transform=ax.transAxes,
        fontsize=16,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.72", alpha=0.96),
    )


def main() -> None:
    path = EXCEL_PATH
    if not path.is_file():
        raise FileNotFoundError(path)

    input_pct, out_pct, n_models, n_tasks = load_rate_from_excel(path)
    mean_out = np.nanmean(out_pct, axis=0)

    r_a, _ = _pearson(input_pct, mean_out)
    x_b = np.tile(input_pct, n_models)
    y_b = out_pct.reshape(-1)
    r_b, _ = _pearson(x_b, y_b)

    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
    plt.rcParams["axes.unicode_minus"] = False

    # fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.2, 6.2))
    fig, ax_b = plt.subplots(1, 1, figsize=(6.8, 6.2))

    # plot_panel(
    #     ax_a,
    #     input_pct,
    #     mean_out,
    #     f"A) Task-level average ({n_tasks} tasks)",
    #     "Reasoning Traces Stigma Rate (%)\n(average across models)",
    #     X_LIM,
    #     Y_LIM_A,
    # )

    n_pair = n_models * n_tasks
    plot_panel(
        ax_b,
        x_b,
        y_b,
        f"B) All model-task combinations ({n_pair:,} pairs)",
        "Reasoning Traces Stigma Rate (%)",
        X_LIM,
        Y_LIM_B,
    )

    fig.tight_layout()
    fig.savefig(f"{OUT_BASE}.svg", dpi=1200, bbox_inches="tight")
    fig.savefig(f"{OUT_BASE}.pdf", bbox_inches="tight")
    plt.close(fig)

    lt, eq, gt = _diag_split(x_b, y_b)
    n_v = int(np.sum(np.isfinite(x_b) & np.isfinite(y_b)))
    stats = [
        f"excel={path}",
        f"models_n={n_models} tasks_n={n_tasks} pairs_n={n_pair}",
        f"task_level_r={r_a:.6f}",
        f"pair_level_r={r_b:.6f}",
        f"pair_below_diagonal={lt} ({lt/n_v*100:.4f}%)",
        f"pair_on_diagonal={eq} ({eq/n_v*100:.4f}%)",
        f"pair_above_diagonal={gt} ({gt/n_v*100:.4f}%)",
    ]
    Path(f"{OUT_BASE}_stats.txt").write_text("\n".join(stats) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
