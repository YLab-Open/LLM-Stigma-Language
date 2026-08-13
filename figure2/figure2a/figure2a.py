import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
REBUILD = BASE.parent.parent
XLSX = REBUILD / "Data" / "stigma_info_107models.xlsx"
BUNDLE_107 = REBUILD / "figure2a"

_SKIP_MODELS = frozenset({"average", "model_average", "model_all"})


def main():
    df = pd.read_excel(XLSX, sheet_name="rate", header=None)
    task_names = df.iloc[1, 1:-1].tolist()
    body = df.iloc[2:, :].copy()
    body = body[~body.iloc[:, 0].astype(str).str.strip().str.lower().isin(_SKIP_MODELS)]
    if len(body) and str(body.iloc[-1, 0]).strip().lower() == "average":
        body = body.iloc[:-1]
    model_names = body.iloc[:, 0].astype(str).tolist()
    mat = body.iloc[:, 1:-1].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n_models = len(model_names)
    frac = np.where(np.isnan(mat), np.nan, np.where(mat <= 1.0, mat, mat / 100.0))
    pct = frac * 100.0

    out_svg = BASE / "figure2a.svg"
    out_pdf = BASE / "figure2a.pdf"
    out_txt = BASE / "figure2a_stats.txt"

    all_vals = pct.reshape(-1)
    all_vals = all_vals[np.isfinite(all_vals)]
    nonzero = all_vals[all_vals > 0]

    total_pairs = int(len(model_names) * len(task_names))
    nonzero_n = int(len(nonzero))
    zero_n = total_pairs - nonzero_n
    mean_nz = float(np.mean(nonzero)) if nonzero_n else float("nan")
    median_nz = float(np.median(nonzero)) if nonzero_n else float("nan")
    high_n = int(np.sum(nonzero >= 5.0))
    low_n = int(np.sum(nonzero < 1.0))
    rng_min = float(np.min(all_vals)) if len(all_vals) else float("nan")
    rng_max = float(np.max(all_vals)) if len(all_vals) else float("nan")

    lines = []
    lines.append(f"source_xlsx={XLSX.resolve()}")
    lines.append(f"models_n={n_models}")
    lines.append(f"total_pairs={total_pairs}")
    lines.append(f"nonzero_pairs={nonzero_n}")
    lines.append(f"nonzero_pct={nonzero_n/total_pairs*100.0:.6f}" if total_pairs else "nonzero_pct=nan")
    lines.append(f"mean_nonzero_pct={mean_nz:.6f}")
    lines.append(f"median_nonzero_pct={median_nz:.6f}")
    lines.append(f"high_ge_5_pct_pairs={high_n}")
    lines.append(f"high_ge_5_pct_share_nonzero={high_n/nonzero_n*100.0:.6f}" if nonzero_n else "high_ge_5_pct_share_nonzero=nan")
    lines.append(f"low_lt_1_pct_pairs={low_n}")
    lines.append(f"low_lt_1_pct_share_nonzero={low_n/nonzero_n*100.0:.6f}" if nonzero_n else "low_lt_1_pct_share_nonzero=nan")
    lines.append(f"range_all_pairs_min_pct={rng_min:.6f}")
    lines.append(f"range_all_pairs_max_pct={rng_max:.6f}")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
    plt.rcParams["axes.unicode_minus"] = False

    bins = np.arange(0, 14.5, 0.5)
    fig, ax = plt.subplots(figsize=(12, 6))
    counts, edges = np.histogram(nonzero, bins=bins)
    left = edges[:-1]
    # left = np.array([x + 0.25 for x in left])  # Shift the bars to the right by 0.25 to avoid overlap with zero bar
    widths = np.diff(edges)
    bin_midpoints = left + widths / 2.0
    cmap = plt.get_cmap("Reds")
    colors = [cmap(0.2 + val / 15.0) for val in bin_midpoints]
    ax.bar(left, counts, width=widths, align="edge", color=colors, edgecolor=(0.0902, 0.0902, 0.0902), linewidth=1.0)
    # The zero stigma rate bar
    ax.bar(
        -0.25,
        zero_n,
        width=0.5,
        align="center",
        color="#60966D",
        edgecolor=(0.0902, 0.0902, 0.0902),
        linewidth=1.0,
        zorder=3,
    )
    ax.axvline(mean_nz, color="#0E6A8E", linestyle=":", linewidth=1.5, label=f"Mean: {mean_nz:.2f}%")
    ax.axvline(median_nz, color="#531393", linestyle="--", linewidth=1.5, label=f"Median: {median_nz:.2f}%")
    ax.set_xlabel("Stigma Rate (%)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Number of Non-Zero Model-Task Pairs", fontsize=16, fontweight="bold")
    # Enlarge x and y axis ticks and labels
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlim(-0.5, 14)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, alpha=0.15)
    ax.set_axisbelow(True)

    txt = (
        f"Total Non-Zero stigma rate model-task pairs: $\\bf{{{nonzero_n:,}\\ ({nonzero_n/total_pairs*100.0:.2f}\\%)}}$\n"
        f"Total Zero stigma rate model-task pairs: $\\bf{{{zero_n:,}\\ ({zero_n/total_pairs*100.0:.2f}\\%)}}$\n"
        f"Stigma Rate Range: $\\bf{{{rng_min:.2f}\\% - {rng_max:.2f}\\%}}$\n"
        f"High Stigma Rate Pairs (≥5%): $\\bf{{{high_n:,}\\ ({high_n/nonzero_n*100.0:.2f}\\%)}}$\n"
        f"Low Stigma Rate Pairs (<1%): $\\bf{{{low_n:,}\\ ({low_n/nonzero_n*100.0:.2f}\\%)}}$"
    )
    ax.text(
        0.3,
        0.95,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor=(0.2, 0.2, 0.2), alpha=0.95),
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True, fontsize=16)
    plt.tight_layout()
    plt.savefig(out_svg, dpi=1200, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, dpi=1200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(str(out_txt))
    print(str(out_svg))
    print(str(out_pdf))


if __name__ == "__main__":
    main()
