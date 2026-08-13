from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
IN_XLSX = BASE / "figure3a.xlsx"
OUT_SVG = BASE / "figure3a.svg"
OUT_PDF = BASE / "figure3a.pdf"


def welch_p(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def stars(p: float) -> str:
    if np.isnan(p):
        return "ns"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def add_pairwise_bracket(ax, x1, x2, y, p):
    h = (y - ax.get_ylim()[0]) * 0.02 + 0.02
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#222222", linewidth=1.2)
    ax.text((x1 + x2) / 2, y + h * 1.15, stars(p), ha="center", va="bottom", fontsize=12, fontweight="bold")


def main():
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 16

    df = pd.read_excel(IN_XLSX, sheet_name="Model_Level")

    medical = df[df["Domain"] == "Medical"]["AvgRate_pct"].dropna().to_numpy(float)
    general = df[df["Domain"] == "General"]["AvgRate_pct"].dropna().to_numpy(float)
    open_src = df[df["OpenProp"] == "Open-source"]["AvgRate_pct"].dropna().to_numpy(float)
    prop = df[df["OpenProp"] == "Proprietary"]["AvgRate_pct"].dropna().to_numpy(float)
    reasoning = df[df["Reasoning"] == "Yes"]["AvgRate_pct"].dropna().to_numpy(float)
    nonreason = df[df["Reasoning"] == "No"]["AvgRate_pct"].dropna().to_numpy(float)

    data_boxplot = [medical, general, open_src, prop, reasoning, nonreason]
    labels = [
        f"Medical Model\n(n={len(medical)})",
        f"General Model\n(n={len(general)})",
        f"Open-Source Model\n(n={len(open_src)})",
        f"Proprietary Model\n(n={len(prop)})",
        f"Reasoning Model\n(n={len(reasoning)})",
        f"Non-Reasoning Model\n(n={len(nonreason)})",
    ]
    colors = ["#fddbc7", "#f4a582", "#d9f0d3", "#a6d96a", "#d1e5f0", "#92c5de"]

    fig, ax = plt.subplots(figsize=(13, 7))
    bp = ax.boxplot(
        data_boxplot,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=False,
        widths=0.62,
        showfliers=True,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("#222222")
        patch.set_linewidth(1.2)

    for element in ["whiskers", "medians", "caps"]:
        plt.setp(bp[element], color="#222222", linewidth=1.2)

    plt.setp(
        bp["fliers"],
        marker="o",
        markersize=3.5,
        markerfacecolor="#222222",
        markeredgecolor="#222222",
        alpha=0.35,
    )
    plt.setp(
        bp["means"],
        marker="D",
        markeredgecolor="black",
        markerfacecolor="white",
        markersize=7,
        markeredgewidth=1.2,
    )

    ax.set_xlabel("Model Category", fontweight="bold")
    ax.set_ylabel("Stigma Rate (%)", fontweight="bold")
    # ax.set_title("Stigma rates by model category", fontweight="bold", pad=10)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.tick_params(axis="x", rotation=0)
    ax.vlines([2.5, 4.5], ymin=0, ymax=5, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)

    ymax = float(np.nanmax([np.nanmax(np.asarray(d, float)) for d in data_boxplot]))
    ytop = ymax * 1.18 + 0.25
    ax.set_ylim(0.0, ytop)

    p_mg = welch_p(medical, general)
    p_op = welch_p(open_src, prop)
    p_rn = welch_p(reasoning, nonreason)

    y0 = ymax * 1.05
    add_pairwise_bracket(ax, 1, 2, y0, p_mg)
    add_pairwise_bracket(ax, 3, 4, y0 + (ytop - ymax) * 0.08, p_op)
    add_pairwise_bracket(ax, 5, 6, y0 + (ytop - ymax) * 0.16, p_rn)

    plt.tight_layout()
    plt.savefig(OUT_SVG, dpi=1200, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(str(OUT_SVG))
    print(str(OUT_PDF))


if __name__ == "__main__":
    main()

