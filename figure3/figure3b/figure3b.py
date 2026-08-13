from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = REBUILD_DIR / "Data"

DISTRIBUTION_PATH = DATA_DIR / "stigma_term_distribution_by_domain.csv"
FIGURE_PATH = REBUILD_DIR / "figure3" / "figure3b" / "figure3b.svg"

TERM_COLUMN = "Stigma Term"


def plot_domain_distribution(distribution_path: Path, figure_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    plt.rcParams["svg.fonttype"] = "none"

    distribution_df = pd.read_csv(distribution_path)
    if TERM_COLUMN not in distribution_df.columns:
        raise ValueError(f"Missing required column: {TERM_COLUMN}")

    distribution_df["GENERAL"] = pd.to_numeric(
        distribution_df["GENERAL"],
        errors="coerce",
    ).fillna(0)
    distribution_df["MEDICAL"] = pd.to_numeric(
        distribution_df["MEDICAL"],
        errors="coerce",
    ).fillna(0)

    # Smoothing keeps terms with zero counts in one domain visible on log axes.
    smoothing = 0.5
    n_terms = len(distribution_df)
    general_total = distribution_df["GENERAL"].sum()
    medical_total = distribution_df["MEDICAL"].sum()

    distribution_df["general_prop"] = (
        (distribution_df["GENERAL"] + smoothing) / (general_total + smoothing * n_terms)
    )
    distribution_df["medical_prop"] = (
        (distribution_df["MEDICAL"] + smoothing) / (medical_total + smoothing * n_terms)
    )
    distribution_df["log_general_prop"] = np.log10(distribution_df["general_prop"])
    distribution_df["log_medical_prop"] = np.log10(distribution_df["medical_prop"])
    distribution_df["log2_medical_general_ratio"] = np.log2(
        distribution_df["medical_prop"] / distribution_df["general_prop"]
    )

    max_abs_ratio = max(distribution_df["log2_medical_general_ratio"].abs().max(), 1)
    # color_norm = TwoSlopeNorm(vmin=-max_abs_ratio, vcenter=0, vmax=max_abs_ratio)
    # red_gray_green = LinearSegmentedColormap.from_list(
    #     "red_gray_green",
    #     ["#b2182b", "#eeeeee", "#1a9850"],
    # )

    fig, ax = plt.subplots(figsize=(12, 9.5))
    ax.scatter(
        distribution_df["log_general_prop"],
        distribution_df["log_medical_prop"],
        c="#303030",
        # c=distribution_df["log2_medical_general_ratio"],
        # cmap=red_gray_green,
        # norm=color_norm,
        s=20,
        edgecolor="#303030",
        linewidth=0.5,
        alpha=0.95,
        zorder=3,
    )

    axis_min = min(
        distribution_df["log_general_prop"].min(),
        distribution_df["log_medical_prop"].min(),
    )
    axis_max = max(
        distribution_df["log_general_prop"].max(),
        distribution_df["log_medical_prop"].max(),
    )
    padding = 0.12 * (axis_max - axis_min)
    axis_min -= padding
    axis_max += padding

    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.2,
        zorder=1,
    )

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    # Make x and y ticks larger
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xlabel("Stigma Term Ratio of General Models (Log Scale)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Stigma Term Ratio of Medical Models (Log Scale)", fontsize=14, fontweight='bold')
    # ax.set_title("Stigma Term Distribution by Model Domain", fontsize=14, pad=14)
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.7, zorder=0)
    ax.set_aspect("equal", adjustable="box")

    # colorbar = fig.colorbar(ax.collections[0], ax=ax, shrink=0.78, pad=0.02)
    # colorbar.set_label("Log Proportion (Medical / General)", fontsize=14, fontweight='bold')

    texts = []
    for _, row in distribution_df.iterrows():
        texts.append(
            ax.text(
                row["log_general_prop"],
                row["log_medical_prop"],
                row[TERM_COLUMN],
                ha="center",
                va="center",
                fontsize=10,
                color="#202020",
                zorder=4,
            )
        )

    x_values = distribution_df["log_general_prop"].to_numpy()
    y_values = distribution_df["log_medical_prop"].to_numpy()
    adjust_text(
        texts,
        x=x_values,
        y=y_values,
        target_x=x_values,
        target_y=y_values,
        ax=ax,
        avoid_self=True,
        prevent_crossings=True,
        ensure_inside_axes=True,
        expand=(1.08, 1.25),
        force_text=(0.28, 0.46),
        force_static=(0.18, 0.32),
        force_pull=(0.01, 0.02),
        force_explode=(0.25, 0.5),
        max_move=(18, 18),
        min_arrow_len=8,
        iter_lim=1000,
        arrowprops={
            "arrowstyle": "-",
            "color": "#9e9e9e",
            "linewidth": 0.35,
            "alpha": 0.75,
        },
    )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_domain_distribution(DISTRIBUTION_PATH, FIGURE_PATH)
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
