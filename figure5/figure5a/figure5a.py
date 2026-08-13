from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
STIGMA_ROOT = Path(__file__).resolve().parents[5]
DATA_DIR = STIGMA_ROOT / "Code" / "Alice_code" / "rebuild" / "Data"
BEFORE_PATH = DATA_DIR / "output_stigma_rate.csv"
AFTER_PATH = DATA_DIR / "output_stigma_rate_destigmatized.csv"
OUT_BASE = HERE / "figure5a"
DISPLAY_MODEL_ORDER = [
    "Mistral-Large-Instruct-2411",
    "gemma-4-31B-it-Non-Thinking",
    "Llama-3.3-70B-Instruct",
    "Qwen3-Next-80B-A3B-Instruct-Non-Thinking",
    "gpt-4o",
]

MODEL_NAME_DISPLAY = {
    "Mistral-Large-Instruct-2411": "Mistral-L",
    "gemma-4-31B-it-Non-Thinking": "Gemma-4",
    "Llama-3.3-70B-Instruct": "Llama-3.3",
    "Qwen3-Next-80B-A3B-Instruct-Non-Thinking": "Qwen-3",
    "gpt-4o": "GPT-4o",
}


def load_task_average(path: Path) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    needed = {"Model", "task_average"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df.loc[df["Model"] != "model_average", ["Model", "task_average"]].copy()
    df["task_average"] = pd.to_numeric(df["task_average"], errors="raise")
    return df.set_index("Model")["task_average"]


def lookup_before_rate(model_name: str, before_rates: pd.Series) -> float:
    if model_name in before_rates.index:
        return float(before_rates.loc[model_name])

    suffixes = ["-Non-Thinking", "-Thinking"]
    for suffix in suffixes:
        if model_name.endswith(suffix):
            base_name = model_name[: -len(suffix)]
            if base_name in before_rates.index:
                return float(before_rates.loc[base_name])

    raise KeyError(f"Could not find baseline stigma rate for {model_name}")


def display_model_name(model_name: str) -> str:
    return model_name.replace("-Non-Thinking", "")


def order_models(models: list[str]) -> list[str]:
    missing = [model for model in DISPLAY_MODEL_ORDER if model not in models]
    if missing:
        raise KeyError(f"Models missing from destigmatized CSV: {missing}")
    return DISPLAY_MODEL_ORDER.copy()


def add_bar_labels(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight="bold",
        )


def main() -> None:
    before_rates = load_task_average(BEFORE_PATH)
    after_rates = load_task_average(AFTER_PATH)

    models = order_models(list(after_rates.index))
    before_pct = np.array([lookup_before_rate(model, before_rates) * 100 for model in models])
    after_pct = after_rates.loc[models].to_numpy(dtype=float) * 100

    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "path" # Use path for reviewing stage
    plt.rcParams["hatch.linewidth"] = 2.0   # Default is around 1.0 pt

    color_pairs = [
        ("#E90F44", "#fddbc7"),
        ("#FFC839", "#F9F9CA"),
        ("#60966D", "#d9f0d3"),
        ("#63ADEE", "#d1e5f0"),
        ("#6D65A3", "#E6CDFA"),
    ]

    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(13.5, 6))
    before_bars = []
    after_bars = []

    for i, model in enumerate(models):
        dark, light = color_pairs[i % len(color_pairs)]
        before_bar = ax.bar(
            x[i] - width / 2,
            before_pct[i],
            width,
            color="none",
            edgecolor=dark,
            linewidth=2.0,
            hatch="///",
            label=None,
        )
        after_bar = ax.bar(
            x[i] + width / 2,
            after_pct[i],
            width,
            color="none",
            edgecolor=dark,
            linewidth=2.0,
            hatch="\\\\\\",
            label=None,
        )
        before_bars.extend(before_bar)
        after_bars.extend(after_bar)

    add_bar_labels(ax, before_bars)
    add_bar_labels(ax, after_bars)

    legend_handles = [
        Patch(
            facecolor="none",
            edgecolor="black",
            hatch="///",
            linewidth=2.0,
            label="Before destigmatizing",
        ),
        Patch(
            facecolor="none",
            edgecolor="black",
            hatch="\\\\\\",
            linewidth=2.0,
            label="After destigmatizing",
        ),
    ]

    y_max = max(before_pct.max(), after_pct.max())
    ax.set_ylim(0, y_max * 1.22)
    ax.set_ylabel("Stigma Rate (%)", fontsize=30, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([display_model_name(MODEL_NAME_DISPLAY.get(model, model)) for model in models], fontsize=26)
    ax.set_yticks(np.arange(0, 3.1, step=0.5))
    ax.tick_params(axis="y", labelsize=26)
    ax.grid(axis="y", linestyle="-", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        fontsize=26,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(f"{OUT_BASE}.svg", dpi=1200, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
