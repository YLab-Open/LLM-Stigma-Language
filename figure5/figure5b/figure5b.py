from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
STIGMA_ROOT = Path(__file__).resolve().parents[5]
DATA_DIR = STIGMA_ROOT / "Code" / "Alice_code" / "rebuild" / "Data"
ACCURACY_DESTIGMATIZED_PATH = DATA_DIR / "CLF_CoT_destigmatized.csv"
ACCURACY_ORIGINAL_PATH = DATA_DIR / "CLF_CoT.csv"
F1_DESTIGMATIZED_PATH = DATA_DIR / "EXT_CoT_destigmatized.csv"
F1_ORIGINAL_PATH = DATA_DIR / "EXT_CoT.csv"
OUT_BASE = HERE / "figure5b"

DISPLAY_MODEL_ORDER = [
    "Mistral-Large-Instruct-2411",
    "gemma-4-31B-it",
    "Llama-3.3-70B-Instruct",
    "Qwen3-Next-80B-A3B-Instruct",
    "gpt-4o",
]

MODEL_COLORS = {
    "Mistral-Large-Instruct-2411": "#E90F44",
    "gemma-4-31B-it": "#FFC839",
    "Llama-3.3-70B-Instruct": "#60966D",
    "Qwen3-Next-80B-A3B-Instruct": "#63ADEE",
    "gpt-4o": "#6D65A3",
}

MODEL_NAME_DISPLAY = {
    "Mistral-Large-Instruct-2411": "Mistral-L",
    "gemma-4-31B-it": "Gemma-4",
    "Llama-3.3-70B-Instruct": "Llama-3.3",
    "Qwen3-Next-80B-A3B-Instruct": "Qwen-3",
    "gpt-4o": "GPT-4o",
}


def parse_metric_value(value: object) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        raise ValueError(f"Could not parse metric value from {value!r}")
    return float(match.group(0))


def canonical_model_name(model_name: str) -> str:
    return model_name.replace("-Non-Thinking", "")


def load_average_metric(path: Path, metric_name: str) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, header=[0, 1])
    average_row = df.iloc[-1]
    values = {}
    for model_name in average_row.index.get_level_values(0).unique():
        if (model_name, metric_name) not in average_row.index:
            continue
        values[canonical_model_name(model_name)] = parse_metric_value(average_row[(model_name, metric_name)])

    if not values:
        raise ValueError(f"{path} does not contain metric {metric_name!r}")
    return pd.Series(values, dtype=float)


def display_model_name(model_name: str) -> str:
    return MODEL_NAME_DISPLAY.get(canonical_model_name(model_name), canonical_model_name(model_name))


def order_models(models: list[str]) -> list[str]:
    missing = [model for model in DISPLAY_MODEL_ORDER if model not in models]
    if missing:
        raise KeyError(f"Models missing from metric CSVs: {missing}")
    return DISPLAY_MODEL_ORDER.copy()


def add_bar_labels(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        offset = 4 if height >= 0 else -12
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=18,
            fontweight="bold",
        )


def main() -> None:
    accuracy_destigmatized = load_average_metric(ACCURACY_DESTIGMATIZED_PATH, "Accuracy")
    accuracy_original = load_average_metric(ACCURACY_ORIGINAL_PATH, "Accuracy")
    f1_destigmatized = load_average_metric(F1_DESTIGMATIZED_PATH, "F1-event")
    f1_original = load_average_metric(F1_ORIGINAL_PATH, "F1-event")

    shared_models = sorted(
        set(accuracy_destigmatized.index)
        & set(accuracy_original.index)
        & set(f1_destigmatized.index)
        & set(f1_original.index)
    )
    models = order_models(shared_models)

    accuracy_delta = np.array([accuracy_destigmatized.loc[model] - accuracy_original.loc[model] for model in models])
    f1_delta = np.array([f1_destigmatized.loc[model] - f1_original.loc[model] for model in models])

    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "path" # Use path for reviewing stage
    plt.rcParams["hatch.linewidth"] = 2.0   # Default is around 1.0 pt

    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(13.5, 6))
    accuracy_bars = []
    f1_bars = []

    for i, model in enumerate(models):
        color = MODEL_COLORS[model]
        accuracy_bar = ax.bar(
            x[i] - width / 2,
            accuracy_delta[i],
            width,
            color="none",
            edgecolor=color,
            linewidth=2.0,
            hatch="///",
            label=None,
        )
        f1_bar = ax.bar(
            x[i] + width / 2,
            f1_delta[i],
            width,
            color="none",
            edgecolor=color,
            linewidth=2.0,
            hatch="\\\\\\",
            label=None,
        )
        accuracy_bars.extend(accuracy_bar)
        f1_bars.extend(f1_bar)

    add_bar_labels(ax, accuracy_bars)
    add_bar_labels(ax, f1_bars)

    legend_handles = [
        Patch(
            facecolor="none",
            edgecolor="black",
            hatch="///",
            linewidth=2.0,
            label="Classification Tasks (Metric: Accuracy)",
        ),
        Patch(
            facecolor="none",
            edgecolor="black",
            hatch="\\\\\\",
            linewidth=2.0,
            label="Extraction Tasks (Metric: F1-event)",
        ),
    ]

    y_limit_lower = max(abs(accuracy_delta).max(), abs(f1_delta).max()) * 1.2
    y_limit_upper = max(accuracy_delta.max(), f1_delta.max()) * 1.2
    ax.axhline(0, color="#666666", linewidth=1.0)
    ax.set_ylim(-y_limit_lower, y_limit_upper)
    ax.set_ylabel("Performance\nDifference (%)", fontsize=30, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([display_model_name(model) for model in models], fontsize=26)
    ax.set_yticks(np.arange(-4, 1.1, step=1.0))
    ax.tick_params(axis="y", labelsize=26)
    ax.grid(axis="y", linestyle="-", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
        ncol=1,
        fontsize=26,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(f"{OUT_BASE}.svg", dpi=1200, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
