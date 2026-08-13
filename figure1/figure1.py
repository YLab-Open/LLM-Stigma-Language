from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

REBUILD = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

# 35 task columns in current ``stigma_info_107models.xlsx`` (indices 1..35; col 0 = model name).
_TASK_COL_SLICE = slice(1, 36)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--excel",
        type=Path,
        default=REBUILD / "Data" / "stigma_info_107models.xlsx",
        help="Workbook with sheet stigma_count_matrix (default: 107 bundle).",
    )
    p.add_argument("--out-dir", type=Path, default=SCRIPT_DIR)
    p.add_argument("--basename", default="figure1")
    return p.parse_args()

def build_task_name_conversion_dict() -> dict:
    source_excel = pd.read_excel(REBUILD / "Data" / "Clinical Benchmark and LLM 107.xlsx", sheet_name="Task-all")
    result_dict = {}
    key_list = source_excel["Task-Original"].dropna().tolist()
    value_list = source_excel["Task name"].dropna().tolist()
    for key, value in zip(key_list, value_list):
        result_dict[key] = value
    return result_dict


def extract_number(name):
    if pd.isna(name):
        return np.nan
    if isinstance(name, (int, float)):
        return float(name)
    match = re.search(r"(\d+)", str(name))
    if match:
        return float(match.group(1))
    return pd.to_numeric(name, errors="coerce")


def _model_block_end_row(df: pd.DataFrame) -> int:
    """First row index after model rows (exclusive); models start at index 2."""
    i = 2
    while i < len(df):
        name = str(df.iloc[i, 0]).strip().lower()
        if name in ("average", "model_average", "model_all", ""):
            break
        i += 1
    return i


def get_model_family(model_name):
    if pd.isna(model_name) or not model_name:
        return "zzz"
    name_str = str(model_name).lower()
    if name_str.startswith("llama"):
        return "a_llama"
    if name_str.startswith("gpt"):
        return "b_gpt"
    if name_str.startswith("mistral"):
        return "c_mistral"
    if name_str.startswith("gemini"):
        return "d_gemini"
    if name_str.startswith("claude"):
        return "e_claude"
    if name_str.startswith("qwen"):
        return "f_qwen"
    if name_str.startswith("deepseek"):
        return "g_deepseek"
    if name_str.startswith("phi"):
        return "h_phi"
    if name_str.startswith("baichuan"):
        return "i_baichuan"
    if name_str.startswith("chatglm"):
        return "j_chatglm"
    if name_str.startswith("yi"):
        return "k_yi"
    if name_str.startswith("open"):
        return "l_open"
    first_word = name_str.split("-")[0].split("_")[0]
    return f"z_{first_word}"


def main() -> None:
    args = parse_args()
    file_path = args.excel
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = args.basename

    if not file_path.is_file():
        raise FileNotFoundError(f"Excel not found: {file_path}")

    print(f"Using Excel: {file_path}")
    df = pd.read_excel(file_path, sheet_name="stigma_count_matrix")

    print("Loading data from stigma_count_matrix sheet...")
    print(f"Data shape: {df.shape}")

    column_names = df.columns[_TASK_COL_SLICE]
    total_tokens = pd.Series([extract_number(name) for name in column_names], index=range(len(column_names)))

    task_name_conversion_dict = build_task_name_conversion_dict()
    task_names = df.iloc[1, _TASK_COL_SLICE].tolist()
    print(f"Extracted {len(task_names)} task names.")
    task_names_clean = []
    for name in task_names:
        if pd.isna(name):
            task_names_clean.append("")
        else:
            name = name.split('-cot')[0].strip()
            assert name in task_name_conversion_dict, f"Task name {name!r} not found in conversion dict"
            task_names_clean.append(task_name_conversion_dict[name])
    task_indices = sorted(range(len(task_names_clean)), key=lambda i: task_names_clean[i].casefold())

    model_end = _model_block_end_row(df)
    model_names = df.iloc[2:model_end, 0].tolist()
    model_names_clean = [str(name) if pd.notna(name) else "" for name in model_names]

    model_block = df.iloc[2:model_end, _TASK_COL_SLICE].apply(pd.to_numeric, errors="coerce")
    task_names_clean = [task_names_clean[i] for i in task_indices]
    # print(f"Task order after sorting alphabetically: {task_names_clean}")
    model_block = model_block.iloc[:, task_indices]
    total_tokens = total_tokens.iloc[task_indices].reset_index(drop=True)

    model_indices = list(range(len(model_names_clean)))
    model_indices.sort(key=lambda i: model_names_clean[i].casefold())
    # print(f"Model order after sorting alphabetically: {[model_names_clean[i] for i in model_indices]}")

    model_names_clean = [model_names_clean[i] for i in model_indices]
    model_block = model_block.iloc[model_indices].reset_index(drop=True)

    total_tokens_array = total_tokens.values
    stigma_rates_matrix = (model_block.values / total_tokens_array) * 100

    stigma_rates_matrix_T = stigma_rates_matrix.T

    task_averages = stigma_rates_matrix_T.mean(axis=1)
    # print(f"Calculated task averages: {task_averages}")
    stigma_rates_with_avg = np.hstack([stigma_rates_matrix_T, task_averages.reshape(-1, 1)])
    task_names_with_avg = task_names_clean + ["Task Average"]

    model_averages = stigma_rates_matrix_T.mean(axis=0)
    # print(f"Calculated model averages: {model_averages}")
    model_avg_col = np.append(model_averages, stigma_rates_matrix_T.mean())
    stigma_rates_final = np.vstack([stigma_rates_with_avg, model_avg_col.reshape(1, -1)])

    model_names_with_avg = model_names_clean + ["Model Average"]

    print(f"\nHeatmap matrix shape: {stigma_rates_final.shape}")
    print(f"Number of models: {len(model_names_clean)}")
    print(f"Number of tasks: {len(task_names_clean)}")
    print(f"Stigma rate range: {stigma_rates_matrix.min():.2f}% - {stigma_rates_matrix.max():.2f}%")

    fig, ax = plt.subplots(figsize=(24, 9))

    cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"])
    im = ax.imshow(stigma_rates_final, aspect="equal", cmap=cmap, vmin=0, vmax=35)

    ax.set_xticks(np.arange(len(model_names_with_avg)))
    ax.set_yticks(np.arange(len(task_names_with_avg)))
    ax.set_xticklabels(model_names_with_avg, rotation=90, ha="center", fontsize=10)
    ax.set_yticklabels(task_names_with_avg, fontsize=10)

    ax.set_xlabel("Model", fontsize=16, fontweight="bold", labelpad=10)
    ax.set_ylabel("Task", fontsize=16, fontweight="bold", labelpad=10)
    # ax.set_title(
    #     "Stigma Rate Distribution across 107 LLMs and 35 Tasks",
    #     fontsize=20,
    #     fontweight="bold",
    #     pad=15,
    # )

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Stigma Rate (%)", fontsize=16, fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=16)

    ax.set_xticks(np.arange(len(model_names_with_avg)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(task_names_with_avg)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.tick_params(
        axis="both",
        which="minor",
        bottom=False,
        top=False,
        left=False,
        right=False,
    )

    plt.tight_layout()
    pdf_path = out_dir / f"{out_base}.pdf"
    svg_path = out_dir / f"{out_base}.svg"
    plt.savefig(pdf_path, dpi=1200, bbox_inches="tight", facecolor="white")
    plt.savefig(svg_path, dpi=1200, bbox_inches="tight", facecolor="white")
    print(f"\nHeatmap saved:\n   {pdf_path}\n   {svg_path}")
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
