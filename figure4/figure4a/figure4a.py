import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
import math

REBUILD = Path(__file__).resolve().parent.parent.parent

def main() -> None:
    stigma_rate_df = pd.read_csv(os.path.join(REBUILD, "Data", "output_stigma_rate.csv"))
    stigma_rate_dict = {}
    for model_name in stigma_rate_df["Model"]:
        if model_name == "model_average":
            continue
        stigma_rate = float(stigma_rate_df[stigma_rate_df["Model"] == model_name]["task_average"].values[0])
        stigma_rate_dict[model_name] = stigma_rate

    deepseek_family = [
        "DeepSeek-R1-Distill-Qwen-1.5B",
        "DeepSeek-R1-Distill-Qwen-7B",
        "DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-Qwen-14B",
        "DeepSeek-R1-Distill-Qwen-32B",
        "DeepSeek-R1-Distill-Llama-70B",
        "DeepSeek-R1" # 671B
    ]

    llama_family = [
        'Llama-3.2-1B-Instruct',
        'Llama-3.2-3B-Instruct',
        'Llama-3.1-8B-Instruct',
        # 'Llama-3.1-70B-Instruct',
        'Llama-3.3-70B-Instruct',
        # 'Llama-4-Scout-17B-16E-Instruct'
    ]

    qwen_family = [
        'Qwen2.5-1.5B-Instruct',
        'Qwen2.5-3B-Instruct',
        'Qwen2.5-7B-Instruct',
        'Qwen2.5-72B-Instruct'
    ]

    mellama_family = [
        'MeLLaMA-13B-chat',
        'MeLLaMA-70B-chat'
    ]

    # baichuan_family = [
    #     'Baichuan-M1-14B-Instruct',
    #     "Baichuan-M2-32B"
    # ]

    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
    plt.rcParams["axes.unicode_minus"] = False

    # Create figure with 1 row and 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(30, 6), sharey=False, gridspec_kw={'wspace': 0.25, 'hspace': 0.35})
    axes = axes.flatten()

    color_list = [
        '#E42320',
        "#E4CE51",
        '#57AF37',
        '#6A8EC9',
    ]

    # Function to get model size from name
    def get_model_size(model_name):
        if "0.6B" in model_name:
            return 0.6
        elif "1.5B" in model_name:
            return 1.5
        elif "1.7B" in model_name:
            return 1.7
        elif "1B" in model_name:
            return 1
        elif "Qwen3-4B" in model_name:
            return 4
        elif "13B" in model_name:
            return 13
        elif "3B" in model_name:
            return 3
        elif "17B-16E" in model_name:
            return 109
        elif "7B" in model_name:
            return 7
        elif "8B" in model_name:
            return 8
        elif "14B" in model_name:
            return 14
        elif "24B" in model_name:
            return 24
        elif "32B" in model_name:
            return 32
        elif "70B" in model_name:
            return 70
        elif "72B" in model_name:
            return 72
        elif "DeepSeek-R1" == model_name:
            return 671
        else:
            return 0

    # Plot families
    families = [deepseek_family, llama_family, qwen_family, mellama_family]
    family_names = ["DeepSeek Family", "Llama Family", "Qwen Family", "MeLLaMA Family"]

    for i, (family, name) in enumerate(zip(families, family_names)):
        ax = axes[i]
        
        # Sort models by size
        family_sizes = [get_model_size(model) for model in family]
        family_ordered = [model for _, model in sorted(zip(family_sizes, family))]
        sizes_ordered = np.arange(5, 5 + 5 * len(family_ordered), 5)
        x_span = max(sizes_ordered) - min(sizes_ordered)
        sizes_ordered_display = [get_model_size(model) for model in family_ordered]
        
        min_score = 100
        max_score = 0
        # Plot for each prompt mode
        scores = [stigma_rate_dict[model] * 100 for model in family_ordered]
        min_score = min(min_score, min(scores))
        max_score = max(max_score, max(scores))
        ax.plot(sizes_ordered, scores, 'o-', ms=10, color=color_list[i])
        if name == "MeLLaMA Family":
            y_min = math.floor(min_score * 100) / 100
            y_max = math.ceil(max_score * 100) / 100
        else:
            y_min = math.floor(min_score * 10) / 10
            y_max = math.ceil(max_score * 10) / 10
        y_span = y_max - y_min
        # Set title and labels
        ax.set_title(name, fontsize=24)
        if i == 0:
            ax.set_ylabel("Stigma Rate (%)", fontsize=24, fontweight='bold')
        
        # Set x-ticks to show exact model sizes
        ax.set_xticks(sizes_ordered)
        ax.set_xticklabels([str(size) for size in sizes_ordered_display], rotation=0)
        if name == "MeLLaMA Family":
            ax.set_yticks(np.arange(y_min, y_max + 0.001, 0.01))
        else:
            ax.set_yticks(np.arange(y_min, y_max + 0.01, 0.1))
        # Set font size for ticks
        ax.tick_params(axis='both', labelsize=20)
        ax.set_xlim(min(sizes_ordered) - 0.2 * x_span, max(sizes_ordered) + 0.2 * x_span)
        ax.set_ylim(y_min - 0.1 * y_span, y_max + 0.1 * y_span)
        
        # Add grid
        # ax.grid(True, linestyle='--', alpha=0.7)

    # Add legend to the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=24, frameon=False)

    # plt.tight_layout()
    fig.text(0.5, -0.02, "Model Size (B)", ha='center', fontsize=24, fontweight='bold')
    plt.savefig("figure4a.svg", bbox_inches='tight', format='svg', dpi=1200, transparent=True)
    # plt.show()

if __name__ == "__main__":
    main()