from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, MultipleLocator

REBUILD = Path(__file__).resolve().parent.parent.parent
FIG2B = REBUILD / "figure2" / "figure2b"
_spec = importlib.util.spec_from_file_location("figure2b", FIG2B / "create_figure2b.py")
if _spec is None or _spec.loader is None:
    raise ImportError(FIG2B / "create_figure2b.py")
fb = importlib.util.module_from_spec(_spec)
sys.modules["figure2b"] = fb
_spec.loader.exec_module(fb)

STIGMA_XLSX = REBUILD / "Data" / "stigma_info_107models.xlsx"
BENCHMARK_XLSX = REBUILD / "Data" / "Clinical Benchmark and LLM 107.xlsx"
OUT_DIR = REBUILD / "figure2" / "figure2cd"

HIGHLIGHT: list[tuple[str, str]] = [
    ("gpt-4o", "gpt-4o-0806"),
    ("gemini-2.5-flash", "gemini-2.5-flash"),
    ("DeepSeek-R1", "DeepSeek-R1"),
    ("gemma-4-31B-it", "gemma-4-31B-it"),
]

STYLE = [
    ("gpt-4o", {"marker": "o", "facecolors": "white", "edgecolors": "#C0392B", "linewidths": 2.1, "s": 168, "zorder": 6}),
    ("gemini-2.5-flash", {"marker": "s", "facecolors": "white", "edgecolors": "#E67E22", "linewidths": 2.1, "s": 162, "zorder": 5}),
    ("DeepSeek-R1", {"marker": "D", "facecolors": "white", "edgecolors": "#2980B9", "linewidths": 2.1, "s": 158, "zorder": 5}),
    ("gemma-4-31B-it", {"marker": "^", "facecolors": "white", "edgecolors": "#8E44AD", "linewidths": 2.1, "s": 178, "zorder": 5}),
]

# Footnote lines appended below the caption
FOOTNOTES = (
    "* Tasks with multiple labels in the source clinical document type are counted in each relevant label\n(e.g., Tasks with source document type \"Discharge Summary, Progress Note\" contribute to the stigma rate of both \"Discharge Summary\" and \"Progress Note\"). "
)


def split_category_cell(raw: object, *, split_commas: bool) -> list[str]:
    if raw is None or pd.isna(raw) or str(raw).strip() == "":
        return ["(missing)"]
    text = str(raw).strip()
    if text.lower() == "nan":
        return ["(missing)"]
    if not split_commas:
        return [text]
    cats: list[str] = []
    seen: set[str] = set()
    for part in text.split(","):
        cat = part.strip()
        if not cat or cat in seen:
            continue
        seen.add(cat)
        cats.append(cat)
    return cats or ["(missing)"]


def column_first_seen_order(df_tasks: pd.DataFrame, col: str, *, split_commas: bool = False) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for v in df_tasks[col].tolist():
        for s in split_category_cell(v, split_commas=split_commas):
            if s in seen:
                continue
            seen.add(s)
            order.append(s)
    return order


def build_cat_to_js(df_tasks: pd.DataFrame, task_headers: list, column: str) -> dict[str, list[int]]:
    by_exact, by_lower = fb.build_taxonomy_lookups(df_tasks)
    cats: dict[str, list[int]] = {}
    split_commas = column == "Source Document Type"
    for j, h in enumerate(task_headers):
        nk = fb.normalize_task_key(h)
        row = fb.resolve_row(nk, by_exact, by_lower)
        if row is None:
            continue
        for cat in split_category_cell(row.get(column), split_commas=split_commas):
            cats.setdefault(cat, []).append(j)
    bo = column_first_seen_order(df_tasks, column, split_commas=split_commas)
    extra = sorted(set(cats.keys()) - set(bo))
    if extra:
        raise ValueError(f"Categories not in Task-all order for {column}: {extra}")
    return {c: cats.get(c, []) for c in bo}


def macro_mean_across_models(frac_block: np.ndarray, js: list[int]) -> float:
    if not js:
        return 0.0
    sub = frac_block[:, js]
    per_model = np.nanmean(sub, axis=1) * 100.0
    return float(np.nanmean(per_model))


def one_model_mean_pct(vec_frac: np.ndarray, js: list[int]) -> float:
    if not js:
        return float("nan")
    return float(np.nanmean(vec_frac[js]) * 100.0)


def aggregate_dot_df(
    *,
    frac_block: np.ndarray,
    model_names: list[str],
    task_headers: list,
    df_tasks: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    cat_js = build_cat_to_js(df_tasks, task_headers, column)
    rows = []
    name_to_idx = {n: i for i, n in enumerate(model_names)}
    for mk, _ in HIGHLIGHT:
        if mk not in name_to_idx:
            raise ValueError(f"Model not in sheet: {mk}")

    for cat, js in cat_js.items():
        if not js:
            continue
        mm = macro_mean_across_models(frac_block, js)
        row = {"Category": cat, "Mean_all_models_macro_pct": mm, "Task_Count": len(js)}
        for mk, mlab in HIGHLIGHT:
            row[f"{mlab}_pct"] = one_model_mean_pct(frac_block[name_to_idx[mk], :], js)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[df["Mean_all_models_macro_pct"] > 1e-9].copy()
    df = df.sort_values("Mean_all_models_macro_pct", ascending=False).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No non-zero rows for column {column}")
    return df


def build_task_classification_detail(
    task_headers: list[str],
    df_tasks: pd.DataFrame,
) -> pd.DataFrame:
    by_exact, by_lower = fb.build_taxonomy_lookups(df_tasks)
    rows: list[dict] = []
    for j, h in enumerate(task_headers):
        nk = fb.normalize_task_key(h)
        row = fb.resolve_row(nk, by_exact, by_lower)
        base = {
            "task_index": j,
            "stigma_rate_column": h,
            "task_key_for_join": nk,
        }
        if row is None:
            rows.append(
                {
                    **base,
                    "mapped_to_Task_all": False,
                    "Task_Original": "",
                    "Task Type": "",
                    "Source Document Type": "",
                    "Clinical Application": "",
                }
            )
            continue

        def cell(col: str) -> str:
            v = row.get(col)
            if v is None or pd.isna(v):
                return ""
            return str(v).strip()

        no = cell("No")
        rows.append(
            {
                **base,
                "mapped_to_Task_all": True,
                "No": no,
                "Task_Original": cell("Task-Original"),
                "Task Type": cell("Task Type"),
                "Source Document Type": cell("Source Document Type"),
                "Clinical Application": cell("Clinical Application"),
            }
        )
    return pd.DataFrame(rows)


def _value_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_pct") or c == "Mean_all_models_macro_pct"]


def _df_xmax(df: pd.DataFrame) -> float:
    return float(np.nanmax(df[_value_columns(df)].astype(float).values))


def xlim_task_type_panel(df_c: pd.DataFrame) -> tuple[float, float]:
    mx = _df_xmax(df_c)
    if mx <= 3.0 + 1e-12:
        return (0.0, 3.0)
    return (0.0, mx * 1.14 + 0.12)


def xlim_document_panel(df_d: pd.DataFrame) -> tuple[float, float]:
    mx = _df_xmax(df_d)
    return (0.0, mx * 1.12 + 0.15)


def legend_handles() -> list[mlines.Line2D]:
    handles = [
        mlines.Line2D(
            [],
            [],
            color="#1A5276",
            marker="o",
            linestyle="None",
            markersize=12,
            markeredgecolor="black",
            label="Mean (all models)",
        ),
    ]
    handles += [
        mlines.Line2D(
            [],
            [],
            color=STYLE[j][1]["edgecolors"],
            marker=STYLE[j][1]["marker"],
            linestyle="None",
            markersize=12,
            markerfacecolor="white",
            markeredgewidth=2,
            label=HIGHLIGHT[j][1],
        )
        for j in range(len(HIGHLIGHT))
    ]
    return handles


def draw_dot_on_ax(
    ax,
    df: pd.DataFrame,
    *,
    panel_label: str,
    title: str = "",
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float],
    ytick_fontsize: int = 16,
) -> None:
    dfp = df.iloc[::-1].reset_index(drop=True)
    n = len(dfp)
    for i, r in dfp.iterrows():
        y = float(i)
        mm = r["Mean_all_models_macro_pct"]
        ax.plot([0, mm], [y, y], color="#5DADE2", linewidth=2.0, alpha=0.88, zorder=2)
        ax.scatter(
            mm,
            y,
            s=188,
            c="#1A5276",
            edgecolors="black",
            linewidths=1.15,
            marker="o",
            zorder=4,
        )
        for key, sty in STYLE:
            colname = next(lab for k, lab in HIGHLIGHT if k == key) + "_pct"
            val = r[colname]
            if np.isfinite(val):
                kw = {k: v for k, v in sty.items()}
                ax.scatter(val, y, **kw)

    yticks = np.arange(n)
    ax.set_yticks(yticks)
    ax.set_yticklabels(dfp["Category"].tolist(), fontsize=ytick_fontsize)
    ax.set_xlabel(xlabel, fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    ax.set_xlim(*xlim)
    ax.tick_params(axis="both", labelsize=16, width=0.9, length=4)
    # if xlim[1] <= 3.001:
    #     ax.set_xticks(np.arange(0.0, 3.51, 0.5))
    #     ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    # else:
    #     ax.xaxis.set_major_locator(MaxNLocator(7, prune=None))
    ax.grid(True, axis="x", linestyle="-", linewidth=0.65, alpha=0.38)
    ax.grid(True, axis="x", which="minor", linestyle=":", linewidth=0.35, alpha=0.22)
    ax.set_axisbelow(True)
    ts = str(title).strip()
    if ts:
        ax.set_title(ts, fontsize=16, fontweight="bold", pad=10)
    # ax.text(
    #     -0.4 if panel_label == "(D)" else -0.28,
    #     1.02,
    #     panel_label,
    #     transform=ax.transAxes,
    #     fontsize=16,
    #     fontweight="bold",
    #     va="bottom",
    # )


def save_combined_cd_figure(
    *,
    out_svg: Path,
    out_pdf: Path,
    df_c: pd.DataFrame,
    df_d: pd.DataFrame,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
    plt.rcParams["axes.unicode_minus"] = False
    n_max = max(len(df_c), len(df_d))
    fig_h = max(5.6, 0.55 * n_max + 1.65)
    fig_w = 21.5
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=False)

    xlim_c = xlim_task_type_panel(df_c)
    xlim_d = xlim_document_panel(df_d)

    draw_dot_on_ax(
        axes[0],
        df_c,
        panel_label="(C)",
        xlabel="Stigma Rate (%)",
        ylabel="NLP Task Type",
        xlim=xlim_c,
        ytick_fontsize=16,
    )
    draw_dot_on_ax(
        axes[1],
        df_d,
        panel_label="(D)",
        xlabel="Stigma Rate (%)",
        ylabel="Source Clinical Document Type",
        xlim=xlim_d,
        ytick_fontsize=16,
    )

    fig.subplots_adjust(wspace=0.5)
    handles = legend_handles()
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=5,
        fontsize=16,
        frameon=True,
        columnspacing=1.1,
    )
    fig.text(0.05, -0.15, FOOTNOTES,
             ha="left", va="center", fontsize=16, color="#666666",
             style="italic")
    fig.savefig(out_svg, dpi=1200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, dpi=1200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def overall_nonzero_stats(frac_block: np.ndarray) -> tuple[float, float]:
    pct = frac_block.reshape(-1) * 100.0
    nz = pct[np.isfinite(pct) & (pct > 0)]
    if len(nz) == 0:
        return float("nan"), float("nan")
    return float(np.mean(nz)), float(np.median(nz))


def assert_highlight_models(model_names: list[str]) -> None:
    names = set(model_names)
    need = [m for m, _ in HIGHLIGHT]
    missing = [m for m in need if m not in names]
    if missing:
        raise ValueError(
            "stigma rate sheet missing model row(s): "
            + ", ".join(repr(m) for m in missing)
            + "; required: "
            + ", ".join(repr(m) for m in need)
        )


def main() -> None:
    task_headers, frac_block, model_names = fb.load_fraction_matrix(STIGMA_XLSX)
    assert_highlight_models(model_names)
    df_tasks = pd.read_excel(BENCHMARK_XLSX, sheet_name="Task-all")

    mean_nz, med_nz = overall_nonzero_stats(frac_block)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_tt = aggregate_dot_df(
        frac_block=frac_block,
        model_names=model_names,
        task_headers=task_headers,
        df_tasks=df_tasks,
        column="Task Type",
    )
    df_sd = aggregate_dot_df(
        frac_block=frac_block,
        model_names=model_names,
        task_headers=task_headers,
        df_tasks=df_tasks,
        column="Source Document Type",
    )

    save_combined_cd_figure(
        out_svg=OUT_DIR / "figure2cd.svg",
        out_pdf=OUT_DIR / "figure2cd.pdf",
        df_c=df_tt,
        df_d=df_sd,
    )

    df_tasks_detail = build_task_classification_detail(task_headers, df_tasks)
    xlsx = OUT_DIR / "figure2cd_lollipop_data.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        df_tt.to_excel(w, sheet_name="Task_Type_summary", index=False)
        df_sd.to_excel(w, sheet_name="Source_Document_summary", index=False)
        df_tasks_detail.to_excel(w, sheet_name="Task_classification", index=False)
        meta_rows = [
            {"item": "stigma_xlsx", "detail": str(STIGMA_XLSX.resolve())},
            {"item": "benchmark_xlsx", "detail": str(BENCHMARK_XLSX.resolve())},
            {"item": "nonzero_pair_mean_pct", "detail": mean_nz},
            {"item": "nonzero_pair_median_pct", "detail": med_nz},
        ]
        for sheet_row, legend_label in HIGHLIGHT:
            meta_rows.append(
                {"item": "plot_uses_rate_row", "detail": f"label={legend_label} sheet_row={sheet_row}"}
            )
        pd.DataFrame(meta_rows).to_excel(w, sheet_name="Run_meta", index=False)

    print(OUT_DIR / "figure2cd.svg")
    print(xlsx)


if __name__ == "__main__":
    main()
