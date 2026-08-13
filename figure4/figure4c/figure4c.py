from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REBUILD = Path(__file__).resolve().parent.parent.parent
ACCURACY_DIR = Path(__file__).resolve().parent


def ols_line_and_mean_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ngrid: int = 200,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """
    Returns:
        x_line, y_hat, y_lo, y_hi, beta0, beta1, r_pearson, s_squared
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = int(x.size)
    if n < 3:
        raise ValueError("Need at least 3 finite points for OLS band.")
    x_bar = float(np.mean(x))
    y_bar = float(np.mean(y))
    Sxx = float(np.sum((x - x_bar) ** 2))
    if Sxx <= 0:
        raise ValueError("x has zero variance.")
    beta1 = float(np.sum((x - x_bar) * (y - y_bar)) / Sxx)
    beta0 = float(y_bar - beta1 * x_bar)
    y_fit = beta0 + beta1 * x
    resid = y - y_fit
    dof = n - 2
    sse = float(np.sum(resid**2))
    s2 = sse / dof if dof > 0 else np.nan
    s = float(np.sqrt(s2))
    r_num = np.sum((x - x_bar) * (y - y_bar))
    r_den = np.sqrt(Sxx * np.sum((y - y_bar) ** 2))
    r_pearson = float(r_num / r_den) if r_den > 0 else float("nan")

    t_crit = float(stats.t.ppf(1 - alpha / 2, dof))
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), ngrid)
    y_hat = beta0 + beta1 * x_line
    se_mean = s * np.sqrt(1.0 / n + (x_line - x_bar) ** 2 / Sxx)
    margin = t_crit * se_mean
    y_lo = y_hat - margin
    y_hi = y_hat + margin
    return x_line, y_hat, y_lo, y_hi, beta0, beta1, r_pearson, s2


def normalize_task_key(header: str) -> str:
    t = str(header).strip()
    if t.endswith(".result.json"):
        t = t[: -len(".result.json")]
    t = t.split("-cot-")[0]
    return t


def load_stigma_rate_matrix(stigma_xlsx: Path) -> tuple[list[str], np.ndarray, list[str]]:
    rate = pd.read_excel(stigma_xlsx, sheet_name="rate", header=None)
    task_headers = [str(x).strip() for x in rate.iloc[1, 1:-1].tolist()]
    n_task = len(task_headers)
    rows: list[np.ndarray] = []
    names: list[str] = []
    for i in range(2, len(rate)):
        name = str(rate.iloc[i, 0]).strip()
        if not name or name.lower() == "nan" or name in ("Average", "Input"):
            continue
        vals = pd.to_numeric(rate.iloc[i, 1 : 1 + n_task], errors="coerce").to_numpy(dtype=float)
        if vals.shape[0] != n_task:
            raise ValueError(f"Row {name}: expected {n_task} task columns, got {vals.shape[0]}")
        rows.append(vals)
        names.append(name)
    if not rows:
        raise ValueError(f"No model rows in {stigma_xlsx}")
    return task_headers, np.vstack(rows), names


def block_start_for_model(df: pd.DataFrame, model: str) -> int | None:
    for j in range(8, df.shape[1], 4):
        if j + 3 >= df.shape[1]:
            break
        if str(df.iloc[0, j]).strip() == model:
            return j
    return None


def accuracy_from_perf_cell(val) -> float:
    """Performance cells are often point + CI, e.g. '76.39 [76.33, 76.44]'; take the leading number."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return np.nan
    for sep in (" ", "\t", "[", "("):
        if sep in s:
            head = s.split(sep, 1)[0].strip()
            if head:
                s = head
            break
    x = pd.to_numeric(s, errors="coerce")
    return float(x) if pd.notna(x) else np.nan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--perf", type=Path, default=REBUILD / "Data" / "performance_107_stigma_models_only.xlsx")
    p.add_argument("--stigma", type=Path, default=REBUILD / "Data" / "stigma_info_107models.xlsx")
    p.add_argument("--out-dir", type=Path, default=ACCURACY_DIR)
    p.add_argument("--sheet", type=str, default="Sheet1")
    args = p.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.perf, sheet_name=args.sheet, header=None)
    if df.shape[1] < 9:
        raise ValueError("Expected task metadata in columns 0-7 and model blocks from column 8.")

    task_headers, stigma_mat, stigma_models = load_stigma_rate_matrix(args.stigma)
    norm_to_task_idx: dict[str, int] = {}
    for idx, h in enumerate(task_headers):
        nk = normalize_task_key(h)
        norm_to_task_idx.setdefault(nk, idx)

    missing_blocks = [m for m in stigma_models if block_start_for_model(df, m) is None]
    if missing_blocks:
        raise ValueError("Performance sheet missing model blocks for: " + ", ".join(missing_blocks[:20]))

    records: list[dict] = []
    tc_rows = [
        r
        for r in range(2, len(df))
        if str(df.iloc[r, 0]).strip() == "Text Classification"
    ]
    unmatched: list[str] = []
    for r in tc_rows:
        raw_task = str(df.iloc[r, 7]).strip()
        key = normalize_task_key(raw_task)
        ti = norm_to_task_idx.get(key)
        if ti is None:
            unmatched.append(raw_task)
            continue
        for mi, model in enumerate(stigma_models):
            j0 = block_start_for_model(df, model)
            assert j0 is not None
            acc = accuracy_from_perf_cell(df.iloc[r, j0])
            stigma_rate = float(stigma_mat[mi, ti])
            records.append(
                {
                    "model": model,
                    "task_type": "Text Classification",
                    "task_classification": raw_task,
                    "accuracy": acc,
                    "stigma_rate": stigma_rate,
                    "stigma_pct": stigma_rate * 100.0,
                }
            )
    if unmatched:
        raise ValueError("No stigma column for task(s): " + "; ".join(unmatched))

    long_df = pd.DataFrame.from_records(records)
    csv_path = out_dir / "accuracy_stigma_long.csv"
    xlsx_path = out_dir / "accuracy_stigma_long.xlsx"
    long_df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        long_df.to_excel(w, sheet_name="long", index=False)
        pd.DataFrame(
            [
                {
                    "perf_xlsx": str(args.perf.resolve()),
                    "stigma_xlsx": str(args.stigma.resolve()),
                    "n_models": len(stigma_models),
                    "n_tc_tasks": len(tc_rows),
                    "n_rows_long": len(long_df),
                }
            ]
        ).to_excel(w, sheet_name="meta", index=False)

    try:
        import matplotlib.pyplot as plt
        from scipy.stats import pearsonr

        x = long_df["accuracy"].to_numpy(dtype=float)
        y = long_df["stigma_pct"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        r, p = pearsonr(x[m], y[m]) if m.sum() >= 2 else (float("nan"), float("nan"))
        p_txt = "p < 0.001" if p < 0.001 else f"p = {p:.4g}"

        xv, yv = x[m], y[m]
        x_line, y_hat, y_lo, y_hi, _, _, _, _ = ols_line_and_mean_ci(xv, yv)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(xv, yv, alpha=0.35, s=12, edgecolors="none", zorder=2)
        ax.fill_between(x_line, y_lo, y_hi, color="#e41a1c", alpha=0.22, zorder=1, linewidth=0)
        ax.plot(x_line, y_hat, color="#e41a1c", linewidth=2.2, zorder=3)
        ax.set_xlabel("Model Accuracy (%)", fontsize=16, fontweight="bold")
        ax.set_ylabel("Stigma Rate (%)", fontsize=16, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.03,
            0.97,
            f"Pearson r = $\\bf{{{r:.3f}}}$\n{p_txt}",
            transform=ax.transAxes,
            fontsize=16,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.72", alpha=0.95),
        )
        fig.tight_layout()
        fig.savefig(out_dir / "figure4c.svg", dpi=1200)
        plt.close(fig)
    except Exception as e:
        print("Skipping plot:", e)

    print(f"Wrote {csv_path}")
    print(f"Wrote {xlsx_path}")
    print(f"Rows: {len(long_df)} (= {len(stigma_models)} models × {len(tc_rows)} tasks)")


if __name__ == "__main__":
    main()
