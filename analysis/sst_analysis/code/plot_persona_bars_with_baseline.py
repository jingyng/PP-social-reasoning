import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams


THIS_DIR = Path(__file__).resolve().parent
CSV_DIR = THIS_DIR / "csv"
PLOTS_DIR = THIS_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)

IN_CSV = CSV_DIR / "persona_delta_vs_baseline_aligned_rows.csv"
OUT_PNG_ALL = PLOTS_DIR / "persona_bars_with_baseline_all_models.png"
OUT_PNG_TOKEN = PLOTS_DIR / "persona_bars_with_baseline_token.png"
OUT_PNG_IOU = PLOTS_DIR / "persona_bars_with_baseline_iou.png"

MODELS = ["gpt_oss_120b", "mistral_medium", "qwen3_32b"]
GROUPS = ["BO", "BY", "LO", "LY", "WO", "WY"]


rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def ffloat(key):
                return float(row[key]) if row.get(key) else 0.0

            row["token_f1"] = ffloat("token_f1")
            row["iou_f1"] = ffloat("iou_f1")
            row["baseline_token_f1"] = ffloat("baseline_token_f1")
            row["baseline_iou_f1"] = ffloat("baseline_iou_f1")
            rows.append(row)
    return rows


def persona_gender(persona: str) -> str:
    try:
        return persona.split("_")[1]
    except Exception:
        return "?"


def compute_group_order(data, anchor_model: str, metric_key: str):
    base_index = {g: idx for idx, g in enumerate(GROUPS)}
    scores = []
    for group in GROUPS:
        rows_g = data.get(anchor_model, {}).get(group, {})
        vals = []
        for gen in ("f", "m"):
            row = rows_g.get(gen)
            if row is None:
                continue
            try:
                vals.append(float(row[metric_key]))
            except Exception:
                continue
        avg = (sum(vals) / len(vals)) if vals else -1.0
        scores.append((group, avg))
    scores.sort(key=lambda item: (-item[1], base_index[item[0]]))
    return [g for g, _ in scores]


def ensure_span(lo: float, hi: float):
    if hi - lo < 0.08:
        mid = (hi + lo) / 2.0
        lo = mid - 0.04
        hi = mid + 0.04
    return max(0.0, lo - 0.015), min(1.0, hi + 0.015)


def draw_row(data, metric_key, baseline_key, ylabel, out_path: Path):
    gender_colors = {"f": "#4C78A8", "m": "#F58518"}
    baseline_color = "#666666"

    vals_all = []
    for model in MODELS:
        for group in GROUPS:
            rows_g = data[model][group]
            for gen in ("f", "m"):
                row = rows_g.get(gen)
                if not row:
                    continue
                vals_all.append(row[metric_key])
                vals_all.append(row[baseline_key])

    lo, hi = ensure_span(min(vals_all) if vals_all else 0.0, max(vals_all) if vals_all else 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 3.6), sharey=True)
    plt.subplots_adjust(wspace=0.12)

    groups_order = compute_group_order(data, anchor_model="gpt_oss_120b", metric_key=metric_key)

    for col, model in enumerate(MODELS):
        ax = axes[col]
        x = list(range(len(groups_order)))
        width = 0.35
        for i, group in enumerate(groups_order):
            f_row = data[model][group].get("f")
            m_row = data[model][group].get("m")
            if f_row:
                bars = ax.bar(
                    i - width / 2,
                    f_row[metric_key],
                    width=width,
                    color=gender_colors["f"],
                    label="persona (f)" if (col == 2 and i == 0) else None,
                )
                for bar in bars:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.004,
                        f"{f_row[metric_key]:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, lw=0),
                    )
            if m_row:
                bars = ax.bar(
                    i + width / 2,
                    m_row[metric_key],
                    width=width,
                    color=gender_colors["m"],
                    label="persona (m)" if (col == 2 and i == 0) else None,
                )
                for bar in bars:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.004,
                        f"{m_row[metric_key]:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, lw=0),
                    )

            base_val = None
            if f_row:
                base_val = f_row[baseline_key]
            elif m_row:
                base_val = m_row[baseline_key]
            if base_val is not None:
                ax.hlines(
                    base_val,
                    i - width * 0.75,
                    i + width * 0.75,
                    colors=baseline_color,
                    linestyles="dashed",
                    linewidth=1.2,
                    label="baseline" if (col == 2 and i == 0) else None,
                )

        ax.set_title(model)
        ax.set_ylabel(ylabel if col == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(groups_order)
        ax.set_ylim(lo, hi)
        if col == 2:
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(loc="upper left")

    fig.suptitle(f"Aligned personas: {ylabel} (bars) with dashed baselines")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing persona delta CSV: {IN_CSV}")

    rows = load_rows(IN_CSV)

    data = {m: {g: {} for g in GROUPS} for m in MODELS}
    for row in rows:
        model = row["model"]
        group = row["group"]
        if model not in data or group not in data[model]:
            continue
        gen = persona_gender(row.get("persona", ""))
        data[model][group][gen] = row

    gender_colors = {"f": "#4C78A8", "m": "#F58518"}
    baseline_color = "#666666"

    token_vals_all = []
    iou_vals_all = []
    for model in MODELS:
        for group in GROUPS:
            rows_g = data[model][group]
            for gen in ("f", "m"):
                row = rows_g.get(gen)
                if not row:
                    continue
                token_vals_all.append(row["token_f1"])
                token_vals_all.append(row["baseline_token_f1"])
                iou_vals_all.append(row["iou_f1"])
                iou_vals_all.append(row["baseline_iou_f1"])

    t_lo, t_hi = ensure_span(min(token_vals_all) if token_vals_all else 0.0, max(token_vals_all) if token_vals_all else 1.0)
    i_lo, i_hi = ensure_span(min(iou_vals_all) if iou_vals_all else 0.0, max(iou_vals_all) if iou_vals_all else 1.0)

    groups_order_token = compute_group_order(data, anchor_model="gpt_oss_120b", metric_key="token_f1")
    groups_order_iou = compute_group_order(data, anchor_model="gpt_oss_120b", metric_key="iou_f1")

    fig, axes = plt.subplots(2, 3, figsize=(18, 7), sharex=True)
    plt.subplots_adjust(hspace=0.28, wspace=0.15)

    for col, model in enumerate(MODELS):
        ax_token = axes[0, col]
        ax_iou = axes[1, col]
        x_token = list(range(len(groups_order_token)))
        x_iou = list(range(len(groups_order_iou)))
        width = 0.35

        for i, group in enumerate(groups_order_token):
            f_row = data[model][group].get("f")
            m_row = data[model][group].get("m")
            if f_row:
                bars = ax_token.bar(
                    i - width / 2,
                    f_row["token_f1"],
                    width=width,
                    color=gender_colors["f"],
                    label="persona (f)" if (col == 0 and i == 0) else None,
                )
                for bar in bars:
                    ax_token.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.004,
                        f"{f_row['token_f1']:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, lw=0),
                    )
            if m_row:
                bars = ax_token.bar(
                    i + width / 2,
                    m_row["token_f1"],
                    width=width,
                    color=gender_colors["m"],
                    label="persona (m)" if (col == 0 and i == 0) else None,
                )
                for bar in bars:
                    ax_token.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.004,
                        f"{m_row['token_f1']:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, lw=0),
                    )

            base_val = None
            if f_row:
                base_val = f_row["baseline_token_f1"]
            elif m_row:
                base_val = m_row["baseline_token_f1"]
            if base_val is not None:
                ax_token.hlines(
                    base_val,
                    i - width * 0.75,
                    i + width * 0.75,
                    colors=baseline_color,
                    linestyles="dashed",
                    linewidth=1.2,
                    label="baseline" if (col == 0 and i == 0) else None,
                )

        ax_token.set_title(f"{model} — token-F1")
        ax_token.set_ylabel("token-F1" if col == 0 else "")
        ax_token.set_xticks(x_token)
        ax_token.set_xticklabels(groups_order_token)
        ax_token.set_ylim(t_lo, t_hi)

        for i, group in enumerate(groups_order_iou):
            f_row = data[model][group].get("f")
            m_row = data[model][group].get("m")
            if f_row:
                bars = ax_iou.bar(
                    i - width / 2,
                    f_row["iou_f1"],
                    width=width,
                    color=gender_colors["f"],
                    label="persona (f)" if (col == 0 and i == 0) else None,
                )
                for bar in bars:
                    ax_iou.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.004,
                        f"{f_row['iou_f1']:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, lw=0),
                    )
            if m_row:
                bars = ax_iou.bar(
                    i + width / 2,
                    m_row["iou_f1"],
                    width=width,
                    color=gender_colors["m"],
                    label="persona (m)" if (col == 0 and i == 0) else None,
                )
                for bar in bars:
                    ax_iou.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.004,
                        f"{m_row['iou_f1']:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, lw=0),
                    )

            base_val = None
            if f_row:
                base_val = f_row["baseline_iou_f1"]
            elif m_row:
                base_val = m_row["baseline_iou_f1"]
            if base_val is not None:
                ax_iou.hlines(
                    base_val,
                    i - width * 0.75,
                    i + width * 0.75,
                    colors=baseline_color,
                    linestyles="dashed",
                    linewidth=1.2,
                    label="baseline" if (col == 0 and i == 0) else None,
                )

        ax_iou.set_title(f"{model} — IOU-F1")
        ax_iou.set_ylabel("IOU-F1" if col == 0 else "")
        ax_iou.set_xticks(x_iou)
        ax_iou.set_xticklabels(groups_order_iou)
        ax_iou.set_ylim(i_lo, i_hi)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if labels:
        axes[0, 0].legend(loc="upper left")

    fig.suptitle("Aligned personas: bar charts with dashed baselines")
    fig.savefig(OUT_PNG_ALL, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {OUT_PNG_ALL}")

    draw_row(data, metric_key="token_f1", baseline_key="baseline_token_f1", ylabel="token-F1", out_path=OUT_PNG_TOKEN)
    draw_row(data, metric_key="iou_f1", baseline_key="baseline_iou_f1", ylabel="IOU-F1", out_path=OUT_PNG_IOU)


if __name__ == "__main__":
    main()

