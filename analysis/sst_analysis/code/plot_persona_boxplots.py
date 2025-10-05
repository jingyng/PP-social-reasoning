import os
import csv

# Ensure writable caches before importing matplotlib
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_mpl_cache = os.path.join(REPO_ROOT, "3_analysis", "3_4_cose_analysis", ".mplcache")
_xdg_cache = os.path.join(REPO_ROOT, ".cache")
os.makedirs(_mpl_cache, exist_ok=True)
os.makedirs(os.path.join(_xdg_cache, "fontconfig"), exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)
os.environ.setdefault("XDG_CACHE_HOME", _xdg_cache)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import PercentFormatter

rcParams.update({
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "boxplot.flierprops.markersize": 2.5,
})

PLOTS_DIR = os.path.join(REPO_ROOT, "3_analysis", "3_4_cose_analysis", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CSV_DIR = os.path.join(REPO_ROOT, "3_analysis", "3_4_cose_analysis", "csv")
ACC_CSV = os.path.join(CSV_DIR, "persona_accuracy_summary.csv")
RAT_CSV = os.path.join(CSV_DIR, "persona_rationale_agreement_summary.csv")
BASE_ACC_CSV = os.path.join(CSV_DIR, "baseline_accuracy_summary.csv")
BASE_RAT_CSV = os.path.join(CSV_DIR, "baseline_rationale_agreement_summary.csv")

MODELS = ["gpt_oss_120b", "mistral_medium", "qwen3_32b"]
GROUPS = ["BO", "BY", "LO", "LY", "WO", "WY"]


def load_accuracy(path):
    # model -> group -> list of accuracies across all personas
    acc = {m: {g: [] for g in GROUPS} for m in MODELS}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            m = row["model"].strip()
            g = row["group"].strip()
            if m in acc and g in acc[m]:
                try:
                    acc[m][g].append(float(row["accuracy"]))
                except Exception:
                    pass
    return acc


def load_rationale(path, key):
    # model -> group -> list of metric values across all personas
    vals = {m: {g: [] for g in GROUPS} for m in MODELS}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            m = row["model"].strip()
            g = row["group"].strip()
            if m in vals and g in vals[m]:
                try:
                    vals[m][g].append(float(row[key]))
                except Exception:
                    pass
    return vals


def compute_group_order(data_by_model, anchor_model: str = "gpt_oss_120b"):
    # Order groups by descending mean value for the anchor model
    order = []
    scores = []
    idx = {g: i for i, g in enumerate(GROUPS)}
    for g in GROUPS:
        vals = data_by_model.get(anchor_model, {}).get(g, [])
        mean_v = sum(vals) / len(vals) if vals else -1.0
        scores.append((g, mean_v))
    scores.sort(key=lambda x: (-x[1], idx[x[0]]))
    return [g for g, _ in scores]


def load_baseline_map_for_accuracy(path):
    # model -> group -> value
    base = {m: {} for m in MODELS}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            m = row["model"].strip()
            g = row["group"].strip()
            if g == "OVERALL" or m not in base:
                continue
            try:
                base[m][g] = float(row["accuracy"]) if row.get("accuracy") else None
            except Exception:
                base[m][g] = None
    return base


def load_baseline_map_for_rationale(path, key):
    base = {m: {} for m in MODELS}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            m = row["model"].strip()
            g = row["group"].strip()
            if m not in base:
                continue
            try:
                base[m][g] = float(row[key]) if row.get(key) else None
            except Exception:
                base[m][g] = None
    return base


def draw_boxplots(data_by_model, ylabel, out_png, percent=False, baseline_map=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 3.6), sharey=True)
    plt.subplots_adjust(wspace=0.12)

    # Determine y-limits across models
    all_vals = []
    for m in MODELS:
        for g in GROUPS:
            all_vals.extend(data_by_model[m][g])
    lo = min(all_vals) if all_vals else 0.0
    hi = max(all_vals) if all_vals else 1.0
    if hi - lo < 0.08:
        mid = (hi + lo) / 2.0
        lo, hi = mid - 0.04, mid + 0.04
    lo = max(0.0, lo - 0.015)
    hi = min(1.0, hi + 0.015)

    # Compute shared group order (based on GPT-OSS-120B mean)
    groups_order = compute_group_order(data_by_model)

    for col, m in enumerate(MODELS):
        ax = axes[col]
        series = [data_by_model[m][g] for g in groups_order]
        ax.boxplot(
            series,
            tick_labels=groups_order,
            showmeans=True,
            meanline=False,
            patch_artist=True,
            boxprops=dict(facecolor="#D9E3F0", edgecolor="#336699", linewidth=1.0),
            medianprops=dict(color="#1F3A5B", linewidth=1.2),
            whiskerprops=dict(color="#336699", linewidth=1.0),
            capprops=dict(color="#336699", linewidth=1.0),
            flierprops=dict(markerfacecolor="#999999", markeredgecolor="#999999"),
            meanprops=dict(marker="o", markerfacecolor="#F58518", markeredgecolor="#F58518", markersize=3),
        )
        ax.set_title(m)
        ax.set_ylabel(ylabel if col == 0 else "")
        ax.set_ylim(lo, hi)
        # Overlay baselines as dashed lines per group
        if baseline_map and m in baseline_map:
            for i, g in enumerate(groups_order):
                bval = baseline_map[m].get(g)
                if bval is not None:
                    ax.hlines(bval, i + 0.65, i + 1.35, colors="#666666", linestyles="dashed", linewidth=1.0)
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))

    fig.suptitle(f"Distribution across personas — {ylabel}")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {out_png}")


def draw_grouped_token_boxplots(token_by_model):
    """Single figure: for each group, draw 3 side-by-side boxplots (one per model)."""
    # Gather series per group and per model
    model_order = ["gpt_oss_120b", "mistral_medium", "qwen3_32b"]
    colors = {
        "gpt_oss_120b": "#4C78A8",
        "mistral_medium": "#F58518",
        "qwen3_32b": "#54A24B",
    }

    fig, ax = plt.subplots(figsize=(12, 4.2))

    # positions: group index centered at integers; model offsets around
    x_centers = list(range(len(GROUPS)))
    offset = 0.25
    positions_map = {
        "gpt_oss_120b": [-offset + xc for xc in x_centers],
        "mistral_medium": [0.0 + xc for xc in x_centers],
        "qwen3_32b": [offset + xc for xc in x_centers],
    }

    # Plot each model's boxplots
    for model in model_order:
        series = [token_by_model[model][g] for g in GROUPS]
        bp = ax.boxplot(
            series,
            positions=positions_map[model],
            widths=0.22,
            patch_artist=True,
            showmeans=True,
            meanline=False,
            boxprops=dict(facecolor=colors[model], edgecolor="#333333", linewidth=1.0, alpha=0.6),
            medianprops=dict(color="#111111", linewidth=1.2),
            whiskerprops=dict(color="#333333", linewidth=1.0),
            capprops=dict(color="#333333", linewidth=1.0),
            flierprops=dict(markerfacecolor="#999999", markeredgecolor="#999999", markersize=2.5),
            meanprops=dict(marker="o", markerfacecolor=colors[model], markeredgecolor=colors[model], markersize=3),
        )

    # X ticks at group centers
    ax.set_xticks(x_centers)
    ax.set_xticklabels(GROUPS)
    ax.set_ylabel("token-F1")
    ax.set_title("Distribution across personas — token-F1 (grouped by model)")

    # Y limits tight across all data
    all_vals = []
    for m in model_order:
        for g in GROUPS:
            all_vals.extend(token_by_model[m][g])
    lo = min(all_vals) if all_vals else 0.0
    hi = max(all_vals) if all_vals else 1.0
    if hi - lo < 0.08:
        mid = (hi + lo) / 2.0
        lo, hi = mid - 0.04, mid + 0.04
    ax.set_ylim(max(0.0, lo - 0.015), min(1.0, hi + 0.015))

    # Legend
    handles = [plt.Line2D([0], [0], color='w', marker='s', markerfacecolor=colors[m], markersize=8,
                           label=m) for m in model_order]
    ax.legend(handles=handles, title="model", loc="upper left")

    out_png = os.path.join(PLOTS_DIR, "persona_tokenf1_boxplots_grouped.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {out_png}")


def draw_grouped_accuracy_boxplots(acc_by_model):
    model_order = ["gpt_oss_120b", "mistral_medium", "qwen3_32b"]
    colors = {
        "gpt_oss_120b": "#4C78A8",
        "mistral_medium": "#F58518",
        "qwen3_32b": "#54A24B",
    }

    fig, ax = plt.subplots(figsize=(12, 4.2))

    x_centers = list(range(len(GROUPS)))
    offset = 0.25
    positions_map = {
        "gpt_oss_120b": [-offset + xc for xc in x_centers],
        "mistral_medium": [0.0 + xc for xc in x_centers],
        "qwen3_32b": [offset + xc for xc in x_centers],
    }

    for model in model_order:
        series = [acc_by_model[model][g] for g in GROUPS]
        bp = ax.boxplot(
            series,
            positions=positions_map[model],
            widths=0.22,
            patch_artist=True,
            showmeans=True,
            meanline=False,
            boxprops=dict(facecolor=colors[model], edgecolor="#333333", linewidth=1.0, alpha=0.6),
            medianprops=dict(color="#111111", linewidth=1.2),
            whiskerprops=dict(color="#333333", linewidth=1.0),
            capprops=dict(color="#333333", linewidth=1.0),
            flierprops=dict(markerfacecolor="#999999", markeredgecolor="#999999", markersize=2.5),
            meanprops=dict(marker="o", markerfacecolor=colors[model], markeredgecolor=colors[model], markersize=3),
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(GROUPS)
    ax.set_ylabel("Accuracy")
    ax.set_title("Distribution across personas — Accuracy (grouped by model)")

    all_vals = []
    for m in model_order:
        for g in GROUPS:
            all_vals.extend(acc_by_model[m][g])
    lo = min(all_vals) if all_vals else 0.0
    hi = max(all_vals) if all_vals else 1.0
    if hi - lo < 0.08:
        mid = (hi + lo) / 2.0
        lo, hi = mid - 0.04, mid + 0.04
    ax.set_ylim(max(0.0, lo - 0.015), min(1.0, hi + 0.015))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))

    handles = [plt.Line2D([0], [0], color='w', marker='s', markerfacecolor=colors[m], markersize=8,
                           label=m) for m in model_order]
    ax.legend(handles=handles, title="model", loc="upper left")

    out_png = os.path.join(PLOTS_DIR, "persona_accuracy_boxplots_grouped.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {out_png}")


def draw_grouped_iou_boxplots(iou_by_model):
    model_order = ["gpt_oss_120b", "mistral_medium", "qwen3_32b"]
    colors = {
        "gpt_oss_120b": "#4C78A8",
        "mistral_medium": "#F58518",
        "qwen3_32b": "#54A24B",
    }

    fig, ax = plt.subplots(figsize=(12, 4.2))

    x_centers = list(range(len(GROUPS)))
    offset = 0.25
    positions_map = {
        "gpt_oss_120b": [-offset + xc for xc in x_centers],
        "mistral_medium": [0.0 + xc for xc in x_centers],
        "qwen3_32b": [offset + xc for xc in x_centers],
    }

    for model in model_order:
        series = [iou_by_model[model][g] for g in GROUPS]
        ax.boxplot(
            series,
            positions=positions_map[model],
            widths=0.22,
            patch_artist=True,
            showmeans=True,
            meanline=False,
            boxprops=dict(facecolor=colors[model], edgecolor="#333333", linewidth=1.0, alpha=0.6),
            medianprops=dict(color="#111111", linewidth=1.2),
            whiskerprops=dict(color="#333333", linewidth=1.0),
            capprops=dict(color="#333333", linewidth=1.0),
            flierprops=dict(markerfacecolor="#999999", markeredgecolor="#999999", markersize=2.5),
            meanprops=dict(marker="o", markerfacecolor=colors[model], markeredgecolor=colors[model], markersize=3),
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(GROUPS)
    ax.set_ylabel("IOU-F1")
    ax.set_title("Distribution across personas — IOU-F1 (grouped by model)")

    all_vals = []
    for m in model_order:
        for g in GROUPS:
            all_vals.extend(iou_by_model[m][g])
    lo = min(all_vals) if all_vals else 0.0
    hi = max(all_vals) if all_vals else 1.0
    if hi - lo < 0.08:
        mid = (hi + lo) / 2.0
        lo, hi = mid - 0.04, mid + 0.04
    ax.set_ylim(max(0.0, lo - 0.015), min(1.0, hi + 0.015))

    handles = [plt.Line2D([0], [0], color='w', marker='s', markerfacecolor=colors[m], markersize=8,
                           label=m) for m in model_order]
    ax.legend(handles=handles, title="model", loc="upper left")

    out_png = os.path.join(PLOTS_DIR, "persona_iouf1_boxplots_grouped.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {out_png}")


def main():
    # Accuracy
    acc = load_accuracy(ACC_CSV)
    base_acc = load_baseline_map_for_accuracy(BASE_ACC_CSV)
    draw_boxplots(acc, ylabel="Accuracy", out_png=os.path.join(PLOTS_DIR, "persona_accuracy_boxplots.png"), percent=True, baseline_map=base_acc)
    draw_grouped_accuracy_boxplots(acc)

    # Token-F1 and IOU-F1
    token = load_rationale(RAT_CSV, key="token_f1")
    base_tok = load_baseline_map_for_rationale(BASE_RAT_CSV, key="token_f1")
    draw_boxplots(token, ylabel="token-F1", out_png=os.path.join(PLOTS_DIR, "persona_tokenf1_boxplots.png"), percent=False, baseline_map=base_tok)
    # Grouped token-F1 boxplots: 3 models side-by-side per group
    draw_grouped_token_boxplots(token)

    iou = load_rationale(RAT_CSV, key="iou_f1")
    base_iou = load_baseline_map_for_rationale(BASE_RAT_CSV, key="iou_f1")
    draw_boxplots(iou, ylabel="IOU-F1", out_png=os.path.join(PLOTS_DIR, "persona_iouf1_boxplots.png"), percent=False, baseline_map=base_iou)
    draw_grouped_iou_boxplots(iou)


if __name__ == "__main__":
    main()
