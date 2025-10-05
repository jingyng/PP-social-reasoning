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

# Tighter, more readable sizing
rcParams.update({
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})


IN_CSV = os.path.join(REPO_ROOT, "3_analysis", "persona_delta_vs_baseline_aligned_rows.csv")
PLOTS_DIR = os.path.join(REPO_ROOT, "3_analysis", "3_4_cose_analysis", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
OUT_PNG = os.path.join(PLOTS_DIR, "persona_abs_and_delta_all_models.png")
SHOW_LABELS = False  # toggle numeric delta labels on/off

MODELS = ["gpt_oss_120b", "mistral_medium", "qwen3_32b"]
GROUPS = ["BO", "BY", "LO", "LY", "WO", "WY"]


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            def ffloat(k):
                return float(row[k]) if row.get(k) else 0.0
            row["token_f1"] = ffloat("token_f1")
            row["iou_f1"] = ffloat("iou_f1")
            row["baseline_token_f1"] = ffloat("baseline_token_f1")
            row["baseline_iou_f1"] = ffloat("baseline_iou_f1")
            row["delta_token_f1"] = ffloat("delta_token_f1")
            row["delta_iou_f1"] = ffloat("delta_iou_f1")
            rows.append(row)
    return rows


def persona_gender(persona: str) -> str:
    try:
        return persona.split("_")[1]
    except Exception:
        return "?"


def main():
    rows = load_rows(IN_CSV)

    # Structure data: model -> group -> gender -> dict of metrics
    data = {m: {g: {} for g in GROUPS} for m in MODELS}
    for row in rows:
        m = row["model"]
        g = row["group"]
        if m not in data or g not in data[m]:
            continue
        gen = persona_gender(row["persona"])  # f/m
        data[m][g][gen] = row

    fig, axes = plt.subplots(2, 3, figsize=(18, 7), sharex=True)
    plt.subplots_adjust(hspace=0.28, wspace=0.15)

    gender_colors = {"f": "#4C78A8", "m": "#F58518"}
    baseline_color = "#666666"

    # Compute shared y-limits per row (token row and IOU row)
    token_vals_all = []
    iou_vals_all = []
    for model in MODELS:
        for g in GROUPS:
            rows_g = data[model][g]
            for gen in ("f", "m"):
                row = rows_g.get(gen)
                if not row:
                    continue
                token_vals_all.extend([row["baseline_token_f1"], row["token_f1"]])
                iou_vals_all.extend([row["baseline_iou_f1"], row["iou_f1"]])

    def span(lo, hi):
        # Ensure minimum span for readability and add margins
        if hi - lo < 0.08:
            mid = (hi + lo) / 2.0
            lo = mid - 0.04
            hi = mid + 0.04
        return max(0.0, lo - 0.015), min(1.0, hi + 0.015)

    t_lo_row, t_hi_row = span(min(token_vals_all) if token_vals_all else 0.0,
                               max(token_vals_all) if token_vals_all else 1.0)
    i_lo_row, i_hi_row = span(min(iou_vals_all) if iou_vals_all else 0.0,
                               max(iou_vals_all) if iou_vals_all else 1.0)

    for col, model in enumerate(MODELS):
        ax_t = axes[0, col]
        ax_b = axes[1, col]

        x = list(range(len(GROUPS)))
        width = 0.18  # horizontal separation unit
        offsets = {"f": -width/2, "m": width/2}

        # Token-F1 subplot (absolute values + delta annotations)
        for i, group in enumerate(GROUPS):
            for gen in ("f", "m"):
                row = data[model][group].get(gen)
                if not row:
                    continue
                x_center = i
                x_base = x_center + offsets[gen] - width*0.6
                x_pers = x_center + offsets[gen] + width*0.6

                # Draw baseline and persona as points
                ax_t.scatter([x_base], [row["baseline_token_f1"]], color=baseline_color, s=24, zorder=3)
                ax_t.scatter([x_pers], [row["token_f1"]], color=gender_colors[gen], s=28, zorder=4, edgecolors='white', linewidths=0.5)

                # Arrow indicating delta direction (baseline -> persona)
                ax_t.annotate(
                    "",
                    xy=(x_pers, row["token_f1"]),
                    xytext=(x_base, row["baseline_token_f1"]),
                    arrowprops=dict(arrowstyle="-|>", color=gender_colors[gen], lw=1.2),
                )

                # Delta annotation next to persona point (optional)
                if SHOW_LABELS:
                    delta = row["delta_token_f1"]
                    ax_t.text(
                        x_pers, row["token_f1"] + 0.004,
                        ("+" if delta >= 0 else "") + f"{delta:.3f}",
                        ha="center", va="bottom", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, lw=0)
                    )

        ax_t.set_title(f"{model} — token-F1 (persona vs baseline)")
        ax_t.set_ylabel("token-F1")
        ax_t.set_xticks(x)
        ax_t.set_xticklabels(GROUPS)
        ax_t.set_ylim(t_lo_row, t_hi_row)
        if col == 0:
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=baseline_color, markersize=6, label='baseline'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=gender_colors['f'], markersize=7, label='persona (f)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=gender_colors['m'], markersize=7, label='persona (m)'),
            ]
            ax_t.legend(handles=handles, loc="upper left")

        # IOU-F1 subplot (absolute values + delta annotations)
        for i, group in enumerate(GROUPS):
            for gen in ("f", "m"):
                row = data[model][group].get(gen)
                if not row:
                    continue
                x_center = i
                x_base = x_center + offsets[gen] - width*0.6
                x_pers = x_center + offsets[gen] + width*0.6

                # Baseline and persona as points
                ax_b.scatter([x_base], [row["baseline_iou_f1"]], color=baseline_color, s=24, zorder=3)
                ax_b.scatter([x_pers], [row["iou_f1"]], color=gender_colors[gen], s=28, zorder=4, edgecolors='white', linewidths=0.5)

                # Arrow (baseline -> persona)
                ax_b.annotate(
                    "",
                    xy=(x_pers, row["iou_f1"]),
                    xytext=(x_base, row["baseline_iou_f1"]),
                    arrowprops=dict(arrowstyle="-|>", color=gender_colors[gen], lw=1.2),
                )

                if SHOW_LABELS:
                    delta = row["delta_iou_f1"]
                    ax_b.text(
                        x_pers, row["iou_f1"] + 0.004,
                        ("+" if delta >= 0 else "") + f"{delta:.3f}",
                        ha="center", va="bottom", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, lw=0)
                    )

        ax_b.set_title(f"{model} — IOU-F1 (persona vs baseline)")
        ax_b.set_ylabel("IOU-F1")
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(GROUPS)
        ax_b.set_ylim(i_lo_row, i_hi_row)

    fig.suptitle("Aligned personas: absolute values and delta vs baseline")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
