import csv
import os
from collections import defaultdict

# Ensure writable config/cache for Matplotlib in sandboxed envs
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
_mpl_cfg = os.path.join(BASE_DIR, "3_analysis", ".mplconfig")
_cache = os.path.join(BASE_DIR, "3_analysis", ".cache")
os.makedirs(_mpl_cfg, exist_ok=True)
os.makedirs(os.path.join(_cache, "fontconfig"), exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cfg)
os.environ.setdefault("XDG_CACHE_HOME", _cache)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IN_BASELINE = os.path.join(BASE_DIR, "3_analysis", "baseline_accuracy_summary.csv")
IN_MATCHED = os.path.join(BASE_DIR, "3_analysis", "persona_accuracy_matched.csv")
OUT_DIR = os.path.join(BASE_DIR, "3_analysis", "plots")

GROUPS = ["BO", "BY", "LO", "LY", "WO", "WY"]
MODELS = ["gpt_oss_120b", "mistral_medium", "qwen3_32b"]


def load_baseline():
    baseline = defaultdict(dict)  # baseline[model][group] = accuracy
    with open(IN_BASELINE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            group = row["group"]
            if group == "OVERALL":
                continue
            try:
                acc = float(row["accuracy"])
            except Exception:
                continue
            baseline[model][group] = acc
    return baseline


def load_persona_matched():
    # persona_acc[model][group][gender] = accuracy
    persona_acc = defaultdict(lambda: defaultdict(dict))
    with open(IN_MATCHED, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            group = row["group"]
            persona_text = row["persona_text"]
            try:
                acc = float(row["accuracy"])
            except Exception:
                continue
            gender = "Female" if "Female" in persona_text else ("Male" if "Male" in persona_text else "Unknown")
            persona_acc[model][group][gender] = acc
    return persona_acc


def plot_all_models(baseline, persona_acc):
    os.makedirs(OUT_DIR, exist_ok=True)
    # Smaller default fonts to reduce overlap
    plt.rcParams.update({
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    colors = {"Baseline": "#444444", "Female": "#1f77b4", "Male": "#ff7f0e"}
    width = 0.35
    x = list(range(len(GROUPS)))

    for ax, model in zip(axes, MODELS):
        # Convert to percentages
        b = [baseline[model].get(g, np.nan) * 100 if baseline[model].get(g) is not None else np.nan for g in GROUPS]
        f = [persona_acc[model][g].get("Female", np.nan) * 100 if persona_acc[model][g].get("Female") is not None else np.nan for g in GROUPS]
        m = [persona_acc[model][g].get("Male", np.nan) * 100 if persona_acc[model][g].get("Male") is not None else np.nan for g in GROUPS]

        # Bars for persona Female and Male (no baseline bar)
        rects_f = ax.bar([i - width/2 for i in x], f, width=width, label="Persona Female", color=colors["Female"]) 
        rects_m = ax.bar([i + width/2 for i in x], m, width=width, label="Persona Male", color=colors["Male"]) 

        # Short dashed horizontal line for baseline per group
        for i, y in enumerate(b):
            if np.isnan(y):
                continue
            ax.hlines(y, i - width*0.6, i + width*0.6, colors=colors["Baseline"], linestyles="--", linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(GROUPS)

        # Fixed shared y-axis for fair comparison
        ax.set_ylim(0, 100)

        ax.set_title(model, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Accuracy (%)", fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

        # Add value labels on bars
        def _label(rects):
            for r in rects:
                h = r.get_height()
                if np.isnan(h):
                    continue
                # Offset labels by 1.5% of current y-range or minimum 0.4
                y0, y1 = ax.get_ylim()
                offset = max(0.4, 0.015 * (y1 - y0))
                ax.text(r.get_x() + r.get_width() / 2.0, h + offset, f"{h:.2f}%",
                        ha="center", va="bottom", fontsize=8)

        _label(rects_f)
        _label(rects_m)

    # Single legend
    # Custom legend (include baseline line and persona bars)
    legend_handles = [
        Line2D([0], [0], color=colors["Baseline"], linestyle='--', linewidth=2, label='Baseline'),
        Patch(facecolor=colors["Female"], label='Persona Female'),
        Patch(facecolor=colors["Male"], label='Persona Male'),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), prop={"size": 9})
    fig.suptitle("Persona vs Baseline Accuracy by Group — All Models", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    out_path = os.path.join(OUT_DIR, "persona_vs_baseline_all_models.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    baseline = load_baseline()
    persona_acc = load_persona_matched()
    out = plot_all_models(baseline, persona_acc)
    print("Saved plot:")
    print(" -", out)


if __name__ == "__main__":
    main()
