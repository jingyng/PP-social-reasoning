import os
import csv

# # Ensure writable caches before importing matplotlib
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# _mpl_cache = os.path.join(REPO_ROOT, "3_analysis", "3_4_sst_analysis", ".mplcache")
# _xdg_cache = os.path.join(REPO_ROOT, ".cache")
# os.makedirs(_mpl_cache, exist_ok=True)
# os.makedirs(os.path.join(_xdg_cache, "fontconfig"), exist_ok=True)
# os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)
# os.environ.setdefault("XDG_CACHE_HOME", _xdg_cache)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import PercentFormatter

# Compact, readable fonts
rcParams.update({
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

CSV_DIR = os.path.join(REPO_ROOT, "3_analysis", "3_4_sst_analysis", "csv")
BASELINE_CSV = os.path.join(CSV_DIR, "baseline_accuracy_summary.csv")
PERSONA_CSV = os.path.join(CSV_DIR, "persona_accuracy_matched.csv")
PLOTS_DIR = os.path.join(REPO_ROOT, "3_analysis", "3_4_sst_analysis", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
OUT_PNG = os.path.join(PLOTS_DIR, "persona_accuracy_bars_with_baseline.png")

MODELS = ["gpt_oss_120b", "mistral_medium", "qwen3_32b"]
GROUPS = ["BO", "BY", "LO", "LY", "WO", "WY"]


def load_baseline(path):
    base = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            group = row["group"].strip()
            if group == "OVERALL":
                continue
            base[(row["model"], group)] = float(row["accuracy"]) if row.get("accuracy") else 0.0
    return base


def gender_from_text(text: str) -> str:
    t = (text or "").lower()
    if "female" in t:
        return "f"
    if "male" in t:
        return "m"
    return "?"


def load_persona(path):
    data = {m: {g: {} for g in GROUPS} for m in MODELS}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            model = row["model"].strip()
            group = row["group"].strip()
            if model not in data or group not in data[model]:
                continue
            gen = gender_from_text(row.get("persona_text", ""))
            try:
                acc = float(row.get("accuracy", 0.0))
            except Exception:
                acc = 0.0
            data[model][group][gen] = acc
    return data


def compute_group_order(persona_data, anchor_model: str):
    scores = []
    idx = {g: i for i, g in enumerate(GROUPS)}
    for g in GROUPS:
        per_g = persona_data.get(anchor_model, {}).get(g, {})
        vals = [v for k, v in per_g.items() if k in ("f", "m")]
        avg = sum(vals) / len(vals) if vals else -1.0
        scores.append((g, avg))
    scores.sort(key=lambda x: (-x[1], idx[x[0]]))
    return [g for g, _ in scores]


def main():
    baseline = load_baseline(BASELINE_CSV)
    persona = load_persona(PERSONA_CSV)

    gender_colors = {"f": "#4C78A8", "m": "#F58518"}
    baseline_color = "#666666"

    # Determine group order from GPT-OSS-120B persona accuracies
    groups_order = compute_group_order(persona, anchor_model="gpt_oss_120b")

    # Compute shared y-limits
    vals_all = []
    for model in MODELS:
        for g in groups_order:
            per_g = persona[model][g]
            vals_all.extend([v for k, v in per_g.items() if k in ("f", "m")])
            vals_all.append(baseline.get((model, g), 0.0))
    lo = min(vals_all) if vals_all else 0.0
    hi = max(vals_all) if vals_all else 1.0
    if hi - lo < 0.08:
        mid = (hi + lo) / 2.0
        lo, hi = mid - 0.04, mid + 0.04
    lo = max(0.0, lo - 0.015)
    hi = min(1.0, hi + 0.015)

    fig, axes = plt.subplots(1, 3, figsize=(18, 3.6), sharey=True)
    plt.subplots_adjust(wspace=0.12)

    for col, model in enumerate(MODELS):
        ax = axes[col]
        x = list(range(len(groups_order)))
        width = 0.35
        for i, g in enumerate(groups_order):
            f_val = persona[model][g].get("f")
            m_val = persona[model][g].get("m")
            if f_val is not None:
                bars = ax.bar(i - width/2, f_val, width=width, color=gender_colors["f"],
                              label="persona (f)" if (col == 2 and i == 0) else None)
                for b in bars:
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.004, f"{f_val*100:.1f}",
                            ha="center", va="bottom", fontsize=7,
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, lw=0))
            if m_val is not None:
                bars = ax.bar(i + width/2, m_val, width=width, color=gender_colors["m"],
                              label="persona (m)" if (col == 2 and i == 0) else None)
                for b in bars:
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.004, f"{m_val*100:.1f}",
                            ha="center", va="bottom", fontsize=7,
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, lw=0))
            base = baseline.get((model, g))
            if base is not None:
                ax.hlines(base, i - width*0.75, i + width*0.75, colors=baseline_color, linestyles="dashed", linewidth=1.2,
                          label="baseline" if (col == 2 and i == 0) else None)

        ax.set_title(f"{model}")
        ax.set_ylabel("Accuracy" if col == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(groups_order)
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
        if col == 2:
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(loc="upper left")

    # fig.suptitle("Aligned personas: accuracy bars with dashed baselines")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
