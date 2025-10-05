import os
import csv
from collections import defaultdict

# Ensure writable caches before importing matplotlib
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


IN_CSV = os.path.join(REPO_ROOT, "3_analysis", "persona_delta_vs_baseline_aligned_rows.csv")
PLOTS_DIR = os.path.join(REPO_ROOT, "3_analysis", "3_4_sst_analysis", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
OUT_PNG = os.path.join(PLOTS_DIR, "persona_delta_vs_baseline_all_models.png")


MODELS = ["gpt_oss_120b", "mistral_medium", "qwen3_32b"]
GROUPS = ["BO", "BY", "LO", "LY", "WO", "WY"]


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["delta_token_f1"] = float(row["delta_token_f1"]) if row["delta_token_f1"] else 0.0
            row["delta_iou_f1"] = float(row["delta_iou_f1"]) if row["delta_iou_f1"] else 0.0
            rows.append(row)
    return rows


def persona_gender(persona: str) -> str:
    # 25_f_b -> f, 45_m_w -> m
    try:
        return persona.split("_")[1]
    except Exception:
        return "?"


def main():
    rows = load_rows(IN_CSV)

    # Structure: by model -> by group -> list[(gender, delta_token, delta_iou)] with two entries f/m
    data = {m: {g: [] for g in GROUPS} for m in MODELS}
    for row in rows:
        model = row["model"]
        group = row["group"]
        if model not in data or group not in data[model]:
            continue
        g = persona_gender(row["persona"])  # f or m
        data[model][group].append((g, row["delta_token_f1"], row["delta_iou_f1"]))

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    plt.subplots_adjust(hspace=0.25, wspace=0.15)

    gender_colors = {"f": "#4C78A8", "m": "#F58518"}

    for col, model in enumerate(MODELS):
        # Top row: token-F1 deltas
        ax_t = axes[0, col]
        # Bottom row: IOU-F1 deltas
        ax_b = axes[1, col]

        x = list(range(len(GROUPS)))
        width = 0.35

        # Per group, two bars (f on left, m on right)
        f_vals_token = []
        m_vals_token = []
        f_vals_iou = []
        m_vals_iou = []
        for g in GROUPS:
            entries = data[model][g]
            f_tok = m_tok = 0.0
            f_iou = m_iou = 0.0
            for gender, d_tok, d_iou in entries:
                if gender == "f":
                    f_tok = d_tok
                    f_iou = d_iou
                elif gender == "m":
                    m_tok = d_tok
                    m_iou = d_iou
            f_vals_token.append(f_tok)
            m_vals_token.append(m_tok)
            f_vals_iou.append(f_iou)
            m_vals_iou.append(m_iou)

        ax_t.bar([i - width/2 for i in x], f_vals_token, width, label="female", color=gender_colors["f"])
        ax_t.bar([i + width/2 for i in x], m_vals_token, width, label="male", color=gender_colors["m"])
        ax_t.axhline(0, color="#777", linewidth=1)
        ax_t.set_title(f"{model} — Δ token-F1 vs baseline")
        ax_t.set_ylabel("Δ token-F1")
        ax_t.set_xticks(x)
        ax_t.set_xticklabels(GROUPS)
        if col == 0:
            ax_t.legend(loc="upper left")

        ax_b.bar([i - width/2 for i in x], f_vals_iou, width, label="female", color=gender_colors["f"])
        ax_b.bar([i + width/2 for i in x], m_vals_iou, width, label="male", color=gender_colors["m"])
        ax_b.axhline(0, color="#777", linewidth=1)
        ax_b.set_title(f"{model} — Δ IOU-F1 vs baseline")
        ax_b.set_ylabel("Δ IOU-F1")
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(GROUPS)

    fig.suptitle("Aligned personas: delta vs baseline (per model × group × gender)")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
