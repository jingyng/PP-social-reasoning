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
import numpy as np


COMBINED_CSV = os.path.join(REPO_ROOT, "3_analysis", "rationale_agreement_combined.csv")
BASELINE_CSV = os.path.join(REPO_ROOT, "3_analysis", "baseline_rationale_agreement_summary.csv")
SIG_CSV = os.path.join(REPO_ROOT, "3_analysis", "rationale_significance.csv")
OUT_DIR = os.path.join(REPO_ROOT, "3_analysis")
PLOTS_DIR = os.path.join(REPO_ROOT, "3_analysis", "3_4_cose_analysis", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

MODELS = ["gpt_oss_120b", "mistral_medium", "qwen3_32b"]
GROUPS = ["BO", "BY", "LO", "LY", "WO", "WY"]
PERSONA_CODES = [
    f"{age}_{gender}_{eth}"
    for age in (25, 45)
    for gender in ("f", "m")
    for eth in ("b", "l", "w")
]


def load_persona_vs_gt(path):
    # model -> persona -> group -> (tok, iou)
    data = {m: {p: {} for p in PERSONA_CODES} for m in MODELS}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("type") != "persona":
                continue
            m = row["model"].strip()
            p = row["persona"].strip()
            g = row["group"].strip()
            if m not in data or p not in data[m] or g not in GROUPS:
                continue
            try:
                tok = float(row.get("token_f1", 0.0))
                iou = float(row.get("iou_f1", 0.0))
            except Exception:
                tok = 0.0
                iou = 0.0
            data[m][p][g] = (tok, iou)
    return data


def load_baseline_map(path):
    # model -> group -> { 'token_f1': v, 'iou_f1': v }
    base = {m: {g: {"token_f1": 0.0, "iou_f1": 0.0} for g in GROUPS} for m in MODELS}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            m = row["model"].strip()
            g = row["group"].strip()
            if m in base and g in base[m]:
                try:
                    base[m][g]["token_f1"] = float(row.get("token_f1", 0.0))
                except Exception:
                    pass
                try:
                    base[m][g]["iou_f1"] = float(row.get("iou_f1", 0.0))
                except Exception:
                    pass
    return base


def save_matrix_csv(mat, personas, groups, metric_name, model):
    out_csv = os.path.join(OUT_DIR, f"persona_vs_gt_{metric_name}_{model}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["persona"] + groups)
        for i, pa in enumerate(personas):
            row = [pa] + [f"{mat[i, j]:.6f}" for j in range(len(groups))]
            w.writerow(row)
    return out_csv


def _annotate(ax, mat, sig_mat=None):
    nrows, ncols = mat.shape
    for i in range(nrows):
        for j in range(ncols):
            val = mat[i, j]
            txt = f"{val:.3f}"
            # Significance markers
            if sig_mat is not None:
                p = sig_mat[i, j]
                if not np.isnan(p):
                    if p < 0.001:
                        txt += "***"
                    elif p < 0.01:
                        txt += "**"
                    elif p < 0.05:
                        txt += "*"
            color = "white" if val >= 0.75 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=color)


def plot_heatmap(mat, personas, groups, title, metric_label, out_png, vmin=0.0, vmax=1.0, baseline_vals=None, sig_mat=None):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(groups)))
    if baseline_vals is not None:
        xt = [f"{g}\n{baseline_vals[j]:.3f}" for j, g in enumerate(groups)]
    else:
        xt = groups
    ax.set_xticklabels(xt, rotation=0)
    ax.set_yticks(range(len(personas)))
    ax.set_yticklabels(personas)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_label)
    _annotate(ax, mat, sig_mat=sig_mat)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    data = load_persona_vs_gt(COMBINED_CSV)
    base = load_baseline_map(BASELINE_CSV)
    # Load significance results
    sig = {m: {p: {g: {"token": float("nan"), "iou": float("nan")} for g in GROUPS} for p in PERSONA_CODES} for m in MODELS}
    if os.path.exists(SIG_CSV):
        with open(SIG_CSV, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                m = row["model"].strip()
                p = row["persona"].strip()
                g = row["group"].strip()
                if m in sig and p in sig[m] and g in sig[m][p]:
                    try:
                        sig[m][p][g]["token"] = float(row.get("p_token_f1", "nan"))
                    except Exception:
                        pass
                    try:
                        sig[m][p][g]["iou"] = float(row.get("p_iou_f1", "nan"))
                    except Exception:
                        pass
    for model in MODELS:
        # Build matrices
        tok = np.zeros((len(PERSONA_CODES), len(GROUPS)), dtype=float)
        iou = np.zeros((len(PERSONA_CODES), len(GROUPS)), dtype=float)
        tok_sig = np.full((len(PERSONA_CODES), len(GROUPS)), np.nan, dtype=float)
        iou_sig = np.full((len(PERSONA_CODES), len(GROUPS)), np.nan, dtype=float)
        for i, p in enumerate(PERSONA_CODES):
            for j, g in enumerate(GROUPS):
                val = data[model][p].get(g, (0.0, 0.0))
                tok[i, j] = val[0]
                iou[i, j] = val[1]
                tok_sig[i, j] = sig[model][p][g]["token"]
                iou_sig[i, j] = sig[model][p][g]["iou"]

        tok_csv = save_matrix_csv(tok, PERSONA_CODES, GROUPS, "token_f1", model)
        iou_csv = save_matrix_csv(iou, PERSONA_CODES, GROUPS, "iou_f1", model)
        print(f"Saved matrices: {tok_csv}, {iou_csv}")

        tok_baselines = [base[model][g]["token_f1"] for g in GROUPS]
        iou_baselines = [base[model][g]["iou_f1"] for g in GROUPS]

        plot_heatmap(tok, PERSONA_CODES, GROUPS,
                     f"{model} — Persona vs GT (token-F1)\nBaseline per group shown under labels",
                     "token-F1",
                     os.path.join(PLOTS_DIR, f"heatmap_persona_vs_gt_token_f1_{model}.png"),
                     baseline_vals=tok_baselines, sig_mat=tok_sig)
        plot_heatmap(iou, PERSONA_CODES, GROUPS,
                     f"{model} — Persona vs GT (IOU-F1)\nBaseline per group shown under labels",
                     "IOU-F1",
                     os.path.join(PLOTS_DIR, f"heatmap_persona_vs_gt_iou_f1_{model}.png"),
                     baseline_vals=iou_baselines, sig_mat=iou_sig)


if __name__ == "__main__":
    main()
