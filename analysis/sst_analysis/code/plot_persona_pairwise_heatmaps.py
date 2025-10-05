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


PAIRWISE_CSV = os.path.join(REPO_ROOT, "3_analysis", "persona_pairwise_rationale_agreement.csv")
OUT_DIR = os.path.join(REPO_ROOT, "3_analysis")
PLOTS_DIR = os.path.join(REPO_ROOT, "3_analysis", "3_4_cose_analysis", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

MODELS = ["gpt_oss_120b", "mistral_medium", "qwen3_32b"]
PERSONA_CODES = [
    f"{age}_{gender}_{eth}"
    for age in (25, 45)
    for gender in ("f", "m")
    for eth in ("b", "l", "w")
]


def load_pairwise(path):
    # model -> (pa,pb) -> (token, iou)
    data = {m: {} for m in MODELS}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            m = row["model"].strip()
            if m not in data:
                continue
            pa = row["persona_a"].strip()
            pb = row["persona_b"].strip()
            try:
                tok = float(row["token_f1"]) if row.get("token_f1") else 0.0
                iou = float(row["iou_f1"]) if row.get("iou_f1") else 0.0
            except Exception:
                tok = 0.0
                iou = 0.0
            data[m][(pa, pb)] = (tok, iou)
    return data


def matrices_for_model(model_pairs):
    n = len(PERSONA_CODES)
    tok_mat = np.zeros((n, n), dtype=float)
    iou_mat = np.zeros((n, n), dtype=float)
    for i, pa in enumerate(PERSONA_CODES):
        for j, pb in enumerate(PERSONA_CODES):
            tok, iou = model_pairs.get((pa, pb), (0.0, 0.0))
            tok_mat[i, j] = tok
            iou_mat[i, j] = iou
    return tok_mat, iou_mat


def save_matrix_csv(mat, metric_name, model):
    out_csv = os.path.join(OUT_DIR, f"persona_pairwise_{metric_name}_{model}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["persona"] + PERSONA_CODES)
        for i, pa in enumerate(PERSONA_CODES):
            row = [pa] + [f"{mat[i, j]:.6f}" for j in range(len(PERSONA_CODES))]
            w.writerow(row)
    return out_csv


def _annotate(ax, mat):
    nrows, ncols = mat.shape
    for i in range(nrows):
        for j in range(ncols):
            if j > i:  # skip upper triangle to reflect symmetry
                continue
            val = mat[i, j]
            txt = f"{val:.3f}"
            color = "white" if val >= 0.75 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=color)


def plot_heatmap(mat, title, metric_label, out_png, vmin=0.0, vmax=1.0):
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    # Mask upper triangle to only show lower triangle (including diagonal)
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    masked = np.ma.array(mat, mask=mask)
    im = ax.imshow(masked, vmin=vmin, vmax=vmax, cmap="viridis", aspect="equal")
    ax.set_facecolor("white")
    ax.set_xticks(range(len(PERSONA_CODES)))
    ax.set_xticklabels(PERSONA_CODES, rotation=45, ha="right")
    ax.set_yticks(range(len(PERSONA_CODES)))
    ax.set_yticklabels(PERSONA_CODES)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_label)
    _annotate(ax, mat)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    data = load_pairwise(PAIRWISE_CSV)
    for model in MODELS:
        tok_mat, iou_mat = matrices_for_model(data[model])
        tok_csv = save_matrix_csv(tok_mat, "token_f1", model)
        iou_csv = save_matrix_csv(iou_mat, "iou_f1", model)
        print(f"Saved matrices: {tok_csv}, {iou_csv}")

        plot_heatmap(tok_mat, f"{model} — Persona pairwise token-F1", "token-F1",
                     os.path.join(PLOTS_DIR, f"heatmap_persona_pairwise_token_f1_{model}.png"))
        plot_heatmap(iou_mat, f"{model} — Persona pairwise IOU-F1", "IOU-F1",
                     os.path.join(PLOTS_DIR, f"heatmap_persona_pairwise_iou_f1_{model}.png"))


if __name__ == "__main__":
    main()
