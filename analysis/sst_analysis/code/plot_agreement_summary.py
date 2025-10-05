import os
import csv
import matplotlib.pyplot as plt


def read_csv(path):
    rows = []
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            rows.append(row)
    return rows


def plot_age_gender(rows, outdir):
    models = [r["model"] for r in rows]
    age_vals = [float(r["age_pair_agree_rate"]) for r in rows]
    gen_vals = [float(r["gender_pair_agree_rate"]) for r in rows]

    x = range(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar([i - w/2 for i in x], age_vals, width=w, label="Age 25 vs 45")
    ax.bar([i + w/2 for i in x], gen_vals, width=w, label="Gender m vs f")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=20, ha='right')
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Agreement')
    ax.set_title('Pairwise Agreement by Attribute (hold others fixed)')
    ax.legend()
    plt.tight_layout()
    out = os.path.join(outdir, 'agreement_age_gender.png')
    plt.savefig(out, dpi=150)
    print(f'Wrote {out}')


def plot_ethnicity(rows, outdir):
    models = [r["model"] for r in rows]
    uni_vals = [float(r["ethnicity_triad_unanimity_rate"]) for r in rows]
    fk_vals = [float(r["ethnicity_triad_fleiss_kappa"]) for r in rows]

    x = range(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar([i - w/2 for i in x], uni_vals, width=w, label="Triad unanimity (b/l/w)")
    ax.bar([i + w/2 for i in x], fk_vals, width=w, label="Fleiss\' kappa (triads)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=20, ha='right')
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Agreement')
    ax.set_title('Ethnicity Agreement (per age+gender triads)')
    ax.legend()
    plt.tight_layout()
    out = os.path.join(outdir, 'agreement_ethnicity.png')
    plt.savefig(out, dpi=150)
    print(f'Wrote {out}')


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Plot agreement summary CSV')
    ap.add_argument('--csv', default='3_analysis/plots/agreement_summary.csv')
    ap.add_argument('--outdir', default='3_analysis/plots')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = read_csv(args.csv)
    # Keep a consistent model order (as in CSV)
    plot_age_gender(rows, args.outdir)
    plot_ethnicity(rows, args.outdir)


if __name__ == '__main__':
    main()

