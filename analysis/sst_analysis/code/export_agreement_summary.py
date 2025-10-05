import csv
import os
from compute_persona_agreement import (
    load_model_results,
    summarize_agreement,
    compute_age_gender_ethnicity_agreements,
)
from compute_iaa import (
    prepare_counts,
    fleiss_kappa,
    krippendorff_alpha_nominal,
    load_baseline,
    prepare_counts_with_baseline,
    fleiss_kappa_statsmodels,
    krippendorff_alpha_pkg,
)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Export persona agreement summary to CSV")
    ap.add_argument("folders", nargs="+", help="Paths to merged result folders")
    ap.add_argument("--out", default="3_analysis/plots/agreement_summary.csv", help="Output CSV path")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for folder in args.folders:
        model = os.path.basename(folder)
        by_qid = load_model_results(folder)
        summary = summarize_agreement(by_qid)
        spec = compute_age_gender_ethnicity_agreements(by_qid)
        # Personas-only IAA (prefer packages)
        per_q_counts, _, k = prepare_counts(by_qid)
        fk_pkg = fleiss_kappa_statsmodels(per_q_counts, k)
        fk = fk_pkg if fk_pkg is not None else fleiss_kappa(per_q_counts, k)
        ka_pkg = krippendorff_alpha_pkg(by_qid)
        ka = ka_pkg if ka_pkg is not None else krippendorff_alpha_nominal(per_q_counts, k)

        # With baseline
        baseline_map = load_baseline(folder)
        fk_bl = float('nan')
        ka_bl = float('nan')
        if baseline_map:
            per_q_counts_bl, _, k_bl = prepare_counts_with_baseline(by_qid, baseline_map)
            fk_bl_pkg = fleiss_kappa_statsmodels(per_q_counts_bl, k_bl)
            fk_bl = fk_bl_pkg if fk_bl_pkg is not None else fleiss_kappa(per_q_counts_bl, k_bl)
            ka_bl_pkg = krippendorff_alpha_pkg(by_qid, include_baseline=True, baseline_map=baseline_map)
            ka_bl = ka_bl_pkg if ka_bl_pkg is not None else krippendorff_alpha_nominal(per_q_counts_bl, k_bl)

        rows.append({
            "model": model,
            "questions": summary["questions"],
            "unanimous_rate": summary["unanimous_rate"],
            "fleiss_kappa_personas": fk,
            "krippendorff_alpha_personas": ka,
            "fleiss_kappa_with_baseline": fk_bl,
            "krippendorff_alpha_with_baseline": ka_bl,
            "age_pair_agree_rate": spec["age_pair_agree_rate"],
            "age_pair_count": spec["age_pair_count"],
            "gender_pair_agree_rate": spec["gender_pair_agree_rate"],
            "gender_pair_count": spec["gender_pair_count"],
            "ethnicity_triad_unanimity_rate": spec["ethnicity_triad_unanimity_rate"],
            "ethnicity_triad_count": spec["ethnicity_triad_count"],
            "ethnicity_triad_fleiss_kappa": spec["ethnicity_triad_fleiss_kappa"],
        })

    fieldnames = [
        "model",
        "questions",
        "unanimous_rate",
        "fleiss_kappa_personas",
        "krippendorff_alpha_personas",
        "fleiss_kappa_with_baseline",
        "krippendorff_alpha_with_baseline",
        "age_pair_agree_rate",
        "age_pair_count",
        "gender_pair_agree_rate",
        "gender_pair_count",
        "ethnicity_triad_unanimity_rate",
        "ethnicity_triad_count",
        "ethnicity_triad_fleiss_kappa",
    ]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
