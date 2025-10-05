import csv
import json
import re
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path

# Optional third-party packages
try:
    from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss_kappa
except Exception:
    sm_fleiss_kappa = None

try:
    import krippendorff as kd
except Exception:
    kd = None

try:
    from sklearn.metrics import cohen_kappa_score
except Exception:
    cohen_kappa_score = None


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
DATASET_ROOT = REPO_ROOT / "results"

# Model configurations for CoSE individual runs
MODEL_CONFIGS = {
    "gpt_oss_120b": {
        "name": "gpt_oss_120b",
        "r1_dir": DATASET_ROOT / "results_gpt_oss_120b_cose_r1",
        "r2_dir": DATASET_ROOT / "results_gpt_oss_120b_cose_r2",
        "r3_dir": DATASET_ROOT / "results_gpt_oss_120b_cose_r3"
    },
    "mistral_medium": {
        "name": "mistral_medium",
        "r1_dir": DATASET_ROOT / "results_mistral_medium_cose_r1",
        "r2_dir": DATASET_ROOT / "results_mistral_medium_cose_r2",
        "r3_dir": DATASET_ROOT / "results_mistral_medium_cose_r3"
    },
    "qwen3_32b": {
        "name": "qwen3_32b",
        "r1_dir": DATASET_ROOT / "results_qwen3_32b_cose_r1",
        "r2_dir": DATASET_ROOT / "results_qwen3_32b_cose_r2",
        "r3_dir": DATASET_ROOT / "results_qwen3_32b_cose_r3"
    }
}

MODEL_ORDER = list(MODEL_CONFIGS.keys())
GROUPS = ["BO", "BY", "LO", "LY", "WO", "WY"]


def norm(s: str) -> str:
    """Normalize string for comparison"""
    s = (s or "").strip().lower()
    s = s.replace("\u00a0", " ").replace("\u2019", "'")
    s = re.sub(r"\s+", " ", s)
    return s


def load_model_persona_predictions(model_dir: Path):
    """Load persona predictions from QID*.jsonl files, return dict[qid][persona_text] = answer"""
    predictions = defaultdict(dict)

    # Iterate over all QID*.jsonl files
    for path in sorted(model_dir.glob("QID*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                persona_id = obj.get("persona_id") or ""
                # Skip baselines
                if "baseline" in persona_id:
                    continue

                persona_text = obj.get("persona_text")
                qid = obj.get("question_id") or obj.get("QID")
                ans = obj.get("model_answer") or obj.get("answer")

                # Require essential fields
                if not (persona_text and qid and ans):
                    continue

                predictions[qid][persona_text] = norm(ans)

    return predictions


def compute_fleiss_kappa(label_matrix):
    """
    Compute Fleiss' kappa for agreement matrix
    label_matrix: list of lists, where each inner list represents one item's labels from all raters
    """
    if not label_matrix:
        return np.nan

    # Convert to counts matrix
    all_labels = set()
    for item_labels in label_matrix:
        all_labels.update(item_labels)
    all_labels = sorted(list(all_labels))

    if len(all_labels) <= 1:
        return 1.0  # Perfect agreement if only one label exists

    # Create count matrix: items x categories
    counts_matrix = []
    for item_labels in label_matrix:
        counts = [item_labels.count(label) for label in all_labels]
        counts_matrix.append(counts)

    # Use statsmodels if available
    if sm_fleiss_kappa is not None:
        try:
            return float(sm_fleiss_kappa(counts_matrix))
        except:
            pass

    # Manual calculation
    n_items = len(counts_matrix)
    n_categories = len(all_labels)

    if n_items == 0:
        return np.nan

    # Calculate category proportions
    total_assignments = sum(sum(counts) for counts in counts_matrix)
    if total_assignments == 0:
        return np.nan

    category_props = []
    for cat_idx in range(n_categories):
        cat_total = sum(counts[cat_idx] for counts in counts_matrix)
        category_props.append(cat_total / total_assignments)

    # Calculate observed agreement P_bar
    P_bar = 0.0
    for counts in counts_matrix:
        n_raters = sum(counts)
        if n_raters <= 1:
            continue
        # Agreement within this item
        agreement = sum(c * (c - 1) for c in counts) / (n_raters * (n_raters - 1))
        P_bar += agreement

    if n_items > 0:
        P_bar /= n_items

    # Expected agreement P_e
    P_e = sum(p * p for p in category_props)

    if P_e >= 1.0:
        return 1.0

    return (P_bar - P_e) / (1.0 - P_e)


def compute_krippendorff_alpha(label_matrix):
    """
    Compute Krippendorff's alpha using external package if available
    """
    if kd is None:
        return np.nan

    if not label_matrix:
        return np.nan

    # Convert to the format expected by krippendorff package
    # We need a rater x item matrix
    all_labels = set()
    for item_labels in label_matrix:
        all_labels.update(item_labels)
    all_labels = sorted(list(all_labels))

    if len(all_labels) <= 1:
        return 1.0

    # Create label to id mapping
    label_to_id = {label: i for i, label in enumerate(all_labels)}

    # Determine number of raters (max length of any item's label list)
    max_raters = max(len(item_labels) for item_labels in label_matrix) if label_matrix else 0

    # Create data matrix: raters x items
    data_matrix = []
    for rater_idx in range(max_raters):
        rater_data = []
        for item_labels in label_matrix:
            if rater_idx < len(item_labels):
                rater_data.append(label_to_id[item_labels[rater_idx]])
            else:
                rater_data.append(np.nan)
        data_matrix.append(rater_data)

    try:
        alpha = kd.alpha(reliability_data=np.array(data_matrix), level_of_measurement='nominal')
        return float(alpha)
    except:
        return np.nan


def compute_pairwise_cohen_kappa(label_matrix):
    """
    Compute average pairwise Cohen's kappa
    """
    if cohen_kappa_score is None:
        return np.nan

    if not label_matrix:
        return np.nan

    # Collect all pairwise kappas
    kappas = []

    for item_labels in label_matrix:
        if len(item_labels) < 2:
            continue

        # All pairwise combinations for this item
        for i in range(len(item_labels)):
            for j in range(i + 1, len(item_labels)):
                label1 = item_labels[i]
                label2 = item_labels[j]

                # For a single comparison, kappa is either 1 (agree) or calculated from larger sample
                # We'll collect these and compute over all pairs across all items
                kappas.append((label1, label2))

    if not kappas:
        return np.nan

    # Convert to arrays for sklearn
    labels1 = [k[0] for k in kappas]
    labels2 = [k[1] for k in kappas]

    try:
        return cohen_kappa_score(labels1, labels2)
    except:
        return np.nan


def compute_iaa_for_run(run_number: int):
    """Compute inter-annotator agreement for a specific run"""
    results = []

    for model_name, model_config in MODEL_CONFIGS.items():
        run_key = f'r{run_number}_dir'
        if run_key not in model_config:
            continue

        model_dir = model_config[run_key]
        if not model_dir.exists():
            print(f"Warning: Directory not found: {model_dir}")
            continue

        print(f"  Processing {model_name} - run {run_number}")
        predictions = load_model_persona_predictions(model_dir)
        print(f"    Questions found: {len(predictions)}")

        # Collect all persona labels across all questions
        label_matrix = []
        all_personas = set()

        for qid, qid_personas in predictions.items():
            item_labels = list(qid_personas.values())
            all_personas.update(qid_personas.keys())

            if len(item_labels) >= 2:  # Need at least 2 raters
                label_matrix.append(item_labels)

        print(f"    Found {len(label_matrix)} questions with multiple personas, {len(all_personas)} total personas")

        if not label_matrix:
            # No questions with multiple persona responses
            fleiss_k = np.nan
            kripp_alpha = np.nan
            cohen_k = np.nan
            n_questions = 0
            n_personas = 0
        else:
            fleiss_k = compute_fleiss_kappa(label_matrix)
            kripp_alpha = compute_krippendorff_alpha(label_matrix)
            cohen_k = compute_pairwise_cohen_kappa(label_matrix)
            n_questions = len(label_matrix)
            n_personas = len(all_personas)

        results.append({
            'model': model_name,
            'run': f'r{run_number}',
            'fleiss_kappa': fleiss_k,
            'krippendorff_alpha': kripp_alpha,
            'cohen_kappa_avg': cohen_k,
            'n_questions_with_multiple_personas': n_questions,
            'n_personas_total': n_personas,
            'total_questions': len(predictions)
        })

    return results


def compute_averages_across_runs(all_results: list) -> list:
    """Compute averages across runs for each model"""
    # Group by model only
    grouped = defaultdict(list)

    for result in all_results:
        key = result['model']
        grouped[key].append(result)

    averaged_results = []

    for model, results in grouped.items():
        # Extract metric values (handle NaN values)
        fleiss_kappas = [r['fleiss_kappa'] for r in results if not np.isnan(r['fleiss_kappa'])]
        kripp_alphas = [r['krippendorff_alpha'] for r in results if not np.isnan(r['krippendorff_alpha'])]
        cohen_kappas = [r['cohen_kappa_avg'] for r in results if not np.isnan(r['cohen_kappa_avg'])]

        n_questions_multi = [r['n_questions_with_multiple_personas'] for r in results]
        n_personas = [r['n_personas_total'] for r in results]
        total_questions = [r['total_questions'] for r in results]

        # Compute means and stds
        fleiss_mean = np.mean(fleiss_kappas) if fleiss_kappas else np.nan
        fleiss_std = np.std(fleiss_kappas) if len(fleiss_kappas) > 1 else 0.0

        kripp_mean = np.mean(kripp_alphas) if kripp_alphas else np.nan
        kripp_std = np.std(kripp_alphas) if len(kripp_alphas) > 1 else 0.0

        cohen_mean = np.mean(cohen_kappas) if cohen_kappas else np.nan
        cohen_std = np.std(cohen_kappas) if len(cohen_kappas) > 1 else 0.0

        averaged_results.append({
            'model': model,
            'fleiss_kappa_mean': fleiss_mean,
            'fleiss_kappa_std': fleiss_std,
            'krippendorff_alpha_mean': kripp_mean,
            'krippendorff_alpha_std': kripp_std,
            'cohen_kappa_avg_mean': cohen_mean,
            'cohen_kappa_avg_std': cohen_std,
            'n_questions_multi_mean': np.mean(n_questions_multi),
            'n_personas_mean': np.mean(n_personas),
            'total_questions_mean': np.mean(total_questions),
            'valid_runs': len(results),
            'total_runs': 3
        })

    return averaged_results


def main():
    print("Computing inter-annotator agreement among personas for CoSE dataset...")
    print("Processing label prediction agreement - per run then average")

    all_results = []

    # Compute for each run
    for run_num in [1, 2, 3]:
        print(f"\n=== Processing Run {run_num} ===")
        run_results = compute_iaa_for_run(run_num)
        all_results.extend(run_results)

    # Save per-run results
    out_dir = THIS_DIR / "csv"
    out_dir.mkdir(exist_ok=True)

    per_run_csv = out_dir / "persona_label_iaa_cose_per_run.csv"
    with open(per_run_csv, "w", newline="", encoding="utf-8") as f:
        if all_results:
            fieldnames = all_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    print(f"\nSaved per-run results to: {per_run_csv}")

    # Compute and save averaged results
    print("\nComputing averages across runs...")
    averaged_results = compute_averages_across_runs(all_results)

    averaged_csv = out_dir / "persona_label_iaa_cose_averaged.csv"
    with open(averaged_csv, "w", newline="", encoding="utf-8") as f:
        if averaged_results:
            fieldnames = averaged_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(averaged_results)

    print(f"Saved averaged results to: {averaged_csv}")

    # Print summary statistics
    if averaged_results:
        print(f"\nSummary by Model (averaged across runs):")
        model_stats = defaultdict(list)
        for result in averaged_results:
            if not np.isnan(result['fleiss_kappa_mean']):
                model_stats[result['model']].append(result['fleiss_kappa_mean'])

        for model, kappas in model_stats.items():
            if kappas:
                print(f"{model}: Fleiss' κ = {np.mean(kappas):.3f}±{np.std(kappas):.3f}")

        # Print detailed results
        print(f"\nDetailed Results (averaged across runs):")
        print(f"{'Model':<15} {'Fleiss κ':<10} {'Kripp α':<10} {'Cohen κ':<10} {'Questions':<10} {'Personas':<10}")
        print("-" * 70)

        for result in sorted(averaged_results, key=lambda x: x['model']):
            fleiss = f"{result['fleiss_kappa_mean']:.3f}" if not np.isnan(result['fleiss_kappa_mean']) else "N/A"
            kripp = f"{result['krippendorff_alpha_mean']:.3f}" if not np.isnan(result['krippendorff_alpha_mean']) else "N/A"
            cohen = f"{result['cohen_kappa_avg_mean']:.3f}" if not np.isnan(result['cohen_kappa_avg_mean']) else "N/A"

            print(f"{result['model']:<15} {fleiss:<10} {kripp:<10} {cohen:<10} "
                  f"{result['n_questions_multi_mean']:<10.1f} {result['n_personas_mean']:<10.1f}")

    print("\nPersona label inter-annotator agreement analysis for CoSE completed!")


if __name__ == "__main__":
    main()