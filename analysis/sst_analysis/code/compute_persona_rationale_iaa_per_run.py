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

# Model configurations for SST individual runs
MODEL_CONFIGS = {
    "gpt_oss_120b": {
        "name": "gpt_oss_120b",
        "r1_dir": DATASET_ROOT / "results_gpt_oss_120b_sst_r1",
        "r2_dir": DATASET_ROOT / "results_gpt_oss_120b_sst_r2",
        "r3_dir": DATASET_ROOT / "results_gpt_oss_120b_sst_r3"
    },
    "mistral_medium": {
        "name": "mistral_medium",
        "r1_dir": DATASET_ROOT / "results_mistral_medium_sst_r1",
        "r2_dir": DATASET_ROOT / "results_mistral_medium_sst_r2",
        "r3_dir": DATASET_ROOT / "results_mistral_medium_sst_r3"
    },
    "qwen3_32b": {
        "name": "qwen3_32b",
        "r1_dir": DATASET_ROOT / "results_qwen3_32b_sst_r1",
        "r2_dir": DATASET_ROOT / "results_qwen3_32b_sst_r2",
        "r3_dir": DATASET_ROOT / "results_qwen3_32b_sst_r3"
    }
}

MODEL_ORDER = list(MODEL_CONFIGS.keys())


def norm(s: str) -> str:
    """Normalize string for comparison"""
    s = (s or "").strip().lower()
    s = s.replace("\u00a0", " ").replace("\u2019", "'")
    s = re.sub(r"\s+", " ", s)
    return s


def extract_rationale_tokens(rationale_data):
    """Extract rationale tokens from the model_rationale field"""
    if not rationale_data:
        return []

    # Handle different formats
    if isinstance(rationale_data, list):
        # Already a list of tokens
        return [norm(token) for token in rationale_data if token]
    elif isinstance(rationale_data, str):
        try:
            # Try parsing as JSON
            parsed = json.loads(rationale_data)
            if isinstance(parsed, list):
                return [norm(token) for token in parsed if token]
        except:
            # Treat as space-separated tokens
            return [norm(token) for token in rationale_data.split() if token]

    return []


def load_model_persona_rationales(model_dir: Path):
    """Load persona rationales from QID*.jsonl files, return dict[qid][persona_text] = rationale_tokens"""
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
                rationale = obj.get("model_rationale")

                # Require essential fields
                if not (persona_text and qid and rationale is not None):
                    continue

                rationale_tokens = extract_rationale_tokens(rationale)
                predictions[qid][persona_text] = rationale_tokens

    return predictions


def compute_average_pairwise_agreement(token_lists):
    """
    Compute average pairwise agreement (Jaccard similarity) across all pairs
    """
    if len(token_lists) <= 1:
        return 1.0 if len(token_lists) == 1 else np.nan

    similarities = []
    for i in range(len(token_lists)):
        for j in range(i + 1, len(token_lists)):
            set_i = set(token_lists[i])
            set_j = set(token_lists[j])

            if len(set_i) == 0 and len(set_j) == 0:
                similarity = 1.0  # Both empty
            elif len(set_i) == 0 or len(set_j) == 0:
                similarity = 0.0  # One empty
            else:
                # Jaccard similarity
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                similarity = intersection / union if union > 0 else 0.0

            similarities.append(similarity)

    return np.mean(similarities) if similarities else np.nan


def compute_fleiss_kappa_for_tokens(token_matrix):
    """
    Compute Fleiss' kappa by treating each unique token as a category
    and whether each rater included it (binary) as the rating
    """
    if not token_matrix:
        return np.nan

    # Collect all unique tokens across all items and raters
    all_tokens = set()
    for item_tokens in token_matrix:
        for token_list in item_tokens:
            all_tokens.update(token_list)

    if not all_tokens:
        return np.nan

    all_tokens = sorted(list(all_tokens))
    n_categories = len(all_tokens)

    if n_categories <= 1:
        return 1.0

    # Create binary rating matrix: items x tokens, with counts of how many raters selected each token
    counts_matrix = []
    for item_tokens in token_matrix:
        item_counts = []
        for token in all_tokens:
            # Count how many raters included this token for this item
            count = sum(1 for token_list in item_tokens if token in token_list)
            item_counts.append(count)
        counts_matrix.append(item_counts)

    # Use statsmodels if available
    if sm_fleiss_kappa is not None:
        try:
            return float(sm_fleiss_kappa(counts_matrix))
        except:
            pass

    # Manual calculation
    n_items = len(counts_matrix)
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


def compute_krippendorff_alpha_for_tokens(token_matrix):
    """
    Compute Krippendorff's alpha by treating tokens as nominal categories
    Each rater-item pair gets a set of tokens they selected
    """
    if kd is None:
        return np.nan

    if not token_matrix:
        return np.nan

    # Collect all unique tokens across all items and raters
    all_tokens = set()
    for item_tokens in token_matrix:
        for token_list in item_tokens:
            all_tokens.update(token_list)

    if not all_tokens:
        return np.nan

    if len(all_tokens) <= 1:
        return 1.0

    all_tokens = sorted(list(all_tokens))
    token_to_id = {token: i for i, token in enumerate(all_tokens)}

    # Determine number of raters (max personas per question)
    max_raters = max(len(item_tokens) for item_tokens in token_matrix) if token_matrix else 0

    # For each token, create a rater x item matrix where value is 1 if rater selected the token, 0 otherwise
    # We'll compute alpha for each token separately and then average (this is one approach)
    # Alternatively, we can create a single reliability matrix

    # Approach: Create reliability matrix where each "measurement" is whether a token was selected
    # Matrix will be raters x (items * tokens)
    data_matrix = []

    for rater_idx in range(max_raters):
        rater_row = []
        for item_tokens in token_matrix:
            if rater_idx < len(item_tokens):
                selected_tokens = set(item_tokens[rater_idx])
                # For each possible token, mark 1 if selected, 0 if not
                for token in all_tokens:
                    rater_row.append(1 if token in selected_tokens else 0)
            else:
                # This rater didn't provide ratings for this item
                for token in all_tokens:
                    rater_row.append(np.nan)
        data_matrix.append(rater_row)

    try:
        alpha = kd.alpha(reliability_data=np.array(data_matrix), level_of_measurement='nominal')
        return float(alpha)
    except:
        return np.nan


def compute_iaa_for_run(run_number: int):
    """Compute inter-annotator agreement on rationales for a specific run"""
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
        predictions = load_model_persona_rationales(model_dir)
        print(f"    Questions found: {len(predictions)}")

        # Collect all persona rationales across all questions
        token_matrix = []  # List of lists of token lists
        all_personas = set()
        valid_questions = 0

        for qid, qid_personas in predictions.items():
            item_token_lists = list(qid_personas.values())
            all_personas.update(qid_personas.keys())

            if len(item_token_lists) >= 2:  # Need at least 2 raters
                token_matrix.append(item_token_lists)
                valid_questions += 1

        print(f"    Found {valid_questions} questions with multiple personas, {len(all_personas)} total personas")

        if not token_matrix:
            # No questions with multiple persona responses
            avg_jaccard = np.nan
            fleiss_k = np.nan
            kripp_alpha = np.nan
            n_questions = 0
            n_personas = 0
        else:
            # Compute average pairwise Jaccard similarity
            all_similarities = []
            for item_tokens in token_matrix:
                item_similarity = compute_average_pairwise_agreement(item_tokens)
                if not np.isnan(item_similarity):
                    all_similarities.append(item_similarity)

            avg_jaccard = np.mean(all_similarities) if all_similarities else np.nan

            # Compute Fleiss' kappa treating tokens as categories
            fleiss_k = compute_fleiss_kappa_for_tokens(token_matrix)

            # Compute Krippendorff's alpha
            kripp_alpha = compute_krippendorff_alpha_for_tokens(token_matrix)

            n_questions = len(token_matrix)
            n_personas = len(all_personas)

        results.append({
            'model': model_name,
            'run': f'r{run_number}',
            'avg_jaccard_similarity': avg_jaccard,
            'fleiss_kappa_tokens': fleiss_k,
            'krippendorff_alpha_tokens': kripp_alpha,
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
        jaccard_sims = [r['avg_jaccard_similarity'] for r in results if not np.isnan(r['avg_jaccard_similarity'])]
        fleiss_kappas = [r['fleiss_kappa_tokens'] for r in results if not np.isnan(r['fleiss_kappa_tokens'])]
        kripp_alphas = [r['krippendorff_alpha_tokens'] for r in results if not np.isnan(r['krippendorff_alpha_tokens'])]

        n_questions_multi = [r['n_questions_with_multiple_personas'] for r in results]
        n_personas = [r['n_personas_total'] for r in results]
        total_questions = [r['total_questions'] for r in results]

        # Compute means and stds
        jaccard_mean = np.mean(jaccard_sims) if jaccard_sims else np.nan
        jaccard_std = np.std(jaccard_sims) if len(jaccard_sims) > 1 else 0.0

        fleiss_mean = np.mean(fleiss_kappas) if fleiss_kappas else np.nan
        fleiss_std = np.std(fleiss_kappas) if len(fleiss_kappas) > 1 else 0.0

        kripp_mean = np.mean(kripp_alphas) if kripp_alphas else np.nan
        kripp_std = np.std(kripp_alphas) if len(kripp_alphas) > 1 else 0.0

        averaged_results.append({
            'model': model,
            'avg_jaccard_similarity_mean': jaccard_mean,
            'avg_jaccard_similarity_std': jaccard_std,
            'fleiss_kappa_tokens_mean': fleiss_mean,
            'fleiss_kappa_tokens_std': fleiss_std,
            'krippendorff_alpha_tokens_mean': kripp_mean,
            'krippendorff_alpha_tokens_std': kripp_std,
            'n_questions_multi_mean': np.mean(n_questions_multi),
            'n_personas_mean': np.mean(n_personas),
            'total_questions_mean': np.mean(total_questions),
            'valid_runs': len(results),
            'total_runs': 3
        })

    return averaged_results


def main():
    print("Computing inter-annotator agreement among personas for SST dataset...")
    print("Processing rationale agreement - per run then average")

    all_results = []

    # Compute for each run
    for run_num in [1, 2, 3]:
        print(f"\n=== Processing Run {run_num} ===")
        run_results = compute_iaa_for_run(run_num)
        all_results.extend(run_results)

    # Save per-run results
    out_dir = THIS_DIR / "csv"
    out_dir.mkdir(exist_ok=True)

    per_run_csv = out_dir / "persona_rationale_iaa_sst_per_run.csv"
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

    averaged_csv = out_dir / "persona_rationale_iaa_sst_averaged.csv"
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
        for result in averaged_results:
            jaccard = f"{result['avg_jaccard_similarity_mean']:.3f}" if not np.isnan(result['avg_jaccard_similarity_mean']) else "N/A"
            fleiss = f"{result['fleiss_kappa_tokens_mean']:.3f}" if not np.isnan(result['fleiss_kappa_tokens_mean']) else "N/A"
            kripp = f"{result['krippendorff_alpha_tokens_mean']:.3f}" if not np.isnan(result['krippendorff_alpha_tokens_mean']) else "N/A"
            print(f"{result['model']}: Jaccard = {jaccard}, Fleiss' κ = {fleiss}, Krippendorff's α = {kripp}")

        # Print detailed results
        print(f"\nDetailed Results (averaged across runs):")
        print(f"{'Model':<15} {'Jaccard':<10} {'Fleiss κ':<10} {'Kripp α':<10} {'Questions':<10} {'Personas':<10}")
        print("-" * 70)

        for result in sorted(averaged_results, key=lambda x: x['model']):
            jaccard = f"{result['avg_jaccard_similarity_mean']:.3f}" if not np.isnan(result['avg_jaccard_similarity_mean']) else "N/A"
            fleiss = f"{result['fleiss_kappa_tokens_mean']:.3f}" if not np.isnan(result['fleiss_kappa_tokens_mean']) else "N/A"
            kripp = f"{result['krippendorff_alpha_tokens_mean']:.3f}" if not np.isnan(result['krippendorff_alpha_tokens_mean']) else "N/A"

            print(f"{result['model']:<15} {jaccard:<10} {fleiss:<10} {kripp:<10} "
                  f"{result['n_questions_multi_mean']:<10.1f} {result['n_personas_mean']:<10.1f}")

    print("\nPersona rationale inter-annotator agreement analysis for SST completed!")


if __name__ == "__main__":
    main()