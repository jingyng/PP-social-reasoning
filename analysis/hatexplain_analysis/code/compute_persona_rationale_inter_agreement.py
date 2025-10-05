import csv
import json
import numpy as np
import pandas as pd
import krippendorff
from collections import defaultdict
from ast import literal_eval
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import cohen_kappa_score


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
DATASET_ROOT = REPO_ROOT / "results"

MODEL_CONFIGS = {
    "gpt_oss_120b": {
        "name": "gpt_oss_120b",
        "r1_dir": DATASET_ROOT / "results_gpt_oss_120b_hatexplain_r1",
        "r2_dir": DATASET_ROOT / "results_gpt_oss_120b_hatexplain_r2",
        "r3_dir": DATASET_ROOT / "results_gpt_oss_120b_hatexplain_r3"
    },
    "mistral_medium": {
        "name": "mistral_medium",
        "r1_dir": DATASET_ROOT / "results_mistral_medium_hatexplain_r1",
        "r2_dir": DATASET_ROOT / "results_mistral_medium_hatexplain_r2",
        "r3_dir": DATASET_ROOT / "results_mistral_medium_hatexplain_r3"
    },
    "qwen3_32b": {
        "name": "qwen3_32b",
        "r1_dir": DATASET_ROOT / "results_qwen3_32b_hatexplain_r1",
        "r2_dir": DATASET_ROOT / "results_qwen3_32b_hatexplain_r2",
        "r3_dir": DATASET_ROOT / "results_qwen3_32b_hatexplain_r3"
    }
}

# Persona groups by attribute category
PERSONA_GROUPS = {
    'Age': ['15', '35', '65'],
    'Education': ['nfe', 'hs', 'he'],
    'Gender': ['m', 'f'],
    'Loneliness': ['nl', 'sl'],
    'Political': ['l', 'r', 'c'],
    'Race': ['w', 'b', 'a'],
    'Religion': ['chr', 'mus', 'jew', 'ath', 'hin']
}


def _to_list_int(x):
    """Convert various rationale_binary formats to list of integers"""
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, str):
        s = x.strip()
        try:
            val = json.loads(s)
            return [int(v) for v in val]
        except Exception:
            try:
                val = literal_eval(s)
                return [int(v) for v in val]
            except Exception:
                pass
    raise ValueError(f"Cannot parse rationale_binary: {type(x)} -> {x!r}")


def load_model_persona_rationales(model_dir: Path) -> Dict[str, Dict[str, List[int]]]:
    """Load rationale predictions from a model directory"""
    per_persona = defaultdict(dict)

    for path in sorted(model_dir.glob("s*.jsonl")):
        question_id = path.stem  # e.g., 's0000'

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
                # Age personas (15, 35, 65) use triple underscore, others use double underscore
                if "___" in persona_id:
                    # Age personas: 's0000___15' -> '15'
                    _, persona_code = persona_id.split("___", 1)
                elif "__" in persona_id:
                    # Other personas: 's0000__chr' -> 'chr'
                    _, persona_code = persona_id.split("__", 1)
                else:
                    continue

                rb = obj.get("rationale_binary")
                if rb is None:
                    continue

                try:
                    per_persona[persona_code][question_id] = _to_list_int(rb)
                except Exception:
                    continue

    return per_persona


def compute_pairwise_rationale_agreement(pred1: Dict[str, List[int]], pred2: Dict[str, List[int]]) -> tuple:
    """
    Compute Cohen's Kappa for pairwise rationale agreement.
    Treats each token position as a binary annotation (0 or 1).
    Returns: (cohens_kappa, exact_agreement, n_tokens)
    """
    # Find common questions
    common_questions = set(pred1.keys()) & set(pred2.keys())

    if len(common_questions) == 0:
        return None, None, 0

    # Collect all token annotations from both annotators
    all_tokens_1 = []
    all_tokens_2 = []

    for question_id in sorted(common_questions):
        rationale1 = pred1[question_id]
        rationale2 = pred2[question_id]

        # Align lengths
        n = min(len(rationale1), len(rationale2))
        all_tokens_1.extend(rationale1[:n])
        all_tokens_2.extend(rationale2[:n])

    if len(all_tokens_1) == 0:
        return None, None, 0

    # Compute exact agreement (percentage of tokens that match)
    exact_agreement = sum(t1 == t2 for t1, t2 in zip(all_tokens_1, all_tokens_2)) / len(all_tokens_1)

    # Compute Cohen's Kappa
    try:
        kappa = cohen_kappa_score(all_tokens_1, all_tokens_2)
    except:
        kappa = np.nan

    return kappa, exact_agreement, len(all_tokens_1)


def compute_krippendorff_alpha_for_rationales(group_predictions: Dict[str, Dict[str, List[int]]]) -> float:
    """
    Compute Krippendorff's alpha for rationale agreement among personas in a group.
    Each token position in each question is treated as a separate annotation task.
    """
    if len(group_predictions) < 2:
        return np.nan

    # Collect all rationale annotations as separate items
    # Each item is: (question_id, token_position) -> {persona: binary_value}
    annotation_items = defaultdict(dict)

    # First pass: collect all question-token pairs
    for persona, predictions in group_predictions.items():
        for question_id, rationale in predictions.items():
            for token_pos, binary_value in enumerate(rationale):
                item_key = f"{question_id}_{token_pos}"
                annotation_items[item_key][persona] = binary_value

    # Filter items that have annotations from at least 2 personas
    valid_items = {item_key: annotations for item_key, annotations in annotation_items.items()
                   if len(annotations) >= 2}

    if len(valid_items) == 0:
        return np.nan

    # Create reliability data matrix
    # Rows = personas (annotators), Columns = items (question_token pairs)
    personas = list(group_predictions.keys())
    items = list(valid_items.keys())

    data_matrix = []
    for persona in personas:
        row = []
        for item_key in items:
            if persona in valid_items[item_key]:
                row.append(valid_items[item_key][persona])
            else:
                row.append(np.nan)
        data_matrix.append(row)

    # Convert to numpy array
    data_matrix = np.array(data_matrix, dtype=float)

    # Calculate Krippendorff's alpha using nominal level (binary annotations)
    try:
        alpha = krippendorff.alpha(reliability_data=data_matrix,
                                 level_of_measurement='nominal')
        return alpha if not np.isnan(alpha) else 0.0
    except Exception as e:
        print(f"    Error computing alpha: {e}")
        return np.nan


def load_baseline_rationales(model_dir: Path, model_name: str) -> Dict[str, List[int]]:
    """Load baseline rationale predictions from a model directory"""
    baseline_file = model_dir / f"baseline_{model_name}.jsonl"
    baseline_rationales = {}

    if not baseline_file.exists():
        print(f"  Warning: Baseline file not found: {baseline_file}")
        return baseline_rationales

    with baseline_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            question_id = obj.get("question_id")
            rb = obj.get("rationale_binary")
            if question_id is None or rb is None:
                continue

            try:
                baseline_rationales[question_id] = _to_list_int(rb)
            except Exception:
                continue

    return baseline_rationales


def compute_all_personas_rationale_agreement(model_dir: Path, model_name: str, include_baseline: bool = False) -> float:
    """
    Compute Krippendorff's alpha for rationale agreement among all personas.
    Optionally includes baseline predictions.
    """
    # Load all persona rationales
    all_predictions = load_model_persona_rationales(model_dir)

    # Optionally load baseline
    if include_baseline:
        baseline_predictions = load_baseline_rationales(model_dir, model_name)
        if baseline_predictions:
            all_predictions['baseline'] = baseline_predictions

    if len(all_predictions) < 2:
        return np.nan

    # Collect all rationale annotations as separate items
    annotation_items = defaultdict(dict)

    # Collect all question-token pairs
    for annotator, predictions in all_predictions.items():
        for question_id, rationale in predictions.items():
            for token_pos, binary_value in enumerate(rationale):
                item_key = f"{question_id}_{token_pos}"
                annotation_items[item_key][annotator] = binary_value

    # Filter items that have annotations from at least 2 annotators
    valid_items = {item_key: annotations for item_key, annotations in annotation_items.items()
                   if len(annotations) >= 2}

    if len(valid_items) == 0:
        return np.nan

    # Create reliability data matrix
    annotators = list(all_predictions.keys())
    items = list(valid_items.keys())

    data_matrix = []
    for annotator in annotators:
        row = []
        for item_key in items:
            if annotator in valid_items[item_key]:
                row.append(valid_items[item_key][annotator])
            else:
                row.append(np.nan)
        data_matrix.append(row)

    # Convert to numpy array
    data_matrix = np.array(data_matrix, dtype=float)

    # Calculate Krippendorff's alpha using nominal level (binary annotations)
    try:
        alpha = krippendorff.alpha(reliability_data=data_matrix,
                                 level_of_measurement='nominal')
        return alpha if not np.isnan(alpha) else 0.0
    except Exception as e:
        print(f"    Error computing alpha: {e}")
        return np.nan


def compute_group_rationale_agreement(model_dir: Path, group_personas: List[str], model_name: str, run_name: str):
    """Compute rationale agreement within a group of personas"""
    print(f"    Processing {model_name}-{run_name}: {len(group_personas)} personas")

    # Load rationale predictions for all personas
    per_persona = load_model_persona_rationales(model_dir)

    # Extract predictions for personas in this group only
    group_predictions = {}
    for persona in group_personas:
        if persona in per_persona:
            group_predictions[persona] = per_persona[persona]

    if len(group_predictions) < 2:
        print(f"    Warning: Not enough personas with rationale data in group for {model_name}-{run_name}")
        return np.nan, 0, 0

    # Count total annotations
    total_questions = set()
    total_tokens = 0
    for persona_predictions in group_predictions.values():
        total_questions.update(persona_predictions.keys())
        for rationale in persona_predictions.values():
            total_tokens += len(rationale)

    # Compute Krippendorff's alpha for rationale agreement
    kripp_alpha = compute_krippendorff_alpha_for_rationales(group_predictions)

    print(f"    Found {len(group_predictions)} personas, {len(total_questions)} questions")
    return kripp_alpha, len(group_predictions), len(total_questions)


def compute_agreements_for_run(run_number: int):
    """Compute rationale agreements for a specific run"""
    results = []
    all_personas_results = []
    persona_baseline_results = []

    for model_name, model_config in MODEL_CONFIGS.items():
        run_key = f'r{run_number}_dir'
        if run_key not in model_config:
            continue

        model_dir = model_config[run_key]
        if not model_dir.exists():
            print(f"Warning: Directory not found: {model_dir}")
            continue

        print(f"  Processing {model_name} - run {run_number}")

        for group_name, group_personas in PERSONA_GROUPS.items():
            alpha, n_personas, n_questions = compute_group_rationale_agreement(
                model_dir, group_personas, model_name, f"r{run_number}")

            results.append({
                'model': model_name,
                'run': f'r{run_number}',
                'group': group_name,
                'personas_in_group': len(group_personas),
                'personas_with_data': n_personas,
                'questions_covered': n_questions,
                'krippendorff_alpha': alpha
            })

            print(f"    {group_name}: α = {alpha:.3f} (n_personas={n_personas}, n_questions={n_questions})")

        # Compute all-personas agreement (without baseline)
        print(f"    Computing ALL personas agreement (without baseline)...")
        all_personas_alpha_no_baseline = compute_all_personas_rationale_agreement(
            model_dir, model_name, include_baseline=False
        )
        all_personas_results.append({
            'model': model_name,
            'run': f'r{run_number}',
            'include_baseline': False,
            'krippendorff_alpha': all_personas_alpha_no_baseline
        })
        print(f"    ALL personas (no baseline): α = {all_personas_alpha_no_baseline:.3f}")

        # Compute all-personas agreement (with baseline)
        print(f"    Computing ALL personas agreement (with baseline)...")
        all_personas_alpha_with_baseline = compute_all_personas_rationale_agreement(
            model_dir, model_name, include_baseline=True
        )
        all_personas_results.append({
            'model': model_name,
            'run': f'r{run_number}',
            'include_baseline': True,
            'krippendorff_alpha': all_personas_alpha_with_baseline
        })
        print(f"    ALL personas (with baseline): α = {all_personas_alpha_with_baseline:.3f}")

        # Compute pairwise agreement (Cohen's Kappa) between each persona and baseline for rationales
        print(f"    Computing pairwise rationale agreement (Cohen's Kappa) between each persona and baseline...")
        baseline_rationales = load_baseline_rationales(model_dir, model_name)
        if baseline_rationales:
            per_persona = load_model_persona_rationales(model_dir)
            for persona_code, persona_rationales in per_persona.items():
                kappa, exact_agr, n_tokens = compute_pairwise_rationale_agreement(
                    persona_rationales,
                    baseline_rationales
                )
                if kappa is not None:
                    persona_baseline_results.append({
                        'model': model_name,
                        'run': f'r{run_number}',
                        'persona': persona_code,
                        'cohens_kappa': kappa,
                        'exact_agreement': exact_agr,
                        'n_tokens': n_tokens
                    })
            print(f"    Computed {len([r for r in persona_baseline_results if r['model'] == model_name and r['run'] == f'r{run_number}'])} persona-baseline rationale agreements")

    return results, all_personas_results, persona_baseline_results


def compute_averages_across_runs(all_results: List[Dict]) -> List[Dict]:
    """Compute averages across runs for each model-group combination"""
    # Group by model and group
    grouped = defaultdict(list)

    for result in all_results:
        key = (result['model'], result['group'])
        grouped[key].append(result)

    averaged_results = []

    for (model, group), results in grouped.items():
        # Extract alpha values
        alphas = [r['krippendorff_alpha'] for r in results]
        valid_alphas = [a for a in alphas if not np.isnan(a)]

        if valid_alphas:
            mean_alpha = np.mean(valid_alphas)
            std_alpha = np.std(valid_alphas) if len(valid_alphas) > 1 else 0.0
        else:
            mean_alpha = np.nan
            std_alpha = np.nan

        # Get representative values for other metrics
        personas_in_group = results[0]['personas_in_group']
        avg_personas_with_data = np.mean([r['personas_with_data'] for r in results])
        avg_questions_covered = np.mean([r['questions_covered'] for r in results])

        averaged_results.append({
            'model': model,
            'group': group,
            'personas_in_group': personas_in_group,
            'avg_personas_with_data': avg_personas_with_data,
            'avg_questions_covered': avg_questions_covered,
            'krippendorff_alpha_mean': mean_alpha,
            'krippendorff_alpha_std': std_alpha,
            'valid_runs': len(valid_alphas),
            'total_runs': len(alphas)
        })

    return averaged_results


def main():
    print("Computing inter-persona rationale agreement using Krippendorff's alpha...")
    print("Treating personas as annotators for rationale annotation task")
    print("Each token position in each question is treated as a separate annotation item")

    all_results = []
    all_personas_results = []
    persona_baseline_results = []

    # Compute for each run
    for run_num in [1, 2, 3]:
        print(f"\n=== Processing Run {run_num} ===")
        run_results, run_all_personas_results, run_persona_baseline = compute_agreements_for_run(run_num)
        all_results.extend(run_results)
        all_personas_results.extend(run_all_personas_results)
        persona_baseline_results.extend(run_persona_baseline)

    # Save per-run results
    output_dir = THIS_DIR / "csv"
    output_dir.mkdir(exist_ok=True)

    per_run_csv = output_dir / "persona_rationale_inter_agreement_per_run.csv"
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

    averaged_csv = output_dir / "persona_rationale_inter_agreement_averaged.csv"
    with open(averaged_csv, "w", newline="", encoding="utf-8") as f:
        if averaged_results:
            fieldnames = averaged_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(averaged_results)

    print(f"Saved averaged results to: {averaged_csv}")

    # Print summary statistics
    if averaged_results:
        print(f"\nSummary by Model:")
        model_stats = defaultdict(list)
        for result in averaged_results:
            if not np.isnan(result['krippendorff_alpha_mean']):
                model_stats[result['model']].append(result['krippendorff_alpha_mean'])

        for model, alphas in model_stats.items():
            if alphas:
                print(f"{model}: α = {np.mean(alphas):.3f}±{np.std(alphas):.3f}")

        print(f"\nSummary by Attribute Group:")
        group_stats = defaultdict(list)
        for result in averaged_results:
            if not np.isnan(result['krippendorff_alpha_mean']):
                group_stats[result['group']].append(result['krippendorff_alpha_mean'])

        for group, alphas in group_stats.items():
            if alphas:
                print(f"{group}: α = {np.mean(alphas):.3f}±{np.std(alphas):.3f}")

    # Save and report all-personas results
    if all_personas_results:
        all_personas_df = pd.DataFrame(all_personas_results)

        # Save detailed all-personas results
        all_personas_csv = output_dir / "persona_rationale_inter_agreement_all_personas.csv"
        all_personas_df.to_csv(all_personas_csv, index=False)

        # Compute averages across runs
        all_personas_summary = all_personas_df.groupby(['model', 'include_baseline']).agg({
            'krippendorff_alpha': ['mean', 'std']
        }).round(4)
        all_personas_summary.columns = ['_'.join(col).strip() for col in all_personas_summary.columns]
        all_personas_summary = all_personas_summary.reset_index()

        # Save averaged all-personas results
        all_personas_avg_csv = output_dir / "persona_rationale_inter_agreement_all_personas_averaged.csv"
        all_personas_summary.to_csv(all_personas_avg_csv, index=False)

        print(f"\nSaved all-personas results to: {all_personas_csv}")
        print(f"Saved all-personas averaged results to: {all_personas_avg_csv}")

        # Print all-personas summary
        print("\n=== Rationale Agreement Among ALL Personas ===")
        print("Krippendorff's Alpha among ALL personas (averaged across runs):")
        for _, row in all_personas_summary.iterrows():
            baseline_str = "with baseline" if row['include_baseline'] else "without baseline"
            print(f"{row['model']} ({baseline_str}): α = {row['krippendorff_alpha_mean']:.4f} ± {row['krippendorff_alpha_std']:.4f}")

    # Save and report persona-baseline results
    if persona_baseline_results:
        persona_baseline_df = pd.DataFrame(persona_baseline_results)

        # Save detailed persona-baseline results
        persona_baseline_csv = output_dir / "persona_baseline_rationale_agreement.csv"
        persona_baseline_df.to_csv(persona_baseline_csv, index=False)

        # Compute averages across runs
        persona_baseline_summary = persona_baseline_df.groupby(['model', 'persona']).agg({
            'cohens_kappa': ['mean', 'std'],
            'exact_agreement': ['mean', 'std']
        }).round(4)
        persona_baseline_summary.columns = ['_'.join(col).strip() for col in persona_baseline_summary.columns]
        persona_baseline_summary = persona_baseline_summary.reset_index()

        # Save averaged persona-baseline results
        persona_baseline_avg_csv = output_dir / "persona_baseline_rationale_agreement_averaged.csv"
        persona_baseline_summary.to_csv(persona_baseline_avg_csv, index=False)

        print(f"\nSaved persona-baseline rationale results to: {persona_baseline_csv}")
        print(f"Saved persona-baseline averaged results to: {persona_baseline_avg_csv}")

        # Overall summary by model
        model_baseline_summary = persona_baseline_df.groupby(['model']).agg({
            'cohens_kappa': ['mean', 'std'],
            'exact_agreement': ['mean', 'std']
        }).round(4)
        model_baseline_summary.columns = ['_'.join(col).strip() for col in model_baseline_summary.columns]
        model_baseline_summary = model_baseline_summary.reset_index()

        # Print persona-baseline summary
        print("\n=== Persona-Baseline Pairwise Rationale Agreement (Cohen's Kappa) ===")
        print("Average persona-baseline rationale agreement by model (across all personas and runs):")
        for _, row in model_baseline_summary.iterrows():
            print(f"{row['model']}: Cohen's κ = {row['cohens_kappa_mean']:.4f} ± {row['cohens_kappa_std']:.4f}, Exact Agreement = {row['exact_agreement_mean']:.4f} ± {row['exact_agreement_std']:.4f}")

    print("\nPersona rationale inter-annotator agreement analysis completed!")


if __name__ == "__main__":
    main()