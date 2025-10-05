import json
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import os
import glob
import krippendorff

def get_persona_groups():
    """Define persona groups by attribute category"""
    persona_groups = {
        'Age': ['15', '35', '65'],
        'Education': ['no_formal_education', 'high_school_education', 'higher_education'],
        'Gender': ['male', 'female'],
        'Loneliness': ['not_lonely', 'somewhat_lonely'],
        'Political': ['left-wing', 'right-wing', 'centrist'],
        'Race': ['white', 'black', 'asian'],
        'Religion': ['christian', 'muslim', 'jewish', 'atheist', 'hindu']
    }
    return persona_groups

def get_persona_mapping():
    """Get mapping from full persona names to abbreviated IDs used in data"""
    mapping = {
        '15': '15',
        '35': '35',
        '65': '65',
        'no_formal_education': 'nfe',
        'high_school_education': 'hs',
        'higher_education': 'he',
        'male': 'm',
        'female': 'f',
        'not_lonely': 'nl',
        'somewhat_lonely': 'sl',
        'left-wing': 'l',
        'right-wing': 'r',
        'centrist': 'c',
        'white': 'w',
        'black': 'b',
        'asian': 'a',
        'christian': 'chr',
        'muslim': 'mus',
        'jewish': 'jew',
        'atheist': 'ath',
        'hindu': 'hin'
    }
    return mapping

def load_persona_predictions(results_dir, persona_id, persona_mapping):
    """Load predictions for a specific persona across all questions"""
    abbreviated_id = persona_mapping.get(persona_id)
    if abbreviated_id is None:
        return {}

    predictions = {}

    # Get all s*.jsonl files
    pattern = os.path.join(results_dir, "s*.jsonl")
    files = glob.glob(pattern)

    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # Check if this line is for our persona
                # Age personas (15, 35, 65) use triple underscore, others use double underscore
                persona_suffix_triple = f'___{abbreviated_id}'
                persona_suffix_double = f'__{abbreviated_id}'
                if data['persona_id'].endswith(persona_suffix_triple) or data['persona_id'].endswith(persona_suffix_double):
                    question_id = data['question_id']
                    label = data.get('label') 
                    if label is not None:
                        predictions[question_id] = label

    return predictions

def compute_pairwise_agreement(predictions1, predictions2):
    """Compute agreement metrics between two sets of predictions"""
    # Find common questions
    common_questions = set(predictions1.keys()) & set(predictions2.keys())

    if len(common_questions) < 2:
        return None, None, None

    # Extract labels for common questions
    labels1 = [predictions1[q] for q in sorted(common_questions)]
    labels2 = [predictions2[q] for q in sorted(common_questions)]

    # Compute accuracy (exact agreement)
    exact_agreement = sum(l1 == l2 for l1, l2 in zip(labels1, labels2)) / len(labels1)

    # Compute Cohen's kappa
    try:
        kappa = cohen_kappa_score(labels1, labels2)
    except:
        kappa = np.nan

    return exact_agreement, kappa, len(common_questions)

def compute_krippendorff_alpha_for_group(group_predictions, level_of_measurement='ordinal'):
    """Compute Krippendorff's alpha for a group of personas using krippendorff package"""
    if len(group_predictions) < 2:
        return np.nan

    # Get all questions that appear in at least one persona's predictions
    all_questions = set()
    for predictions in group_predictions.values():
        all_questions.update(predictions.keys())

    all_questions = sorted(list(all_questions))

    if len(all_questions) == 0:
        return np.nan

    # Create mapping from string labels to numeric values for ordinal scale
    label_to_numeric = {
        'Normal': 0,
        'Offensive language': 1,
        'Hate speech': 2
    }

    # Create data matrix: rows = personas, columns = questions (as required by krippendorff package)
    personas = list(group_predictions.keys())
    data_matrix = []

    for persona in personas:
        row = []
        for question in all_questions:
            if question in group_predictions[persona]:
                label = group_predictions[persona][question]
                # Convert string label to numeric value
                numeric_value = label_to_numeric.get(label, np.nan)
                row.append(numeric_value)
            else:
                row.append(np.nan)
        data_matrix.append(row)

    # Convert to numpy array
    data_matrix = np.array(data_matrix, dtype=float)

    # Calculate Krippendorff's alpha using the package
    try:
        alpha = krippendorff.alpha(reliability_data=data_matrix, level_of_measurement=level_of_measurement)
        return alpha
    except Exception as e:
        print(f"    Error: {e}")
        return np.nan

def compute_group_agreement(results_dir, group_personas, persona_mapping, model_name):
    """Compute agreement within a group of personas"""
    # Load predictions for all personas in the group
    group_predictions = {}
    for persona in group_personas:
        predictions = load_persona_predictions(results_dir, persona, persona_mapping)
        if predictions:
            group_predictions[persona] = predictions

    if len(group_predictions) < 2:
        return [], np.nan

    # Compute Krippendorff's alpha for the group
    kripp_alpha_ordinal = compute_krippendorff_alpha_for_group(group_predictions, 'ordinal')

    # Compute pairwise agreements
    agreements = []
    personas_in_group = list(group_predictions.keys())

    for persona1, persona2 in combinations(personas_in_group, 2):
        exact_agr, kappa, n_samples = compute_pairwise_agreement(
            group_predictions[persona1],
            group_predictions[persona2]
        )

        if exact_agr is not None:
            agreements.append({
                'model': model_name,
                'persona1': persona1,
                'persona2': persona2,
                'exact_agreement': exact_agr,
                'cohens_kappa': kappa,
                'n_samples': n_samples
            })

    return agreements, kripp_alpha_ordinal

def load_baseline_predictions(results_dir, model_name):
    """Load baseline predictions from a model directory"""
    baseline_file = os.path.join(results_dir, f'baseline_{model_name}.jsonl')
    predictions = {}

    if not os.path.exists(baseline_file):
        print(f"  Warning: Baseline file not found: {baseline_file}")
        return predictions

    with open(baseline_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            question_id = data.get('question_id')
            label = data.get('label') or data.get('label_agreement')
            if question_id is not None and label is not None:
                predictions[question_id] = label

    return predictions

def compute_all_personas_agreement(results_dir, persona_mapping, model_name, include_baseline=False):
    """Compute Krippendorff's alpha among all personas (optionally including baseline)"""
    # Get all persona IDs (full names)
    all_persona_ids = list(persona_mapping.keys())

    # Load predictions for all personas
    all_predictions = {}
    for persona_id in all_persona_ids:
        predictions = load_persona_predictions(results_dir, persona_id, persona_mapping)
        if predictions:
            all_predictions[persona_id] = predictions

    # Optionally load baseline
    if include_baseline:
        baseline_predictions = load_baseline_predictions(results_dir, model_name)
        if baseline_predictions:
            all_predictions['baseline'] = baseline_predictions

    if len(all_predictions) < 2:
        return np.nan

    # Get all questions that appear in at least one prediction
    all_questions = set()
    for predictions in all_predictions.values():
        all_questions.update(predictions.keys())

    all_questions = sorted(list(all_questions))

    if len(all_questions) == 0:
        return np.nan

    # Create mapping from string labels to numeric values for ordinal scale
    label_to_numeric = {
        'Normal': 0,
        'Offensive language': 1,
        'Hate speech': 2
    }

    # Create data matrix: rows = annotators (personas + optional baseline), columns = questions
    annotators = list(all_predictions.keys())
    data_matrix = []

    for annotator in annotators:
        row = []
        for question in all_questions:
            if question in all_predictions[annotator]:
                label = all_predictions[annotator][question]
                numeric_value = label_to_numeric.get(label, np.nan)
                row.append(numeric_value)
            else:
                row.append(np.nan)
        data_matrix.append(row)

    # Convert to numpy array
    data_matrix = np.array(data_matrix, dtype=float)

    # Calculate Krippendorff's alpha
    try:
        alpha = krippendorff.alpha(reliability_data=data_matrix, level_of_measurement='ordinal')
        return alpha
    except Exception as e:
        print(f"    Error: {e}")
        return np.nan

def compute_agreements_for_run(run_number):
    """Compute agreements for a specific run"""
    persona_groups = get_persona_groups()
    persona_mapping = get_persona_mapping()

    # Get path to project root (two levels up from script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # Model configurations
    models = [
        {
            'name': 'gpt_oss_120b',
            'results_dir': os.path.join(project_root, f'results/results_gpt_oss_120b_hatexplain_r{run_number}')
        },
        {
            'name': 'mistral_medium',
            'results_dir': os.path.join(project_root, f'results/results_mistral_medium_hatexplain_r{run_number}')
        },
        {
            'name': 'qwen3_32b',
            'results_dir': os.path.join(project_root, f'results/results_qwen3_32b_hatexplain_r{run_number}')
        }
    ]

    all_agreements = []
    all_kripp_alphas = []
    all_personas_alphas = []
    persona_baseline_agreements = []

    for model in models:
        print(f"Processing {model['name']} - Run {run_number}...")

        # Load baseline predictions once for this model
        baseline_predictions = load_baseline_predictions(model['results_dir'], model['name'])

        for group_name, group_personas in persona_groups.items():
            print(f"  Computing agreement for {group_name} group...")

            agreements, kripp_alpha = compute_group_agreement(
                model['results_dir'],
                group_personas,
                persona_mapping,
                model['name']
            )

            # Add group information to pairwise agreements
            for agreement in agreements:
                agreement['attribute_group'] = group_name
                agreement['run'] = run_number

            all_agreements.extend(agreements)

            # Store Krippendorff's alpha
            all_kripp_alphas.append({
                'model': model['name'],
                'attribute_group': group_name,
                'run': run_number,
                'krippendorff_alpha': kripp_alpha,
                'n_personas': len(group_personas)
            })

            print(f"    Found {len(agreements)} pairwise agreements, Krippendorff's α = {kripp_alpha:.4f}")

        # Compute all-personas agreement (without baseline)
        print(f"  Computing agreement among ALL personas (without baseline)...")
        all_personas_alpha = compute_all_personas_agreement(
            model['results_dir'],
            persona_mapping,
            model['name'],
            include_baseline=False
        )
        all_personas_alphas.append({
            'model': model['name'],
            'run': run_number,
            'include_baseline': False,
            'krippendorff_alpha': all_personas_alpha,
            'n_annotators': len(persona_mapping)
        })
        print(f"    All personas (no baseline): Krippendorff's α = {all_personas_alpha:.4f}")

        # Compute all-personas agreement (with baseline)
        print(f"  Computing agreement among ALL personas (with baseline)...")
        all_personas_baseline_alpha = compute_all_personas_agreement(
            model['results_dir'],
            persona_mapping,
            model['name'],
            include_baseline=True
        )
        all_personas_alphas.append({
            'model': model['name'],
            'run': run_number,
            'include_baseline': True,
            'krippendorff_alpha': all_personas_baseline_alpha,
            'n_annotators': len(persona_mapping) + 1  # personas + baseline
        })
        print(f"    All personas (with baseline): Krippendorff's α = {all_personas_baseline_alpha:.4f}")

        # Compute pairwise agreement between each persona and baseline
        print(f"  Computing pairwise agreement (Cohen's kappa) between each persona and baseline...")
        if baseline_predictions:
            for persona_id in persona_mapping.keys():
                persona_predictions = load_persona_predictions(model['results_dir'], persona_id, persona_mapping)
                if persona_predictions:
                    exact_agr, kappa, n_samples = compute_pairwise_agreement(
                        persona_predictions,
                        baseline_predictions
                    )
                    if exact_agr is not None:
                        persona_baseline_agreements.append({
                            'model': model['name'],
                            'run': run_number,
                            'persona': persona_id,
                            'exact_agreement': exact_agr,
                            'cohens_kappa': kappa,
                            'n_samples': n_samples
                        })
            print(f"    Computed {len([a for a in persona_baseline_agreements if a['model'] == model['name'] and a['run'] == run_number])} persona-baseline agreements")

    return all_agreements, all_kripp_alphas, all_personas_alphas, persona_baseline_agreements

def summarize_agreements(all_agreements_df, all_kripp_alphas_df):
    """Create summary statistics for agreements"""
    # Summary by group and model
    group_summary = all_agreements_df.groupby(['model', 'attribute_group', 'run']).agg({
        'exact_agreement': ['mean', 'std', 'count'],
        'cohens_kappa': ['mean', 'std', 'count']
    }).round(4)

    group_summary.columns = ['_'.join(col).strip() for col in group_summary.columns]
    group_summary = group_summary.reset_index()

    # Overall summary by model and run
    model_summary = all_agreements_df.groupby(['model', 'run']).agg({
        'exact_agreement': ['mean', 'std', 'count'],
        'cohens_kappa': ['mean', 'std', 'count']
    }).round(4)

    model_summary.columns = ['_'.join(col).strip() for col in model_summary.columns]
    model_summary = model_summary.reset_index()

    # Average across runs
    avg_group_summary = all_agreements_df.groupby(['model', 'attribute_group']).agg({
        'exact_agreement': ['mean', 'std'],
        'cohens_kappa': ['mean', 'std']
    }).round(4)

    avg_group_summary.columns = ['_'.join(col).strip() for col in avg_group_summary.columns]
    avg_group_summary = avg_group_summary.reset_index()

    avg_model_summary = all_agreements_df.groupby(['model']).agg({
        'exact_agreement': ['mean', 'std'],
        'cohens_kappa': ['mean', 'std']
    }).round(4)

    avg_model_summary.columns = ['_'.join(col).strip() for col in avg_model_summary.columns]
    avg_model_summary = avg_model_summary.reset_index()

    # Krippendorff's alpha summaries
    kripp_group_summary = all_kripp_alphas_df.groupby(['model', 'attribute_group']).agg({
        'krippendorff_alpha': ['mean', 'std']
    }).round(4)

    kripp_group_summary.columns = ['_'.join(col).strip() for col in kripp_group_summary.columns]
    kripp_group_summary = kripp_group_summary.reset_index()

    kripp_model_summary = all_kripp_alphas_df.groupby(['model']).agg({
        'krippendorff_alpha': ['mean', 'std']
    }).round(4)

    kripp_model_summary.columns = ['_'.join(col).strip() for col in kripp_model_summary.columns]
    kripp_model_summary = kripp_model_summary.reset_index()

    return group_summary, model_summary, avg_group_summary, avg_model_summary, kripp_group_summary, kripp_model_summary

def main():
    print("Computing inter-annotator agreement among personas within attribute groups...")

    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(script_dir, 'csv')

    all_agreements = []
    all_kripp_alphas = []
    all_personas_alphas = []
    persona_baseline_agreements = []

    # Compute for each run
    for run_num in [1, 2, 3]:
        print(f"\n=== Processing Run {run_num} ===")
        run_agreements, run_kripp_alphas, run_all_personas_alphas, run_persona_baseline = compute_agreements_for_run(run_num)
        all_agreements.extend(run_agreements)
        all_kripp_alphas.extend(run_kripp_alphas)
        all_personas_alphas.extend(run_all_personas_alphas)
        persona_baseline_agreements.extend(run_persona_baseline)

    # Convert to DataFrames
    all_agreements_df = pd.DataFrame(all_agreements)
    all_kripp_alphas_df = pd.DataFrame(all_kripp_alphas)
    all_personas_alphas_df = pd.DataFrame(all_personas_alphas)
    persona_baseline_df = pd.DataFrame(persona_baseline_agreements)

    if len(all_agreements_df) == 0:
        print("No agreements found!")
        return

    # Save detailed results
    all_agreements_df.to_csv(os.path.join(csv_dir, 'persona_agreements_detailed.csv'), index=False)
    all_kripp_alphas_df.to_csv(os.path.join(csv_dir, 'krippendorff_alphas_detailed.csv'), index=False)
    all_personas_alphas_df.to_csv(os.path.join(csv_dir, 'krippendorff_alphas_all_personas.csv'), index=False)
    persona_baseline_df.to_csv(os.path.join(csv_dir, 'persona_baseline_agreements.csv'), index=False)

    # Create summaries
    group_summary, model_summary, avg_group_summary, avg_model_summary, kripp_group_summary, kripp_model_summary = summarize_agreements(all_agreements_df, all_kripp_alphas_df)

    # Save summaries
    group_summary.to_csv(os.path.join(csv_dir, 'persona_agreements_by_group_and_run.csv'), index=False)
    model_summary.to_csv(os.path.join(csv_dir, 'persona_agreements_by_model_and_run.csv'), index=False)
    avg_group_summary.to_csv(os.path.join(csv_dir, 'persona_agreements_by_group_averaged.csv'), index=False)
    avg_model_summary.to_csv(os.path.join(csv_dir, 'persona_agreements_by_model_averaged.csv'), index=False)
    kripp_group_summary.to_csv(os.path.join(csv_dir, 'krippendorff_alphas_by_group_averaged.csv'), index=False)
    kripp_model_summary.to_csv(os.path.join(csv_dir, 'krippendorff_alphas_by_model_averaged.csv'), index=False)

    print(f"\n=== Results Summary ===")
    print(f"Total pairwise agreements computed: {len(all_agreements_df)}")
    print(f"Total Krippendorff's alpha values computed: {len(all_kripp_alphas_df)}")
    print(f"Results saved to {csv_dir}")

    print("\nAverage Agreement by Model (across all runs):")
    print(avg_model_summary[['model', 'exact_agreement_mean', 'cohens_kappa_mean']].to_string(index=False))

    print("\nAverage Krippendorff's Alpha by Model (across all runs):")
    print(kripp_model_summary[['model', 'krippendorff_alpha_mean', 'krippendorff_alpha_std']].to_string(index=False))

    print("\nAverage Agreement by Attribute Group (across all models and runs):")
    group_avg = all_agreements_df.groupby('attribute_group')[['exact_agreement', 'cohens_kappa']].mean().round(4)
    print(group_avg.to_string())

    print("\nAverage Krippendorff's Alpha by Attribute Group (across all models and runs):")
    kripp_group_avg = all_kripp_alphas_df.groupby('attribute_group')['krippendorff_alpha'].mean().round(4)
    print(kripp_group_avg.to_string())

    # Print all-personas agreement summary
    print("\n=== Agreement Among ALL Personas ===")

    # Average across runs by model and baseline condition
    all_personas_summary = all_personas_alphas_df.groupby(['model', 'include_baseline']).agg({
        'krippendorff_alpha': ['mean', 'std']
    }).round(4)
    all_personas_summary.columns = ['_'.join(col).strip() for col in all_personas_summary.columns]
    all_personas_summary = all_personas_summary.reset_index()

    # Save averaged all-personas results
    all_personas_summary.to_csv(os.path.join(csv_dir, 'krippendorff_alphas_all_personas_averaged.csv'), index=False)

    print("\nKrippendorff's Alpha among ALL personas (averaged across runs):")
    for _, row in all_personas_summary.iterrows():
        baseline_str = "with baseline" if row['include_baseline'] else "without baseline"
        print(f"{row['model']} ({baseline_str}): α = {row['krippendorff_alpha_mean']:.4f} ± {row['krippendorff_alpha_std']:.4f}")

    # Print persona-baseline agreement summary
    if len(persona_baseline_df) > 0:
        print("\n=== Persona-Baseline Pairwise Agreement (Cohen's Kappa) ===")

        # Average across runs by model and persona
        persona_baseline_summary = persona_baseline_df.groupby(['model', 'persona']).agg({
            'exact_agreement': ['mean', 'std'],
            'cohens_kappa': ['mean', 'std']
        }).round(4)
        persona_baseline_summary.columns = ['_'.join(col).strip() for col in persona_baseline_summary.columns]
        persona_baseline_summary = persona_baseline_summary.reset_index()

        # Save averaged persona-baseline results
        persona_baseline_summary.to_csv(os.path.join(csv_dir, 'persona_baseline_agreements_averaged.csv'), index=False)

        # Overall summary by model
        model_baseline_summary = persona_baseline_df.groupby(['model']).agg({
            'exact_agreement': ['mean', 'std'],
            'cohens_kappa': ['mean', 'std']
        }).round(4)
        model_baseline_summary.columns = ['_'.join(col).strip() for col in model_baseline_summary.columns]
        model_baseline_summary = model_baseline_summary.reset_index()

        print("\nAverage persona-baseline agreement by model (across all personas and runs):")
        for _, row in model_baseline_summary.iterrows():
            print(f"{row['model']}: Exact Agreement = {row['exact_agreement_mean']:.4f} ± {row['exact_agreement_std']:.4f}, Cohen's κ = {row['cohens_kappa_mean']:.4f} ± {row['cohens_kappa_std']:.4f}")

if __name__ == "__main__":
    main()