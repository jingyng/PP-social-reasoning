import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import glob
from pathlib import Path
import numpy as np
from collections import defaultdict

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

def load_ground_truth(gt_file):
    """Load ground truth data"""
    gt_data = {}
    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            gt_data[data['id']] = data['majority_label']
    return gt_data

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

def load_personas():
    """Load the 21 personas from personas.jsonl"""
    personas = []
    personas_file = REPO_ROOT / "datasets" / "personas_&_questions" / "personas.jsonl"
    with open(personas_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                personas.append(data['id'])
    return personas

def get_label_mapping():
    """Get mapping from string labels to numerical values for error calculation"""
    # Map labels to ordinal values: Normal=0, Offensive=1, Hate=2
    return {
        'Normal': 0,
        'Offensive language': 1,
        'Hate speech': 2
    }

def calculate_flagging_rates(y_true, y_pred):
    """Calculate flagging rates: over-flagging from normal and offensive-to-hate escalation"""
    # Count cases where GT is Normal
    normal_gt_indices = [i for i, true_label in enumerate(y_true) if true_label == 'Normal']

    # Count cases where GT is Offensive language
    offensive_gt_indices = [i for i, true_label in enumerate(y_true) if true_label == 'Offensive language']

    # Over-flagging rates from Normal
    if normal_gt_indices:
        # Count cases where GT is Normal and prediction is Hate speech
        over_flagged_hate = sum(1 for i in normal_gt_indices
                               if y_pred[i] == 'Hate speech')

        # Count cases where GT is Normal and prediction is Offensive language
        over_flagged_offensive = sum(1 for i in normal_gt_indices
                                    if y_pred[i] == 'Offensive language')

        hate_over_flagging_rate = over_flagged_hate / len(normal_gt_indices)
        offensive_over_flagging_rate = over_flagged_offensive / len(normal_gt_indices)
    else:
        hate_over_flagging_rate = 0.0
        offensive_over_flagging_rate = 0.0

    # Offensive-to-hate escalation rate
    if offensive_gt_indices:
        # Count cases where GT is Offensive and prediction is Hate speech
        offensive_to_hate = sum(1 for i in offensive_gt_indices
                               if y_pred[i] == 'Hate speech')

        offensive_to_hate_rate = offensive_to_hate / len(offensive_gt_indices)
    else:
        offensive_to_hate_rate = 0.0

    return hate_over_flagging_rate, offensive_over_flagging_rate, offensive_to_hate_rate

def calculate_metrics(results_file, gt_data):
    """Calculate accuracy, F1 scores, mean error, mean absolute error, and flagging rates for a single results file"""
    y_true = []
    y_pred = []
    label_mapping = get_label_mapping()

    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            question_id = data['question_id']
            if question_id in gt_data:
                # Handle different label field names
                label = data.get('label') 
                if label is not None:  # Only add if label exists
                    y_true.append(gt_data[question_id])
                    y_pred.append(label)

    if not y_true:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Calculate accuracy and F1 scores
    accuracy = accuracy_score(y_true, y_pred)
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    # Calculate mean error and mean absolute error
    y_true_numeric = [label_mapping[label] for label in y_true]
    y_pred_numeric = [label_mapping[label] for label in y_pred]

    errors = [pred - true for pred, true in zip(y_pred_numeric, y_true_numeric)]
    mean_error = np.mean(errors)
    mean_absolute_error = np.mean([abs(error) for error in errors])

    # Calculate flagging rates
    hate_over_flagging_rate, offensive_over_flagging_rate, offensive_to_hate_rate = calculate_flagging_rates(y_true, y_pred)

    return accuracy, f1_weighted, f1_macro, mean_error, mean_absolute_error, hate_over_flagging_rate, offensive_over_flagging_rate, offensive_to_hate_rate

def calculate_persona_metrics(results_dir, persona_id, gt_data, persona_mapping):
    """Calculate accuracy, F1 scores, mean error, mean absolute error, and flagging rates for a specific persona across all files"""
    y_true = []
    y_pred = []
    label_mapping = get_label_mapping()

    # Get abbreviated persona ID
    abbreviated_id = persona_mapping.get(persona_id)
    if abbreviated_id is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Get all s*.jsonl files
    files = list(results_dir.glob("s*.jsonl"))

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
                    if question_id in gt_data:
                        label = data.get('label')
                        if label is not None:  # Only add if label exists
                            y_true.append(gt_data[question_id])
                            y_pred.append(label)

    if not y_true:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Calculate accuracy and F1 scores
    accuracy = accuracy_score(y_true, y_pred)
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    # Calculate mean error and mean absolute error
    y_true_numeric = [label_mapping[label] for label in y_true]
    y_pred_numeric = [label_mapping[label] for label in y_pred]

    errors = [pred - true for pred, true in zip(y_pred_numeric, y_true_numeric)]
    mean_error = np.mean(errors)
    mean_absolute_error = np.mean([abs(error) for error in errors])

    # Calculate flagging rates
    hate_over_flagging_rate, offensive_over_flagging_rate, offensive_to_hate_rate = calculate_flagging_rates(y_true, y_pred)

    return accuracy, f1_weighted, f1_macro, mean_error, mean_absolute_error, hate_over_flagging_rate, offensive_over_flagging_rate, offensive_to_hate_rate

def compute_averages(all_results):
    """Compute averages across runs for each model-persona combination"""
    # Group by model, persona_type, persona_id
    grouped = defaultdict(list)

    for result in all_results:
        key = (result['model'], result['persona_type'], result['persona_id'])
        grouped[key].append(result)

    averaged_results = []

    for (model, persona_type, persona_id), results in grouped.items():
        if not results:
            continue

        accuracy_values = [r['accuracy'] for r in results]
        f1_weighted_values = [r['f1_weighted'] for r in results]
        f1_macro_values = [r['f1_macro'] for r in results]
        mean_error_values = [r['mean_error'] for r in results]
        mean_absolute_error_values = [r['mean_absolute_error'] for r in results]
        hate_over_flagging_values = [r['hate_over_flagging_rate'] for r in results]
        offensive_over_flagging_values = [r['offensive_over_flagging_rate'] for r in results]
        offensive_to_hate_values = [r['offensive_to_hate_rate'] for r in results]

        averaged_results.append({
            'model': model,
            'persona_type': persona_type,
            'persona_id': persona_id,
            'accuracy_mean': np.mean(accuracy_values),
            'accuracy_std': np.std(accuracy_values),
            'f1_weighted_mean': np.mean(f1_weighted_values),
            'f1_weighted_std': np.std(f1_weighted_values),
            'f1_macro_mean': np.mean(f1_macro_values),
            'f1_macro_std': np.std(f1_macro_values),
            'mean_error_mean': np.mean(mean_error_values),
            'mean_error_std': np.std(mean_error_values),
            'mean_absolute_error_mean': np.mean(mean_absolute_error_values),
            'mean_absolute_error_std': np.std(mean_absolute_error_values),
            'hate_over_flagging_rate_mean': np.mean(hate_over_flagging_values),
            'hate_over_flagging_rate_std': np.std(hate_over_flagging_values),
            'offensive_over_flagging_rate_mean': np.mean(offensive_over_flagging_values),
            'offensive_over_flagging_rate_std': np.std(offensive_over_flagging_values),
            'offensive_to_hate_rate_mean': np.mean(offensive_to_hate_values),
            'offensive_to_hate_rate_std': np.std(offensive_to_hate_values),
            'n_runs': len(results)
        })

    return averaged_results

def main():
    # Load ground truth and personas
    gt_file = REPO_ROOT / "datasets" / "personas_&_questions" / "hatexplain_500_hybrid.jsonl"
    gt_data = load_ground_truth(gt_file)
    personas = load_personas()
    persona_mapping = get_persona_mapping()

    # Model configurations for all 3 runs
    models = [
        {
            'name': 'gpt_oss_120b',
            'r1_dir': REPO_ROOT / "results" / 'results_gpt_oss_120b_hatexplain_r1',
            'r2_dir': REPO_ROOT / "results" / 'results_gpt_oss_120b_hatexplain_r2',
            'r3_dir': REPO_ROOT / "results" / 'results_gpt_oss_120b_hatexplain_r3'
        },
        {
            'name': 'mistral_medium',
            'r1_dir': REPO_ROOT / "results" / 'results_mistral_medium_hatexplain_r1',
            'r2_dir': REPO_ROOT / "results" / 'results_mistral_medium_hatexplain_r2',
            'r3_dir': REPO_ROOT / "results" / 'results_mistral_medium_hatexplain_r3'
        },
        {
            'name': 'qwen3_32b',
            'r1_dir': REPO_ROOT / "results" / 'results_qwen3_32b_hatexplain_r1',
            'r2_dir': REPO_ROOT / "results" / 'results_qwen3_32b_hatexplain_r2',
            'r3_dir': REPO_ROOT / "results" / 'results_qwen3_32b_hatexplain_r3'
        }
    ]

    all_results = []

    for model in models:
        print(f"\nProcessing {model['name']}...")

        # Process all 3 runs
        for run_name, run_dir in [('r1', model['r1_dir']), ('r2', model['r2_dir']), ('r3', model['r3_dir'])]:
            print(f"  Processing {run_name}...")

            # Calculate baseline metrics
            baseline_files = list(run_dir.glob('baseline_*.jsonl'))
            if not baseline_files:
                print(f"    Warning: No baseline file found in {run_dir}")
                continue

            baseline_file = baseline_files[0]
            baseline_acc, baseline_f1_weighted, baseline_f1_macro, baseline_me, baseline_mae, baseline_hate_ofr, baseline_offensive_ofr, baseline_off_to_hate = calculate_metrics(baseline_file, gt_data)

            all_results.append({
                'model': model['name'],
                'run': run_name,
                'persona_type': 'baseline',
                'persona_id': 'baseline',
                'accuracy': baseline_acc,
                'f1_weighted': baseline_f1_weighted,
                'f1_macro': baseline_f1_macro,
                'mean_error': baseline_me,
                'mean_absolute_error': baseline_mae,
                'hate_over_flagging_rate': baseline_hate_ofr,
                'offensive_over_flagging_rate': baseline_offensive_ofr,
                'offensive_to_hate_rate': baseline_off_to_hate
            })
            print(f"    Baseline: accuracy={baseline_acc:.4f}, f1_weighted={baseline_f1_weighted:.4f}, f1_macro={baseline_f1_macro:.4f}, me={baseline_me:.4f}, mae={baseline_mae:.4f}, hate_ofr={baseline_hate_ofr:.4f}, offensive_ofr={baseline_offensive_ofr:.4f}, off_to_hate={baseline_off_to_hate:.4f}")

            # Calculate metrics for each persona
            for persona_id in personas:
                accuracy, f1_weighted, f1_macro, mean_error, mean_absolute_error, hate_over_flagging_rate, offensive_over_flagging_rate, offensive_to_hate_rate = calculate_persona_metrics(run_dir, persona_id, gt_data, persona_mapping)

                all_results.append({
                    'model': model['name'],
                    'run': run_name,
                    'persona_type': 'persona',
                    'persona_id': persona_id,
                    'accuracy': accuracy,
                    'f1_weighted': f1_weighted,
                    'f1_macro': f1_macro,
                    'mean_error': mean_error,
                    'mean_absolute_error': mean_absolute_error,
                    'hate_over_flagging_rate': hate_over_flagging_rate,
                    'offensive_over_flagging_rate': offensive_over_flagging_rate,
                    'offensive_to_hate_rate': offensive_to_hate_rate
                })

    # Save per-run results
    per_run_df = pd.DataFrame(all_results)
    per_run_file = THIS_DIR / "csv" / 'accuracy_f1_me_mae_flagging_rates_results_per_run.csv'
    per_run_df.to_csv(per_run_file, index=False)
    print(f"\nPer-run results saved to {per_run_file}")

    # Compute and save averaged results
    print("Computing averages across runs...")
    averaged_results = compute_averages(all_results)

    averaged_df = pd.DataFrame(averaged_results)
    averaged_file = THIS_DIR / "csv" / 'accuracy_f1_me_mae_flagging_rates_results_averaged.csv'
    averaged_df.to_csv(averaged_file, index=False)
    print(f"Averaged results saved to {averaged_file}")

    # Print summary statistics
    print(f"\nSummary Statistics:")
    for model_name in ['gpt_oss_120b', 'mistral_medium', 'qwen3_32b']:
        model_results = [r for r in averaged_results if r['model'] == model_name]

        # Baseline results
        baseline_results = [r for r in model_results if r['persona_type'] == 'baseline']
        if baseline_results:
            baseline = baseline_results[0]
            print(f"{model_name} baseline:")
            print(f"  accuracy={baseline['accuracy_mean']:.3f}±{baseline['accuracy_std']:.3f}, "
                  f"f1_weighted={baseline['f1_weighted_mean']:.3f}±{baseline['f1_weighted_std']:.3f}, "
                  f"f1_macro={baseline['f1_macro_mean']:.3f}±{baseline['f1_macro_std']:.3f}")
            print(f"  mean_error={baseline['mean_error_mean']:.3f}±{baseline['mean_error_std']:.3f}, "
                  f"mae={baseline['mean_absolute_error_mean']:.3f}±{baseline['mean_absolute_error_std']:.3f}")
            print(f"  hate_ofr={baseline['hate_over_flagging_rate_mean']:.3f}±{baseline['hate_over_flagging_rate_std']:.3f}, "
                  f"offensive_ofr={baseline['offensive_over_flagging_rate_mean']:.3f}±{baseline['offensive_over_flagging_rate_std']:.3f}, "
                  f"off_to_hate={baseline['offensive_to_hate_rate_mean']:.3f}±{baseline['offensive_to_hate_rate_std']:.3f}")

        # Persona results (average across all personas)
        persona_results = [r for r in model_results if r['persona_type'] == 'persona']
        if persona_results:
            accuracy_means = [r['accuracy_mean'] for r in persona_results]
            f1_weighted_means = [r['f1_weighted_mean'] for r in persona_results]
            f1_macro_means = [r['f1_macro_mean'] for r in persona_results]
            me_means = [r['mean_error_mean'] for r in persona_results]
            mae_means = [r['mean_absolute_error_mean'] for r in persona_results]
            hate_ofr_means = [r['hate_over_flagging_rate_mean'] for r in persona_results]
            offensive_ofr_means = [r['offensive_over_flagging_rate_mean'] for r in persona_results]
            off_to_hate_means = [r['offensive_to_hate_rate_mean'] for r in persona_results]

            print(f"{model_name} personas:")
            print(f"  accuracy={np.mean(accuracy_means):.3f}±{np.std(accuracy_means):.3f}, "
                  f"f1_weighted={np.mean(f1_weighted_means):.3f}±{np.std(f1_weighted_means):.3f}, "
                  f"f1_macro={np.mean(f1_macro_means):.3f}±{np.std(f1_macro_means):.3f}")
            print(f"  mean_error={np.mean(me_means):.3f}±{np.std(me_means):.3f}, "
                  f"mae={np.mean(mae_means):.3f}±{np.std(mae_means):.3f}")
            print(f"  hate_ofr={np.mean(hate_ofr_means):.3f}±{np.std(hate_ofr_means):.3f}, "
                  f"offensive_ofr={np.mean(offensive_ofr_means):.3f}±{np.std(offensive_ofr_means):.3f}, "
                  f"off_to_hate={np.mean(off_to_hate_means):.3f}±{np.std(off_to_hate_means):.3f}")

    print("\nAccuracy, F1, Mean Error, Mean Absolute Error, and Flagging Rates computation completed!")

if __name__ == "__main__":
    main()