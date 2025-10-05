import csv
import json
import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

DATASET_ROOT = REPO_ROOT / "results"
GT_DIR = REPO_ROOT / "data" / "SST"

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
GROUPS = ["BO", "BY", "LO", "LY", "WO", "WY"]


def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("\u00a0", " ").replace("\u2019", "'")
    s = re.sub(r"\s+", " ", s)
    return s


def print_confusion_matrix(true_labels, pred_labels, title, labels=None):
    """Print a nicely formatted confusion matrix"""
    if len(true_labels) == 0:
        print(f"    {title}: No data")
        return

    # Compute confusion matrix
    if labels is None:
        labels = sorted(list(set(true_labels + pred_labels)))

    cm = confusion_matrix(true_labels, pred_labels, labels=labels)

    print(f"    {title}:")
    print(f"    {'':>12}", end="")
    for label in labels:
        print(f"{label:>8}", end="")
    print()

    for i, true_label in enumerate(labels):
        print(f"    {true_label:>12}", end="")
        for j, pred_label in enumerate(labels):
            print(f"{cm[i,j]:>8}", end="")
        print()
    print()


def load_group_ground_truth(group: str) -> dict:
    path = GT_DIR / f"{group}_processed.json"
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    return {item["QID"]: item["label"] for item in items}


def load_model_baseline_predictions(model_dir: Path, model_name: str):
    """Load baseline predictions for a model"""
    # Try different naming patterns
    possible_names = [
        f"baseline_{model_name}_sst.jsonl",
        f"baseline_mistral_sst.jsonl" if "mistral" in model_name else f"baseline_{model_name}_sst.jsonl",
        f"baseline_qwen_sst.jsonl" if "qwen" in model_name else f"baseline_{model_name}_sst.jsonl"
    ]

    baseline_file = None
    for name in possible_names:
        candidate = model_dir / name
        if candidate.exists():
            baseline_file = candidate
            break

    if not baseline_file or not baseline_file.exists():
        print(f"Warning: Baseline file not found. Tried: {possible_names}")
        return {}

    predictions = {}
    with baseline_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            qid = obj.get("question_id") or obj.get("QID")
            ans = obj.get("model_answer") or obj.get("answer")

            if qid and ans:
                predictions[qid] = ans.strip()

    return predictions


def compute_baseline_accuracy(predictions: dict, gt_map: dict):
    correct = 0
    total = 0
    total_possible = len(gt_map)

    # For F1 calculation - collect all labels
    pred_labels = []
    true_labels = []

    # For binary F1 calculation (positive vs negative only)
    binary_pred_labels = []
    binary_true_labels = []

    # Count label distributions
    true_pos_count = 0
    true_neg_count = 0
    true_neutral_count = 0
    pred_pos_count = 0
    pred_neg_count = 0
    pred_neutral_count = 0

    # Define SST label mapping for consistency
    sst_labels = ["positive", "negative", "no sentiment"]

    for qid, gt_label in gt_map.items():
        if qid not in predictions:
            continue
        total += 1

        pred_label = norm(predictions[qid])
        true_label = norm(gt_label)

        if pred_label == true_label:
            correct += 1

        # Count label distributions
        if true_label == "positive":
            true_pos_count += 1
        elif true_label == "negative":
            true_neg_count += 1
        elif true_label == "no sentiment":
            true_neutral_count += 1

        if pred_label == "positive":
            pred_pos_count += 1
        elif pred_label == "negative":
            pred_neg_count += 1
        elif pred_label == "no sentiment":
            pred_neutral_count += 1

        # Collect labels for 3-class F1 calculation
        pred_labels.append(pred_label)
        true_labels.append(true_label)

        # Collect labels for binary F1 calculation (exclude "no sentiment" from ground truth AND predictions)
        if true_label in ["positive", "negative"] and pred_label in ["positive", "negative"]:
            binary_pred_labels.append(pred_label)
            binary_true_labels.append(true_label)

    missing = total_possible - total
    accuracy = (correct / total) if total else 0.0

    # Calculate 3-class macro F1
    macro_f1_3class = 0.0
    if len(pred_labels) > 0 and len(set(true_labels)) > 1:
        try:
            macro_f1_3class = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        except Exception as e:
            print(f"Warning: Could not compute 3-class F1 score: {e}")
            macro_f1_3class = 0.0

    # Calculate binary macro F1 (positive vs negative only)
    macro_f1_binary = 0.0
    if len(binary_pred_labels) > 0 and len(set(binary_true_labels)) > 1:
        try:
            macro_f1_binary = f1_score(binary_true_labels, binary_pred_labels, average='macro', zero_division=0)
        except Exception as e:
            print(f"Warning: Could not compute binary F1 score: {e}")
            macro_f1_binary = 0.0

    return {
        "accuracy": accuracy,
        "macro_f1_3class": macro_f1_3class,
        "macro_f1_binary": macro_f1_binary,
        "correct": correct,
        "total": total,
        "missing": missing,
        "true_pos_count": true_pos_count,
        "true_neg_count": true_neg_count,
        "true_neutral_count": true_neutral_count,
        "pred_pos_count": pred_pos_count,
        "pred_neg_count": pred_neg_count,
        "pred_neutral_count": pred_neutral_count,
        # For confusion matrices
        "pred_labels_3class": pred_labels,
        "true_labels_3class": true_labels,
        "pred_labels_binary": binary_pred_labels,
        "true_labels_binary": binary_true_labels
    }


def compute_baseline_accuracy_for_run(run_number: int):
    """Compute baseline accuracy for a specific run"""
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
        predictions = load_model_baseline_predictions(model_dir, model_name)

        if not predictions:
            print(f"    No baseline predictions found for {model_name}")
            continue

        print(f"    Baseline predictions: {len(predictions)}")

        for group in GROUPS:
            gt_map = load_group_ground_truth(group)
            stats = compute_baseline_accuracy(predictions, gt_map)

            results.append({
                'model': model_name,
                'run': f'r{run_number}',
                'group': group,
                'accuracy': stats['accuracy'],
                'macro_f1_3class': stats['macro_f1_3class'],
                'macro_f1_binary': stats['macro_f1_binary'],
                'correct': stats['correct'],
                'total': stats['total'],
                'missing': stats['missing'],
                'true_pos_count': stats['true_pos_count'],
                'true_neg_count': stats['true_neg_count'],
                'true_neutral_count': stats['true_neutral_count'],
                'pred_pos_count': stats['pred_pos_count'],
                'pred_neg_count': stats['pred_neg_count'],
                'pred_neutral_count': stats['pred_neutral_count']
            })

            print(f"    {group}: {stats['accuracy']*100:.2f}% acc, {stats['macro_f1_3class']*100:.2f}% 3cls-F1, {stats['macro_f1_binary']*100:.2f}% bin-F1")
            print(f"      True: pos={stats['true_pos_count']}, neg={stats['true_neg_count']}, neu={stats['true_neutral_count']}")
            print(f"      Pred: pos={stats['pred_pos_count']}, neg={stats['pred_neg_count']}, neu={stats['pred_neutral_count']}")

            # Print confusion matrices
            print_confusion_matrix(
                stats['true_labels_3class'],
                stats['pred_labels_3class'],
                "3-class Confusion Matrix",
                labels=["negative", "no sentiment", "positive"]
            )
            print_confusion_matrix(
                stats['true_labels_binary'],
                stats['pred_labels_binary'],
                "Binary Confusion Matrix (pos/neg only)",
                labels=["negative", "positive"]
            )

    return results


def compute_averages_across_runs(all_results: list) -> list:
    """Compute averages across runs for each model-group combination"""
    # Group by model and group
    grouped = defaultdict(list)

    for result in all_results:
        key = (result['model'], result['group'])
        grouped[key].append(result)

    averaged_results = []

    for (model, group), results in grouped.items():
        # Extract accuracy and F1 values
        accuracies = [r['accuracy'] for r in results]
        macro_f1s_3class = [r['macro_f1_3class'] for r in results]
        macro_f1s_binary = [r['macro_f1_binary'] for r in results]
        corrects = [r['correct'] for r in results]
        totals = [r['total'] for r in results]
        missings = [r['missing'] for r in results]
        true_pos_counts = [r['true_pos_count'] for r in results]
        true_neg_counts = [r['true_neg_count'] for r in results]
        true_neutral_counts = [r['true_neutral_count'] for r in results]
        pred_pos_counts = [r['pred_pos_count'] for r in results]
        pred_neg_counts = [r['pred_neg_count'] for r in results]
        pred_neutral_counts = [r['pred_neutral_count'] for r in results]

        if accuracies:
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies) if len(accuracies) > 1 else 0.0
            mean_macro_f1_3class = np.mean(macro_f1s_3class)
            std_macro_f1_3class = np.std(macro_f1s_3class) if len(macro_f1s_3class) > 1 else 0.0
            mean_macro_f1_binary = np.mean(macro_f1s_binary)
            std_macro_f1_binary = np.std(macro_f1s_binary) if len(macro_f1s_binary) > 1 else 0.0
            mean_correct = np.mean(corrects)
            mean_total = np.mean(totals)
            mean_missing = np.mean(missings)
            mean_true_pos = np.mean(true_pos_counts)
            mean_true_neg = np.mean(true_neg_counts)
            mean_true_neutral = np.mean(true_neutral_counts)
            mean_pred_pos = np.mean(pred_pos_counts)
            mean_pred_neg = np.mean(pred_neg_counts)
            mean_pred_neutral = np.mean(pred_neutral_counts)
        else:
            mean_accuracy = np.nan
            std_accuracy = np.nan
            mean_macro_f1_3class = np.nan
            std_macro_f1_3class = np.nan
            mean_macro_f1_binary = np.nan
            std_macro_f1_binary = np.nan
            mean_correct = np.nan
            mean_total = np.nan
            mean_missing = np.nan
            mean_true_pos = np.nan
            mean_true_neg = np.nan
            mean_true_neutral = np.nan
            mean_pred_pos = np.nan
            mean_pred_neg = np.nan
            mean_pred_neutral = np.nan

        averaged_results.append({
            'model': model,
            'group': group,
            'accuracy_mean': mean_accuracy,
            'accuracy_std': std_accuracy,
            'macro_f1_3class_mean': mean_macro_f1_3class,
            'macro_f1_3class_std': std_macro_f1_3class,
            'macro_f1_binary_mean': mean_macro_f1_binary,
            'macro_f1_binary_std': std_macro_f1_binary,
            'correct_mean': mean_correct,
            'total_mean': mean_total,
            'missing_mean': mean_missing,
            'true_pos_mean': mean_true_pos,
            'true_neg_mean': mean_true_neg,
            'true_neutral_mean': mean_true_neutral,
            'pred_pos_mean': mean_pred_pos,
            'pred_neg_mean': mean_pred_neg,
            'pred_neutral_mean': mean_pred_neutral,
            'valid_runs': len(accuracies),
            'total_runs': 3
        })

    return averaged_results


def main():
    print("Computing baseline accuracy for SST dataset...")
    print("Processing individual runs and computing averages")

    all_results = []

    # Compute for each run
    for run_num in [1, 2, 3]:
        print(f"\n=== Processing Run {run_num} ===")
        run_results = compute_baseline_accuracy_for_run(run_num)
        all_results.extend(run_results)

    # Save per-run results
    out_dir = THIS_DIR / "csv"
    out_dir.mkdir(exist_ok=True)

    per_run_csv = out_dir / "baseline_accuracy_sst_per_run.csv"
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

    averaged_csv = out_dir / "baseline_accuracy_sst_averaged.csv"
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
        model_acc_stats = defaultdict(list)
        model_f1_3class_stats = defaultdict(list)
        model_f1_binary_stats = defaultdict(list)

        for result in averaged_results:
            if not np.isnan(result['accuracy_mean']):
                model_acc_stats[result['model']].append(result['accuracy_mean'])
            if not np.isnan(result['macro_f1_3class_mean']):
                model_f1_3class_stats[result['model']].append(result['macro_f1_3class_mean'])
            if not np.isnan(result['macro_f1_binary_mean']):
                model_f1_binary_stats[result['model']].append(result['macro_f1_binary_mean'])

        for model in model_acc_stats.keys():
            acc_values = model_acc_stats[model]
            f1_3class_values = model_f1_3class_stats.get(model, [])
            f1_binary_values = model_f1_binary_stats.get(model, [])

            if acc_values:
                acc_str = f"accuracy = {np.mean(acc_values)*100:.2f}±{np.std(acc_values)*100:.2f}%"
            else:
                acc_str = "accuracy = N/A"
            if f1_3class_values:
                f1_3class_str = f"3-class F1 = {np.mean(f1_3class_values)*100:.2f}±{np.std(f1_3class_values)*100:.2f}%"
            else:
                f1_3class_str = "3-class F1 = N/A"
            if f1_binary_values:
                f1_binary_str = f"binary F1 = {np.mean(f1_binary_values)*100:.2f}±{np.std(f1_binary_values)*100:.2f}%"
            else:
                f1_binary_str = "binary F1 = N/A"
            print(f"{model}: {acc_str}, {f1_3class_str}, {f1_binary_str}")

        print(f"\nSummary by Group:")
        group_acc_stats = defaultdict(list)
        group_f1_3class_stats = defaultdict(list)
        group_f1_binary_stats = defaultdict(list)

        for result in averaged_results:
            if not np.isnan(result['accuracy_mean']):
                group_acc_stats[result['group']].append(result['accuracy_mean'])
            if not np.isnan(result['macro_f1_3class_mean']):
                group_f1_3class_stats[result['group']].append(result['macro_f1_3class_mean'])
            if not np.isnan(result['macro_f1_binary_mean']):
                group_f1_binary_stats[result['group']].append(result['macro_f1_binary_mean'])

        for group in group_acc_stats.keys():
            acc_values = group_acc_stats[group]
            f1_3class_values = group_f1_3class_stats.get(group, [])
            f1_binary_values = group_f1_binary_stats.get(group, [])

            if acc_values:
                acc_str = f"accuracy = {np.mean(acc_values)*100:.2f}±{np.std(acc_values)*100:.2f}%"
            else:
                acc_str = "accuracy = N/A"
            if f1_3class_values:
                f1_3class_str = f"3-class F1 = {np.mean(f1_3class_values)*100:.2f}±{np.std(f1_3class_values)*100:.2f}%"
            else:
                f1_3class_str = "3-class F1 = N/A"
            if f1_binary_values:
                f1_binary_str = f"binary F1 = {np.mean(f1_binary_values)*100:.2f}±{np.std(f1_binary_values)*100:.2f}%"
            else:
                f1_binary_str = "binary F1 = N/A"
            print(f"{group}: {acc_str}, {f1_3class_str}, {f1_binary_str}")

    print("\nBaseline accuracy, 3-class macro F1, and binary macro F1 analysis for SST completed!")


if __name__ == "__main__":
    main()