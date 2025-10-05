import csv
import json
import re
import numpy as np
from collections import defaultdict
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

DATASET_ROOT = REPO_ROOT / "results"
GT_DIR = REPO_ROOT / "data" / "cose"

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
    s = (s or "").strip().lower()
    s = s.replace("\u00a0", " ").replace("\u2019", "'")
    s = re.sub(r"\s+", " ", s)
    return s


def load_baseline_predictions(baseline_file: Path) -> dict:
    """Load baseline predictions from a specific run directory"""
    preds = {}
    if not baseline_file.exists():
        return preds

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
                preds[qid] = ans.strip()
    return preds


def load_group_ground_truth(group: str) -> dict:
    path = GT_DIR / f"{group}_processed_with_queries.json"
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    return {item["QID"]: item["label"] for item in items}


def compute_accuracy(preds: dict, gt_map: dict):
    total = 0
    correct = 0
    missing = 0
    for qid, label in gt_map.items():
        if qid not in preds:
            missing += 1
            continue
        total += 1
        if norm(preds[qid]) == norm(label):
            correct += 1
    acc = (correct / total) if total else 0.0
    return acc, correct, total, missing


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

        # Find baseline file (different naming patterns possible)
        baseline_files = list(model_dir.glob("baseline_*.jsonl"))
        if not baseline_files:
            print(f"Warning: No baseline file found in {model_dir}")
            continue

        baseline_file = baseline_files[0]  # Take the first match
        print(f"  Processing {model_name} - run {run_number}: {baseline_file.name}")

        # Load baseline predictions
        baseline_preds = load_baseline_predictions(baseline_file)

        if not baseline_preds:
            print(f"    Warning: No predictions loaded from {baseline_file}")
            continue

        for group in GROUPS:
            gt_map = load_group_ground_truth(group)
            acc, correct, total, missing = compute_accuracy(baseline_preds, gt_map)

            results.append({
                'model': model_name,
                'run': f'r{run_number}',
                'group': group,
                'accuracy': acc,
                'correct': correct,
                'total': total,
                'missing': missing
            })

            print(f"    {group}: {acc*100:.2f}% (correct {correct}/{total}, missing {missing})")

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
        # Extract accuracy values
        accuracies = [r['accuracy'] for r in results]
        corrects = [r['correct'] for r in results]
        totals = [r['total'] for r in results]
        missings = [r['missing'] for r in results]

        if accuracies:
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies) if len(accuracies) > 1 else 0.0
            mean_correct = np.mean(corrects)
            mean_total = np.mean(totals)
            mean_missing = np.mean(missings)
        else:
            mean_accuracy = np.nan
            std_accuracy = np.nan
            mean_correct = np.nan
            mean_total = np.nan
            mean_missing = np.nan

        averaged_results.append({
            'model': model,
            'group': group,
            'accuracy_mean': mean_accuracy,
            'accuracy_std': std_accuracy,
            'correct_mean': mean_correct,
            'total_mean': mean_total,
            'missing_mean': mean_missing,
            'valid_runs': len(accuracies),
            'total_runs': 3
        })

    return averaged_results


def main():
    print("Computing baseline accuracy for CoSE dataset...")
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

    per_run_csv = out_dir / "baseline_accuracy_cose_per_run.csv"
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

    averaged_csv = out_dir / "baseline_accuracy_cose_averaged.csv"
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
            if not np.isnan(result['accuracy_mean']):
                model_stats[result['model']].append(result['accuracy_mean'])

        for model, accuracies in model_stats.items():
            if accuracies:
                print(f"{model}: accuracy = {np.mean(accuracies)*100:.2f}±{np.std(accuracies)*100:.2f}%")

        print(f"\nSummary by Group:")
        group_stats = defaultdict(list)
        for result in averaged_results:
            if not np.isnan(result['accuracy_mean']):
                group_stats[result['group']].append(result['accuracy_mean'])

        for group, accuracies in group_stats.items():
            if accuracies:
                print(f"{group}: accuracy = {np.mean(accuracies)*100:.2f}±{np.std(accuracies)*100:.2f}%")

    print("\nBaseline accuracy analysis for CoSE completed!")


if __name__ == "__main__":
    main()