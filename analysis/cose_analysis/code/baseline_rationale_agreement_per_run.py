import csv
import json
import numpy as np
from ast import literal_eval
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


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

GROUPS = ["BO", "BY", "LO", "LY", "WO", "WY"]


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


def load_baseline_binaries(baseline_file: Path) -> Dict[str, List[int]]:
    """Load baseline rationale binaries from a specific run directory"""
    qid_to_bin = {}
    if not baseline_file.exists():
        return qid_to_bin

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
            rb = obj.get("rationale_binary")
            if qid and rb is not None:
                try:
                    qid_to_bin[qid] = _to_list_int(rb)
                except Exception:
                    continue
    return qid_to_bin


def load_group_ground_truth_binaries(group: str) -> Dict[str, List[int]]:
    """Load ground truth rationale binaries for a group"""
    path = GT_DIR / f"{group}_processed_with_queries.json"
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    qid_to_bin = {}
    for item in items:
        qid = item["QID"]
        rb = item.get("rationale_binary")
        if rb is not None:
            try:
                qid_to_bin[qid] = _to_list_int(rb)
            except Exception:
                continue
    return qid_to_bin


def compute_rationale_f1(pred_bin: List[int], gt_bin: List[int]) -> float:
    """Compute F1 score between predicted and ground truth rationale binaries"""
    if len(pred_bin) != len(gt_bin):
        return 0.0

    pred_set = set(i for i, val in enumerate(pred_bin) if val == 1)
    gt_set = set(i for i, val in enumerate(gt_bin) if val == 1)

    if not gt_set and not pred_set:
        return 1.0
    if not gt_set or not pred_set:
        return 0.0

    intersection = len(pred_set & gt_set)
    precision = intersection / len(pred_set) if pred_set else 0.0
    recall = intersection / len(gt_set) if gt_set else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_baseline_rationale_agreement_for_run(run_number: int):
    """Compute baseline rationale agreement for a specific run"""
    results = []

    for model_name, model_config in MODEL_CONFIGS.items():
        run_key = f'r{run_number}_dir'
        if run_key not in model_config:
            continue

        model_dir = model_config[run_key]
        if not model_dir.exists():
            print(f"Warning: Directory not found: {model_dir}")
            continue

        # Find baseline file
        baseline_files = list(model_dir.glob("baseline_*.jsonl"))
        if not baseline_files:
            print(f"Warning: No baseline file found in {model_dir}")
            continue

        baseline_file = baseline_files[0]
        print(f"  Processing {model_name} - run {run_number}: {baseline_file.name}")

        # Load baseline rationale binaries
        baseline_bins = load_baseline_binaries(baseline_file)

        if not baseline_bins:
            print(f"    Warning: No rationale binaries loaded from {baseline_file}")
            continue

        for group in GROUPS:
            gt_bins = load_group_ground_truth_binaries(group)

            # Compute F1 scores for matching QIDs
            f1_scores = []
            matched_qids = []

            for qid in gt_bins:
                if qid in baseline_bins:
                    f1 = compute_rationale_f1(baseline_bins[qid], gt_bins[qid])
                    f1_scores.append(f1)
                    matched_qids.append(qid)

            if f1_scores:
                mean_f1 = np.mean(f1_scores)
                std_f1 = np.std(f1_scores)
            else:
                mean_f1 = 0.0
                std_f1 = 0.0

            results.append({
                'model': model_name,
                'run': f'r{run_number}',
                'group': group,
                'rationale_f1_mean': mean_f1,
                'rationale_f1_std': std_f1,
                'n_evaluated': len(f1_scores),
                'total_gt': len(gt_bins)
            })

            print(f"    {group}: F1 = {mean_f1:.3f}±{std_f1:.3f} (n={len(f1_scores)}/{len(gt_bins)})")

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
        # Extract F1 values
        f1_means = [r['rationale_f1_mean'] for r in results]
        f1_stds = [r['rationale_f1_std'] for r in results]
        n_evaluateds = [r['n_evaluated'] for r in results]
        total_gts = [r['total_gt'] for r in results]

        if f1_means:
            mean_f1_mean = np.mean(f1_means)
            std_f1_mean = np.std(f1_means) if len(f1_means) > 1 else 0.0
            mean_f1_std = np.mean(f1_stds)
            mean_n_evaluated = np.mean(n_evaluateds)
            mean_total_gt = np.mean(total_gts)
        else:
            mean_f1_mean = np.nan
            std_f1_mean = np.nan
            mean_f1_std = np.nan
            mean_n_evaluated = np.nan
            mean_total_gt = np.nan

        averaged_results.append({
            'model': model,
            'group': group,
            'rationale_f1_mean_avg': mean_f1_mean,
            'rationale_f1_mean_std_across_runs': std_f1_mean,
            'rationale_f1_std_avg': mean_f1_std,
            'n_evaluated_avg': mean_n_evaluated,
            'total_gt_avg': mean_total_gt,
            'valid_runs': len(f1_means),
            'total_runs': 3
        })

    return averaged_results


def main():
    print("Computing baseline rationale agreement for CoSE dataset...")
    print("Processing individual runs and computing averages")

    all_results = []

    # Compute for each run
    for run_num in [1, 2, 3]:
        print(f"\n=== Processing Run {run_num} ===")
        run_results = compute_baseline_rationale_agreement_for_run(run_num)
        all_results.extend(run_results)

    # Save per-run results
    out_dir = THIS_DIR / "csv"
    out_dir.mkdir(exist_ok=True)

    per_run_csv = out_dir / "baseline_rationale_agreement_cose_per_run.csv"
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

    averaged_csv = out_dir / "baseline_rationale_agreement_cose_averaged.csv"
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
            if not np.isnan(result['rationale_f1_mean_avg']):
                model_stats[result['model']].append(result['rationale_f1_mean_avg'])

        for model, f1s in model_stats.items():
            if f1s:
                print(f"{model}: F1 = {np.mean(f1s):.3f}±{np.std(f1s):.3f}")

        print(f"\nSummary by Group:")
        group_stats = defaultdict(list)
        for result in averaged_results:
            if not np.isnan(result['rationale_f1_mean_avg']):
                group_stats[result['group']].append(result['rationale_f1_mean_avg'])

        for group, f1s in group_stats.items():
            if f1s:
                print(f"{group}: F1 = {np.mean(f1s):.3f}±{np.std(f1s):.3f}")

    print("\nBaseline rationale agreement analysis for CoSE completed!")


if __name__ == "__main__":
    main()