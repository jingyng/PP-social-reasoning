import csv
import json
import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import f1_score


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

DATASET_ROOT = REPO_ROOT / "results"
GT_DIR = REPO_ROOT / "datasets" / "SST"

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

ETHNICITY_BY_GROUP = {
    "B": "African American",
    "L": "Hispanic",
    "W": "Caucasian",
}

AGE_BY_GROUP = {
    "Y": "25-year-old",
    "O": "45-year-old",
}


def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("\u00a0", " ").replace("\u2019", "'")
    s = re.sub(r"\s+", " ", s)
    return s


def load_group_ground_truth(group: str) -> dict:
    path = GT_DIR / f"{group}_processed.json"
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    return {item["QID"]: item["label"] for item in items}


def load_model_persona_predictions(model_dir: Path):
    """Map persona_text -> {qid -> answer}"""
    preds_by_persona = defaultdict(dict)
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

                preds_by_persona[persona_text][qid] = ans.strip()
    return preds_by_persona


def compute_persona_accuracies(preds_by_persona: dict, gt_map: dict):
    stats = {}
    total_possible = len(gt_map)
    for persona_text, qid2ans in preds_by_persona.items():
        correct = 0
        total = 0

        # For F1 calculation - collect all labels
        pred_labels = []
        true_labels = []

        # For binary F1 calculation (positive vs negative only)
        binary_pred_labels = []
        binary_true_labels = []

        for qid, gt_label in gt_map.items():
            if qid not in qid2ans:
                continue
            total += 1

            pred_label = norm(qid2ans[qid])
            true_label = norm(gt_label)

            if pred_label == true_label:
                correct += 1

            # Collect labels for 3-class F1 calculation
            pred_labels.append(pred_label)
            true_labels.append(true_label)

            # Collect labels for binary F1 calculation (exclude "no sentiment" from ground truth AND predictions)
            if true_label in ["positive", "negative"] and pred_label in ["positive", "negative"]:
                binary_pred_labels.append(pred_label)
                binary_true_labels.append(true_label)

        missing = total_possible - total
        acc = (correct / total) if total else 0.0

        # Calculate 3-class macro F1
        macro_f1_3class = 0.0
        if len(pred_labels) > 0 and len(set(true_labels)) > 1:
            try:
                macro_f1_3class = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
            except Exception as e:
                print(f"Warning: Could not compute 3-class F1 score for {persona_text}: {e}")
                macro_f1_3class = 0.0

        # Calculate binary macro F1 (positive vs negative only)
        macro_f1_binary = 0.0
        if len(binary_pred_labels) > 0 and len(set(binary_true_labels)) > 1:
            try:
                macro_f1_binary = f1_score(binary_true_labels, binary_pred_labels, average='macro', zero_division=0)
            except Exception as e:
                print(f"Warning: Could not compute binary F1 score for {persona_text}: {e}")
                macro_f1_binary = 0.0

        stats[persona_text] = {
            "accuracy": acc,
            "macro_f1_3class": macro_f1_3class,
            "macro_f1_binary": macro_f1_binary,
            "correct": correct,
            "total": total,
            "missing": missing
        }
    return stats


def persona_matches_group(persona_text: str, group: str) -> bool:
    if len(group) != 2:
        return False
    eth = ETHNICITY_BY_GROUP.get(group[0])
    age = AGE_BY_GROUP.get(group[1])
    if not eth or not age:
        return False
    return (eth in persona_text) and (age in persona_text)


def compute_persona_accuracy_for_run(run_number: int):
    """Compute persona accuracy for a specific run"""
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
        preds_by_persona = load_model_persona_predictions(model_dir)
        print(f"    Personas found: {len(preds_by_persona)}")

        for group in GROUPS:
            gt_map = load_group_ground_truth(group)
            stats = compute_persona_accuracies(preds_by_persona, gt_map)

            for persona_text, s in stats.items():
                results.append({
                    'model': model_name,
                    'run': f'r{run_number}',
                    'group': group,
                    'persona_text': persona_text,
                    'accuracy': s['accuracy'],
                    'macro_f1_3class': s['macro_f1_3class'],
                    'macro_f1_binary': s['macro_f1_binary'],
                    'correct': s['correct'],
                    'total': s['total'],
                    'missing': s['missing'],
                    'matches_group': persona_matches_group(persona_text, group)
                })

    return results


def compute_averages_across_runs(all_results: list) -> list:
    """Compute averages across runs for each model-group-persona combination"""
    # Group by model, group, and persona_text
    grouped = defaultdict(list)

    for result in all_results:
        key = (result['model'], result['group'], result['persona_text'])
        grouped[key].append(result)

    averaged_results = []

    for (model, group, persona_text), results in grouped.items():
        # Extract accuracy and F1 values
        accuracies = [r['accuracy'] for r in results]
        macro_f1s_3class = [r['macro_f1_3class'] for r in results]
        macro_f1s_binary = [r['macro_f1_binary'] for r in results]
        corrects = [r['correct'] for r in results]
        totals = [r['total'] for r in results]
        missings = [r['missing'] for r in results]
        matches_group = results[0]['matches_group']  # This should be consistent across runs

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

        averaged_results.append({
            'model': model,
            'group': group,
            'persona_text': persona_text,
            'accuracy_mean': mean_accuracy,
            'accuracy_std': std_accuracy,
            'macro_f1_3class_mean': mean_macro_f1_3class,
            'macro_f1_3class_std': std_macro_f1_3class,
            'macro_f1_binary_mean': mean_macro_f1_binary,
            'macro_f1_binary_std': std_macro_f1_binary,
            'correct_mean': mean_correct,
            'total_mean': mean_total,
            'missing_mean': mean_missing,
            'matches_group': matches_group,
            'valid_runs': len(accuracies),
            'total_runs': 3
        })

    return averaged_results


def main():
    print("Computing persona accuracy for SST dataset...")
    print("Processing individual runs and computing averages")

    all_results = []

    # Compute for each run
    for run_num in [1, 2, 3]:
        print(f"\n=== Processing Run {run_num} ===")
        run_results = compute_persona_accuracy_for_run(run_num)
        all_results.extend(run_results)

    # Save per-run results
    out_dir = THIS_DIR / "csv"
    out_dir.mkdir(exist_ok=True)

    per_run_csv = out_dir / "persona_accuracy_sst_per_run.csv"
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

    averaged_csv = out_dir / "persona_accuracy_sst_averaged.csv"
    with open(averaged_csv, "w", newline="", encoding="utf-8") as f:
        if averaged_results:
            fieldnames = averaged_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(averaged_results)

    print(f"Saved averaged results to: {averaged_csv}")

    # Create matched personas only file
    matched_results = [r for r in averaged_results if r['matches_group']]
    matched_csv = out_dir / "persona_accuracy_sst_matched_averaged.csv"
    with open(matched_csv, "w", newline="", encoding="utf-8") as f:
        if matched_results:
            fieldnames = matched_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matched_results)

    print(f"Saved matched personas results to: {matched_csv}")

    # Print summary statistics
    if matched_results:
        print(f"\nSummary by Model (matched personas only):")
        model_stats = defaultdict(list)
        for result in matched_results:
            if not np.isnan(result['accuracy_mean']):
                model_stats[result['model']].append(result['accuracy_mean'])

        for model, accuracies in model_stats.items():
            if accuracies:
                print(f"{model}: accuracy = {np.mean(accuracies)*100:.2f}±{np.std(accuracies)*100:.2f}%")

        print(f"\nSummary by Group (matched personas only):")
        group_stats = defaultdict(list)
        for result in matched_results:
            if not np.isnan(result['accuracy_mean']):
                group_stats[result['group']].append(result['accuracy_mean'])

        for group, accuracies in group_stats.items():
            if accuracies:
                print(f"{group}: accuracy = {np.mean(accuracies)*100:.2f}±{np.std(accuracies)*100:.2f}%")

        # Print individual matched personas
        print(f"\nMatched Persona Performance (averaged across runs):")
        for model in MODEL_ORDER:
            print(f"\nModel: {model}")
            model_matched = [r for r in matched_results if r['model'] == model]
            for r in sorted(model_matched, key=lambda x: (x['group'], x['persona_text'])):
                print(f"  {r['persona_text']} [{r['group']}] → {r['accuracy_mean']*100:.2f}±{r['accuracy_std']*100:.2f}%")

    print("\nPersona accuracy analysis for SST completed!")


if __name__ == "__main__":
    main()