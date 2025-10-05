import csv
import json
import numpy as np
from collections import defaultdict
from ast import literal_eval
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

DATASET_ROOT = REPO_ROOT / "results"

# Model configurations for HatEXplain
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
    'Education': ['nfe', 'hs', 'he'],  # no_formal_education, high_school, higher_education
    'Gender': ['m', 'f'],  # male, female
    'Loneliness': ['nl', 'sl'],  # not_lonely, somewhat_lonely
    'Political': ['l', 'r', 'c'],  # left-wing, right-wing, centrist
    'Race': ['w', 'b', 'a'],  # white, black, asian
    'Religion': ['chr', 'mus', 'jew', 'ath', 'hin']  # christian, muslim, jewish, atheist, hindu
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


def load_ground_truth_rationales_filtered():
    """Load filtered ground truth rationales (offensive + hate speech only)"""
    gt_file = THIS_DIR / "ground_truth_rationales_filtered.jsonl"
    gt_rationales = {}

    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            gt_rationales[data['id']] = data['majority_rationale']

    return gt_rationales


def load_model_persona_predictions(model_dir: Path) -> Dict[str, Dict[str, List[int]]]:
    """Load predictions from a model directory"""
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


def load_baseline_predictions(model_dir: Path, model_name: str) -> Dict[str, List[int]]:
    """Load baseline predictions from a model directory"""
    baseline_file = model_dir / f"baseline_{model_name}.jsonl"
    baseline_predictions = {}

    if not baseline_file.exists():
        print(f"  Warning: Baseline file not found: {baseline_file}")
        return baseline_predictions

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
                baseline_predictions[question_id] = _to_list_int(rb)
            except Exception:
                continue

    return baseline_predictions


def prf1_for_pair(pred: List[int], gold: List[int]) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 for a pair of rationale sequences"""
    n = min(len(pred), len(gold))
    p = pred[:n]
    g = gold[:n]
    tp = sum(1 for i in range(n) if p[i] == 1 and g[i] == 1)
    fp = sum(1 for i in range(n) if p[i] == 1 and g[i] == 0)
    fn = sum(1 for i in range(n) if p[i] == 0 and g[i] == 1)

    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def iou_for_pair(pred: List[int], gold: List[int]) -> float:
    """Calculate IoU (Intersection over Union) for a pair of rationale sequences"""
    n = min(len(pred), len(gold))
    p = pred[:n]
    g = gold[:n]
    inter = sum(1 for i in range(n) if p[i] == 1 and g[i] == 1)
    union = sum(1 for i in range(n) if p[i] == 1 or g[i] == 1)
    return 1.0 if union == 0 else inter / union


def compute_baseline_performance_for_run_filtered(model_name: str, model_dir: Path, gt_rationales: Dict):
    """Compute baseline rationale performance for one run (filtered data)"""
    print(f"  Processing {model_name} baseline - {model_dir.name} (filtered)")

    # Load baseline predictions
    baseline_predictions = load_baseline_predictions(model_dir, model_name)

    f1_sum = 0.0
    iou_hits = 0
    n_eval = 0
    iou_total = 0

    for question_id, gold_rationale in gt_rationales.items():
        pred_rationale = baseline_predictions.get(question_id)
        if pred_rationale is None:
            continue

        # Calculate token F1
        _, _, f1 = prf1_for_pair(pred_rationale, gold_rationale)
        f1_sum += f1
        n_eval += 1

        # Calculate IoU F1 (binary: IoU >= 0.5)
        iou = iou_for_pair(pred_rationale, gold_rationale)
        if iou >= 0.5:
            iou_hits += 1
        iou_total += 1

    token_f1 = (f1_sum / n_eval) if n_eval else 0.0
    iou_f1 = (iou_hits / iou_total) if iou_total else 0.0
    missing = len(gt_rationales) - n_eval

    return {
        'model': model_name,
        'run': model_dir.name,
        'group': 'baseline',
        'persona': 'baseline',
        'token_f1': token_f1,
        'iou_f1': iou_f1,
        'n_eval': n_eval,
        'missing': missing,
        'gt_total': len(gt_rationales)
    }


def compute_rationale_performance_for_run_filtered(model_name: str, model_dir: Path, gt_rationales: Dict):
    """Compute rationale performance for one run (filtered data)"""
    print(f"  Processing {model_name} - {model_dir.name} (filtered)")

    # Load model predictions
    per_persona = load_model_persona_predictions(model_dir)

    results = []

    # Process each persona group
    for group_name, persona_codes in PERSONA_GROUPS.items():
        for persona_code in persona_codes:
            if persona_code not in per_persona:
                # No predictions for this persona
                results.append({
                    'model': model_name,
                    'run': model_dir.name,
                    'group': group_name,
                    'persona': persona_code,
                    'token_f1': 0.0,
                    'iou_f1': 0.0,
                    'n_eval': 0,
                    'missing': len(gt_rationales),
                    'gt_total': len(gt_rationales)
                })
                continue

            persona_predictions = per_persona[persona_code]

            f1_sum = 0.0
            iou_hits = 0
            n_eval = 0
            iou_total = 0

            for question_id, gold_rationale in gt_rationales.items():
                pred_rationale = persona_predictions.get(question_id)
                if pred_rationale is None:
                    continue

                # Calculate token F1
                _, _, f1 = prf1_for_pair(pred_rationale, gold_rationale)
                f1_sum += f1
                n_eval += 1

                # Calculate IoU F1 (binary: IoU >= 0.5)
                iou = iou_for_pair(pred_rationale, gold_rationale)
                if iou >= 0.5:
                    iou_hits += 1
                iou_total += 1

            token_f1 = (f1_sum / n_eval) if n_eval else 0.0
            iou_f1 = (iou_hits / iou_total) if iou_total else 0.0
            missing = len(gt_rationales) - n_eval

            results.append({
                'model': model_name,
                'run': model_dir.name,
                'group': group_name,
                'persona': persona_code,
                'token_f1': token_f1,
                'iou_f1': iou_f1,
                'n_eval': n_eval,
                'missing': missing,
                'gt_total': len(gt_rationales)
            })

    return results


def compute_averages(all_results: List[Dict]) -> List[Dict]:
    """Compute averages across runs for each model-group-persona combination"""
    # Group by model, group, persona
    grouped = defaultdict(list)

    for result in all_results:
        key = (result['model'], result['group'], result['persona'])
        grouped[key].append(result)

    averaged_results = []

    for (model, group, persona), results in grouped.items():
        if not results:
            continue

        token_f1_values = [r['token_f1'] for r in results]
        iou_f1_values = [r['iou_f1'] for r in results]
        n_eval_values = [r['n_eval'] for r in results]

        avg_token_f1 = np.mean(token_f1_values)
        avg_iou_f1 = np.mean(iou_f1_values)
        avg_n_eval = np.mean(n_eval_values)

        std_token_f1 = np.std(token_f1_values)
        std_iou_f1 = np.std(iou_f1_values)

        averaged_results.append({
            'model': model,
            'group': group,
            'persona': persona,
            'token_f1_mean': avg_token_f1,
            'token_f1_std': std_token_f1,
            'iou_f1_mean': avg_iou_f1,
            'iou_f1_std': std_iou_f1,
            'n_eval_mean': avg_n_eval,
            'gt_total': results[0]['gt_total']
        })

    return averaged_results


def main():
    print("Computing filtered rationale performance (offensive + hate speech only)...")

    # Load filtered ground truth rationales
    print("Loading filtered ground truth rationales...")
    gt_rationales = load_ground_truth_rationales_filtered()
    print(f"Loaded {len(gt_rationales)} filtered ground truth rationales")

    # Check that all model directories exist
    missing_dirs = []
    for model_config in MODEL_CONFIGS.values():
        for run_dir in [model_config['r1_dir'], model_config['r2_dir'], model_config['r3_dir']]:
            if not run_dir.exists():
                missing_dirs.append(str(run_dir))

    if missing_dirs:
        raise FileNotFoundError("Missing model directories: " + ", ".join(missing_dirs))

    all_results = []
    baseline_results = []

    # Process each model and run
    for model_name, model_config in MODEL_CONFIGS.items():
        print(f"\nProcessing model: {model_name}")

        for run_name, run_dir in [('r1', model_config['r1_dir']),
                                  ('r2', model_config['r2_dir']),
                                  ('r3', model_config['r3_dir'])]:
            # Process persona results
            results = compute_rationale_performance_for_run_filtered(model_name, run_dir, gt_rationales)
            all_results.extend(results)

            # Process baseline results
            baseline_result = compute_baseline_performance_for_run_filtered(model_name, run_dir, gt_rationales)
            baseline_results.append(baseline_result)

    # Save per-run results
    output_dir = THIS_DIR / "csv"
    output_dir.mkdir(exist_ok=True)

    per_run_csv = output_dir / "persona_rationale_performance_per_run_filtered.csv"
    with open(per_run_csv, "w", newline="", encoding="utf-8") as f:
        if all_results:
            fieldnames = all_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    print(f"\nSaved filtered per-run results to: {per_run_csv}")

    # Save baseline per-run results
    baseline_per_run_csv = output_dir / "baseline_rationale_performance_per_run_filtered.csv"
    with open(baseline_per_run_csv, "w", newline="", encoding="utf-8") as f:
        if baseline_results:
            fieldnames = baseline_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(baseline_results)

    print(f"Saved filtered baseline per-run results to: {baseline_per_run_csv}")

    # Compute and save averaged results
    print("Computing averages across runs...")
    averaged_results = compute_averages(all_results)
    averaged_baseline_results = compute_averages(baseline_results)

    averaged_csv = output_dir / "persona_rationale_performance_averaged_filtered.csv"
    with open(averaged_csv, "w", newline="", encoding="utf-8") as f:
        if averaged_results:
            fieldnames = averaged_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(averaged_results)

    print(f"Saved filtered averaged results to: {averaged_csv}")

    # Save baseline averaged results
    baseline_averaged_csv = output_dir / "baseline_rationale_performance_averaged_filtered.csv"
    with open(baseline_averaged_csv, "w", newline="", encoding="utf-8") as f:
        if averaged_baseline_results:
            fieldnames = averaged_baseline_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(averaged_baseline_results)

    print(f"Saved filtered baseline averaged results to: {baseline_averaged_csv}")

    # Print summary statistics
    if averaged_results:
        print(f"\nFiltered Persona Results Summary:")
        print(f"Total model-group-persona combinations: {len(averaged_results)}")

        # Group by model
        model_stats = defaultdict(list)
        for result in averaged_results:
            model_stats[result['model']].append(result)

        for model, results in model_stats.items():
            token_f1_values = [r['token_f1_mean'] for r in results]
            iou_f1_values = [r['iou_f1_mean'] for r in results]
            print(f"{model} personas: Token-F1 = {np.mean(token_f1_values):.3f}±{np.std(token_f1_values):.3f}, "
                  f"IoU-F1 = {np.mean(iou_f1_values):.3f}±{np.std(iou_f1_values):.3f}")

    if averaged_baseline_results:
        print(f"\nFiltered Baseline Results Summary:")
        for result in averaged_baseline_results:
            print(f"{result['model']} baseline: Token-F1 = {result['token_f1_mean']:.3f}±{result['token_f1_std']:.3f}, "
                  f"IoU-F1 = {result['iou_f1_mean']:.3f}±{result['iou_f1_std']:.3f}")

    print("\nFiltered rationale performance computation completed!")


if __name__ == "__main__":
    main()