import csv
import json
import numpy as np
from collections import defaultdict
from ast import literal_eval
from pathlib import Path
from typing import Dict, List, Tuple


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

DATASET_ROOT = REPO_ROOT / "results"
GT_DIR = REPO_ROOT / "datasets" / "cose"

# Model configurations for CoSE
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

# CoSE persona groups
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


def load_ground_truth_rationales(group: str) -> Dict[str, List[int]]:
    """Load ground truth rationales for a specific group"""
    gt_file = GT_DIR / f"{group}_processed_with_queries.json"
    gt_rationales = {}

    with open(gt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            qid = item['QID']
            rb = item.get('rationale_binary')
            if rb is not None:
                try:
                    gt_rationales[qid] = _to_list_int(rb)
                except Exception:
                    continue

    return gt_rationales


def load_model_persona_predictions(model_dir: Path) -> Dict[str, Dict[str, List[int]]]:
    """Load predictions from a model directory"""
    per_persona = defaultdict(dict)

    for path in sorted(model_dir.glob("QID*.jsonl")):
        question_id = path.stem  # e.g., 'QID1'

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
                # Extract persona code from persona_id (e.g., "QID1___25_m_b" -> "25_m_b")
                if "___" in persona_id:
                    _, persona_code = persona_id.split("___", 1)
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
    # Try multiple possible baseline filenames
    possible_names = [
        f"baseline_{model_name}_cose.jsonl",
        f"baseline_{model_name.split('_')[0]}_cose.jsonl",  # e.g., mistral_cose.jsonl
        f"baseline_{model_name}.jsonl",
        f"baseline_cose.jsonl"
    ]

    baseline_file = None
    for name in possible_names:
        test_path = model_dir / name
        if test_path.exists():
            baseline_file = test_path
            break

    baseline_predictions = {}

    if baseline_file is None:
        print(f"  Warning: Baseline file not found in: {model_dir}")
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

            question_id = obj.get("question_id") or obj.get("QID")
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


def compute_baseline_performance_for_run(model_name: str, run_name: str, model_dir: Path, gt_rationales_by_group: Dict[str, Dict]):
    """Compute baseline rationale performance for one run"""
    print(f"  Processing {model_name} baseline - {run_name}")

    # Load baseline predictions
    baseline_predictions = load_baseline_predictions(model_dir, model_name)

    results = []

    for group in GROUPS:
        gt_rationales = gt_rationales_by_group[group]

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

        results.append({
            'model': model_name,
            'run': run_name,
            'group': group,
            'token_f1': token_f1,
            'iou_f1': iou_f1,
            'n_eval': n_eval,
            'missing': missing,
            'gt_total': len(gt_rationales)
        })

    return results


def compute_persona_performance_for_run(model_name: str, run_name: str, model_dir: Path, gt_rationales_by_group: Dict[str, Dict]):
    """Compute persona rationale performance for one run"""
    print(f"  Processing {model_name} personas - {run_name}")

    # Load model predictions
    per_persona = load_model_persona_predictions(model_dir)

    results = []

    # Process each group
    for group in GROUPS:
        gt_rationales = gt_rationales_by_group[group]

        # Process each persona
        for persona_code in per_persona.keys():
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
                'run': run_name,
                'group': group,
                'persona_code': persona_code,
                'token_f1': token_f1,
                'iou_f1': iou_f1,
                'n_eval': n_eval,
                'missing': missing,
                'gt_total': len(gt_rationales)
            })

    return results


def compute_averages(all_results: List[Dict], is_baseline: bool = False) -> List[Dict]:
    """Compute averages across runs for each model-group-(persona) combination"""
    # Group by model, group, (persona_code if not baseline)
    grouped = defaultdict(list)

    for result in all_results:
        if is_baseline:
            key = (result['model'], result['group'])
        else:
            key = (result['model'], result['group'], result['persona_code'])
        grouped[key].append(result)

    averaged_results = []

    for key, results in grouped.items():
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

        if is_baseline:
            model, group = key
            averaged_results.append({
                'model': model,
                'group': group,
                'token_f1_mean': avg_token_f1,
                'token_f1_std': std_token_f1,
                'iou_f1_mean': avg_iou_f1,
                'iou_f1_std': std_iou_f1,
                'n_eval_mean': avg_n_eval,
                'gt_total': results[0]['gt_total']
            })
        else:
            model, group, persona_code = key
            averaged_results.append({
                'model': model,
                'group': group,
                'persona_code': persona_code,
                'token_f1_mean': avg_token_f1,
                'token_f1_std': std_token_f1,
                'iou_f1_mean': avg_iou_f1,
                'iou_f1_std': std_iou_f1,
                'n_eval_mean': avg_n_eval,
                'gt_total': results[0]['gt_total']
            })

    return averaged_results


def main():
    print("Computing CoSE Token-F1 and IOU-F1 performance...")

    # Load ground truth rationales for all groups
    print("Loading ground truth rationales...")
    gt_rationales_by_group = {}
    for group in GROUPS:
        gt_rationales_by_group[group] = load_ground_truth_rationales(group)
        print(f"  {group}: {len(gt_rationales_by_group[group])} samples")

    all_baseline_results = []
    all_persona_results = []

    # Process each model and run
    for model_name, model_config in MODEL_CONFIGS.items():
        print(f"\nProcessing model: {model_name}")

        for run_name, run_dir in [('r1', model_config['r1_dir']),
                                  ('r2', model_config['r2_dir']),
                                  ('r3', model_config['r3_dir'])]:
            if not run_dir.exists():
                print(f"  Warning: Directory not found: {run_dir}")
                continue

            # Process baseline results
            baseline_results = compute_baseline_performance_for_run(
                model_name, run_name, run_dir, gt_rationales_by_group
            )
            all_baseline_results.extend(baseline_results)

            # Process persona results
            persona_results = compute_persona_performance_for_run(
                model_name, run_name, run_dir, gt_rationales_by_group
            )
            all_persona_results.extend(persona_results)

    # Save per-run results
    output_dir = THIS_DIR / "csv"
    output_dir.mkdir(exist_ok=True)

    # Baseline per-run results
    baseline_per_run_csv = output_dir / "baseline_token_iou_f1_cose_per_run.csv"
    with open(baseline_per_run_csv, "w", newline="", encoding="utf-8") as f:
        if all_baseline_results:
            fieldnames = all_baseline_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_baseline_results)

    print(f"\nSaved baseline per-run results to: {baseline_per_run_csv}")

    # Persona per-run results
    persona_per_run_csv = output_dir / "persona_token_iou_f1_cose_per_run.csv"
    with open(persona_per_run_csv, "w", newline="", encoding="utf-8") as f:
        if all_persona_results:
            fieldnames = all_persona_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_persona_results)

    print(f"Saved persona per-run results to: {persona_per_run_csv}")

    # Compute and save averaged results
    print("\nComputing averages across runs...")
    averaged_baseline_results = compute_averages(all_baseline_results, is_baseline=True)
    averaged_persona_results = compute_averages(all_persona_results, is_baseline=False)

    # Baseline averaged results
    baseline_averaged_csv = output_dir / "baseline_token_iou_f1_cose_averaged.csv"
    with open(baseline_averaged_csv, "w", newline="", encoding="utf-8") as f:
        if averaged_baseline_results:
            fieldnames = averaged_baseline_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(averaged_baseline_results)

    print(f"Saved baseline averaged results to: {baseline_averaged_csv}")

    # Persona averaged results
    persona_averaged_csv = output_dir / "persona_token_iou_f1_cose_averaged.csv"
    with open(persona_averaged_csv, "w", newline="", encoding="utf-8") as f:
        if averaged_persona_results:
            fieldnames = averaged_persona_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(averaged_persona_results)

    print(f"Saved persona averaged results to: {persona_averaged_csv}")

    # Print summary statistics
    if averaged_baseline_results:
        print(f"\nBaseline Results Summary:")
        model_stats = defaultdict(list)
        for result in averaged_baseline_results:
            model_stats[result['model']].append(result)

        for model, results in model_stats.items():
            token_f1_values = [r['token_f1_mean'] for r in results]
            iou_f1_values = [r['iou_f1_mean'] for r in results]
            print(f"{model} baseline: Token-F1 = {np.mean(token_f1_values):.3f}±{np.std(token_f1_values):.3f}, "
                  f"IOU-F1 = {np.mean(iou_f1_values):.3f}±{np.std(iou_f1_values):.3f}")

    print("\nCoSE Token-F1 and IOU-F1 computation completed!")


if __name__ == "__main__":
    main()
