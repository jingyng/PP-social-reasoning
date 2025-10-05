import csv
import json
import re
from collections import defaultdict
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

DATASET_ROOT = REPO_ROOT / "datasets"
GT_DIR = REPO_ROOT / "data" / "SST"

MODEL_DIRS = {
    "gpt_oss_120b": DATASET_ROOT / "results_gpt_oss_120b_sst_merged",
    "mistral_medium": DATASET_ROOT / "results_mistral_medium_sst_merged",
    "qwen3_32b": DATASET_ROOT / "results_qwen3_32b_sst_merged",
}

MODEL_ORDER = list(MODEL_DIRS.keys())
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
    # Map persona_text -> {qid -> answer}
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
        for qid, gt_label in gt_map.items():
            if qid not in qid2ans:
                continue
            total += 1
            if norm(qid2ans[qid]) == norm(gt_label):
                correct += 1
        missing = total_possible - total
        acc = (correct / total) if total else 0.0
        stats[persona_text] = {"accuracy": acc, "correct": correct, "total": total, "missing": missing}
    return stats


def persona_matches_group(persona_text: str, group: str) -> bool:
    if len(group) != 2:
        return False
    eth = ETHNICITY_BY_GROUP.get(group[0])
    age = AGE_BY_GROUP.get(group[1])
    if not eth or not age:
        return False
    return (eth in persona_text) and (age in persona_text)


def main():
    missing_dirs = [str(path) for path in MODEL_DIRS.values() if not path.exists()]
    if missing_dirs:
        raise FileNotFoundError("Model result folders missing: " + ", ".join(missing_dirs))

    out_dir = THIS_DIR / "csv"
    out_dir.mkdir(exist_ok=True)
    all_rows_path = out_dir / "persona_accuracy_summary.csv"
    matched_rows_path = out_dir / "persona_accuracy_matched.csv"

    all_rows = []
    matched_rows = []

    for model in MODEL_ORDER:
        model_dir = MODEL_DIRS[model]
        print(f"Processing model: {model}")
        preds_by_persona = load_model_persona_predictions(model_dir)
        print(f"  Personas found: {len(preds_by_persona)}")

        for group in GROUPS:
            gt_map = load_group_ground_truth(group)
            stats = compute_persona_accuracies(preds_by_persona, gt_map)

            for persona_text, s in stats.items():
                row = {
                    "model": model,
                    "group": group,
                    "persona_text": persona_text,
                    "accuracy": f"{s['accuracy']:.4f}",
                    "correct": s["correct"],
                    "total": s["total"],
                    "missing": s["missing"],
                }
                all_rows.append(row)
                if persona_matches_group(persona_text, group):
                    matched_rows.append(row)

    fieldnames = ["model", "group", "persona_text", "accuracy", "correct", "total", "missing"]

    with all_rows_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    with matched_rows_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matched_rows)

    print(f"Saved per-persona accuracy summary to: {all_rows_path}")
    print(f"Saved matched persona accuracy rows to: {matched_rows_path}")

    # Console summary for the 12 personas (matched rows only)
    print("\nPersona accuracy (matched personas per group):")
    for model in MODEL_ORDER:
        print(f"\nModel: {model}")
        rows = [r for r in matched_rows if r["model"] == model]
        for r in sorted(rows, key=lambda x: (x["group"], x["persona_text"])):
            print(
                f"  {r['persona_text']} [{r['group']}] → {float(r['accuracy'])*100:.2f}%"
                f" (correct {r['correct']}/{r['total']}, missing {r['missing']})"
            )


if __name__ == "__main__":
    main()
