import csv
from collections import defaultdict
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
CSV_DIR = THIS_DIR / "csv"
IN_CSV = CSV_DIR / "persona_rationale_agreement_summary.csv"
OUT_CSV = CSV_DIR / "persona_rationale_agreement_aligned_summary.csv"

# Group code: {B,L,W}{O,Y}
ETH_MAP = {"B": "b", "L": "l", "W": "w"}
AGE_MAP = {"Y": 25, "O": 45}


def parse_persona_code(code: str):
    # e.g., 25_f_b -> (age:int, gender:str, eth:str)
    parts = code.split("_")
    if len(parts) != 3:
        return None
    age = int(parts[0])
    gender = parts[1]
    eth = parts[2]
    return age, gender, eth


def target_personas_for_group(group: str):
    eth_c = ETH_MAP[group[0]]
    age_c = AGE_MAP[group[1]]
    return {f"{age_c}_f_{eth_c}", f"{age_c}_m_{eth_c}"}


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {IN_CSV}")

    # Load input CSV rows
    rows = []
    with IN_CSV.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # Accumulate per model x group over aligned personas (two genders)
    acc = defaultdict(lambda: defaultdict(lambda: {"token_num": 0.0, "iou_num": 0.0, "den": 0, "missing": 0, "gt_total": 0}))

    for row in rows:
        model = row["model"]
        persona = row["persona"]
        group = row["group"]
        token_f1 = float(row["token_f1"]) if row["token_f1"] else 0.0
        iou_f1 = float(row["iou_f1"]) if row["iou_f1"] else 0.0
        N = int(row["N"]) if row["N"] else 0
        missing = int(row["missing"]) if row["missing"] else 0
        gt_total = int(row["gt_total"]) if row["gt_total"] else 0

        aligned = target_personas_for_group(group)
        if persona not in aligned:
            continue

        bucket = acc[model][group]
        bucket["token_num"] += token_f1 * N
        bucket["iou_num"] += iou_f1 * N
        bucket["den"] += N
        bucket["missing"] += missing
        bucket["gt_total"] = gt_total  # same for both genders

    # Write output
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "group", "token_f1", "iou_f1", "N", "missing", "gt_total"])
        models_order = ["gpt_oss_120b", "mistral_medium", "qwen3_32b"]
        groups_order = ["BO", "BY", "LO", "LY", "WO", "WY"]
        for model in models_order:
            for group in groups_order:
                b = acc[model][group]
                den = b["den"]
                token = (b["token_num"] / den) if den else 0.0
                iou = (b["iou_num"] / den) if den else 0.0
                w.writerow([model, group, f"{token:.6f}", f"{iou:.6f}", den, b["missing"], b["gt_total"]])

    print(f"Saved aligned summary to: {OUT_CSV}")


if __name__ == "__main__":
    main()
