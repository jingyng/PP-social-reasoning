import csv
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
CSV_DIR = THIS_DIR / "csv"
IN_CSV = CSV_DIR / "persona_rationale_agreement_summary.csv"
OUT_CSV = CSV_DIR / "persona_rationale_agreement_aligned_rows.csv"

ETH_MAP = {"B": "b", "L": "l", "W": "w"}
AGE_MAP = {"Y": 25, "O": 45}


def aligned_personas_for_group(group: str):
    eth = ETH_MAP[group[0]]
    age = AGE_MAP[group[1]]
    return {f"{age}_f_{eth}", f"{age}_m_{eth}"}


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {IN_CSV}")

    with IN_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    groups = ["BO", "BY", "LO", "LY", "WO", "WY"]
    filtered = []

    for row in rows:
        group = row["group"]
        if group not in groups:
            continue
        persona = row["persona"]
        if persona in aligned_personas_for_group(group):
            filtered.append(row)

    # Keep same columns and write output
    fieldnames = rows[0].keys() if rows else [
        "model", "persona", "group", "token_f1", "iou_f1", "N", "missing", "gt_total"
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Sort for readability: model, group, persona
        filtered.sort(key=lambda r: (r["model"], r["group"], r["persona"]))
        writer.writerows(filtered)

    print(f"Saved aligned rows to: {OUT_CSV}")


if __name__ == "__main__":
    main()
