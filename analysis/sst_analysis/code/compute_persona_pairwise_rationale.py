import csv
import json
from ast import literal_eval
from pathlib import Path
from typing import Dict, List, Tuple


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
DATASET_ROOT = REPO_ROOT / "datasets"

MODEL_DIRS = {
    "gpt_oss_120b": DATASET_ROOT / "results_gpt_oss_120b_cose_merged",
    "mistral_medium": DATASET_ROOT / "results_mistral_medium_cose_merged",
    "qwen3_32b": DATASET_ROOT / "results_qwen3_32b_cose_merged",
}

# 12 personas (age in {25,45} x gender in {f,m} x ethnicity in {b,l,w})
PERSONA_CODES = [
    f"{age}_{gender}_{eth}"
    for age in (25, 45)
    for gender in ("f", "m")
    for eth in ("b", "l", "w")
]


def _to_list_int(x):
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


def prf1_for_pair(a: List[int], b: List[int]) -> Tuple[float, float, float]:
    n = min(len(a), len(b))
    pa = a[:n]
    pb = b[:n]
    tp = sum(1 for i in range(n) if pa[i] == 1 and pb[i] == 1)
    fp = sum(1 for i in range(n) if pa[i] == 1 and pb[i] == 0)
    fn = sum(1 for i in range(n) if pa[i] == 0 and pb[i] == 1)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def iou_for_pair(a: List[int], b: List[int]) -> float:
    n = min(len(a), len(b))
    pa = a[:n]
    pb = b[:n]
    inter = sum(1 for i in range(n) if pa[i] == 1 and pb[i] == 1)
    union = sum(1 for i in range(n) if pa[i] == 1 or pb[i] == 1)
    if union == 0:
        return 1.0
    return inter / union


def load_model_persona_bins(model_dir: Path) -> Dict[str, Dict[str, List[int]]]:
    per_persona: Dict[str, Dict[str, List[int]]] = {code: {} for code in PERSONA_CODES}
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
                if "___" not in persona_id:
                    continue
                _, code = persona_id.split("___", 1)
                if code.endswith("_baseline") or code not in per_persona:
                    continue
                qid = obj.get("question_id") or obj.get("QID")
                rb = obj.get("rationale_binary")
                if not qid or rb is None:
                    continue
                try:
                    per_persona[code][qid] = _to_list_int(rb)
                except Exception:
                    continue
    return per_persona


def aggregate_pairwise(a_map: Dict[str, List[int]], b_map: Dict[str, List[int]]):
    # Aggregate over shared QIDs
    shared = sorted(set(a_map.keys()) & set(b_map.keys()))
    if not shared:
        return 0.0, 0.0, 0
    f1_sum = 0.0
    iou_hits = 0
    n = 0
    for qid in shared:
        a = a_map[qid]
        b = b_map[qid]
        _, _, f1 = prf1_for_pair(a, b)
        f1_sum += f1
        iou = iou_for_pair(a, b)
        if iou >= 0.5:
            iou_hits += 1
        n += 1
    return (f1_sum / n) if n else 0.0, (iou_hits / n) if n else 0.0, n


def main():
    missing_dirs = [str(path) for path in MODEL_DIRS.values() if not path.exists()]
    if missing_dirs:
        raise FileNotFoundError("Missing model directories: " + ", ".join(missing_dirs))

    rows = []
    for model, model_dir in MODEL_DIRS.items():
        per_persona = load_model_persona_bins(model_dir)
        codes = PERSONA_CODES
        for i, pa in enumerate(codes):
            for j, pb in enumerate(codes):
                a_map = per_persona.get(pa, {})
                b_map = per_persona.get(pb, {})
                if pa == pb:
                    # Perfect agreement on diagonal (computed from available QIDs)
                    # Still compute to capture any length mismatches
                    tok_f1, iou_f1, n = aggregate_pairwise(a_map, b_map)
                else:
                    tok_f1, iou_f1, n = aggregate_pairwise(a_map, b_map)
                rows.append({
                    "model": model,
                    "persona_a": pa,
                    "persona_b": pb,
                    "token_f1": f"{tok_f1:.6f}",
                    "iou_f1": f"{iou_f1:.6f}",
                    "N": n,
                })

    out_dir = THIS_DIR / "csv"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "persona_pairwise_rationale_agreement.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "persona_a", "persona_b", "token_f1", "iou_f1", "N"])
        w.writeheader()
        rows.sort(key=lambda r: (r["model"], r["persona_a"], r["persona_b"]))
        w.writerows(rows)

    print(f"Saved pairwise persona rationale agreement to: {out_csv}")


if __name__ == "__main__":
    main()
