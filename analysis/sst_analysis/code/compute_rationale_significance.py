import os
import json
import csv
from ast import literal_eval
from typing import Dict, List, Tuple
import math


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_BASELINES = {
    "gpt_oss_120b": os.path.join(
        REPO_ROOT, "datasets", "results_gpt_oss_120b_cose_merged", "baseline_gpt_oss_120b_cose.jsonl"
    ),
    "mistral_medium": os.path.join(
        REPO_ROOT, "datasets", "results_mistral_medium_cose_merged", "baseline_mistral_cose.jsonl"
    ),
    "qwen3_32b": os.path.join(
        REPO_ROOT, "datasets", "results_qwen3_32b_cose_merged", "baseline_qwen3_32b_cose.jsonl"
    ),
}

MODEL_DIRS = {
    "gpt_oss_120b": os.path.join(REPO_ROOT, "datasets", "results_gpt_oss_120b_cose_merged"),
    "mistral_medium": os.path.join(REPO_ROOT, "datasets", "results_mistral_medium_cose_merged"),
    "qwen3_32b": os.path.join(REPO_ROOT, "datasets", "results_qwen3_32b_cose_merged"),
}

GT_DIR = os.path.join(REPO_ROOT, "data", "cose")
GROUPS = ["BO", "BY", "LO", "LY", "WO", "WY"]

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


def load_group_gt_binaries(group: str) -> Dict[str, List[int]]:
    path = os.path.join(GT_DIR, f"{group}_processed_with_queries.json")
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    out = {}
    for item in items:
        qid = item["QID"]
        rb = item.get("rationale_binary")
        if rb is None:
            continue
        try:
            out[qid] = _to_list_int(rb)
        except Exception:
            continue
    return out


def prf1_for_pair(pred: List[int], gold: List[int]) -> float:
    n = min(len(pred), len(gold))
    p = pred[:n]
    g = gold[:n]
    tp = sum(1 for i in range(n) if p[i] == 1 and g[i] == 1)
    fp = sum(1 for i in range(n) if p[i] == 1 and g[i] == 0)
    fn = sum(1 for i in range(n) if p[i] == 0 and g[i] == 1)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def iou_for_pair(pred: List[int], gold: List[int]) -> float:
    n = min(len(pred), len(gold))
    p = pred[:n]
    g = gold[:n]
    inter = sum(1 for i in range(n) if p[i] == 1 and g[i] == 1)
    union = sum(1 for i in range(n) if p[i] == 1 or g[i] == 1)
    return 1.0 if union == 0 else inter / union


def load_binaries_from_jsonl(path: str) -> Dict[str, List[int]]:
    qid_to_bin = {}
    with open(path, "r", encoding="utf-8") as f:
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
            if qid is None or rb is None:
                continue
            try:
                qid_to_bin[qid] = _to_list_int(rb)
            except Exception:
                continue
    return qid_to_bin


def load_model_persona_bins(model_dir: str) -> Dict[str, Dict[str, List[int]]]:
    per_persona = {code: {} for code in PERSONA_CODES}
    for name in sorted(os.listdir(model_dir)):
        if not name.startswith("QID") or not name.endswith(".jsonl"):
            continue
        path = os.path.join(model_dir, name)
        with open(path, "r", encoding="utf-8") as f:
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
                if code.endswith("_baseline"):
                    continue
                if code not in per_persona:
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


def normal_approx_sign_test(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    # Continuity correction
    mean = n / 2.0
    sd = math.sqrt(n * 0.25)
    z = (k - mean) / sd
    # two-sided
    # Phi(z) via erfc: Phi(z) = 0.5 * erfc(-z/sqrt(2))
    def phi(x):
        return 0.5 * math.erfc(-x / math.sqrt(2.0))
    p_one_tail = 1.0 - phi(abs(z))
    return max(0.0, min(1.0, 2.0 * p_one_tail))


def paired_significance(baseline_vals: List[float], persona_vals: List[float]) -> Tuple[float, float, int]:
    # Compute paired differences and sign test p-value
    diffs = [pv - bv for pv, bv in zip(persona_vals, baseline_vals)]
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0, 0.0, 0
    k_pos = sum(1 for d in nonzero if d > 0)
    p = normal_approx_sign_test(k_pos, n)
    effect = sum(nonzero) / n
    return p, effect, n


def main():
    gt_by_group = {g: load_group_gt_binaries(g) for g in GROUPS}

    out_rows = []

    for model, base_path in MODEL_BASELINES.items():
        base_map = load_binaries_from_jsonl(base_path)
        per_persona = load_model_persona_bins(MODEL_DIRS[model])

        for persona_code, persona_map in per_persona.items():
            for group in GROUPS:
                gt_map = gt_by_group[group]
                # gather per-QID metrics for both baseline and persona
                tok_base = []
                tok_pers = []
                iou_base = []
                iou_pers = []
                for qid, gold in gt_map.items():
                    pb = base_map.get(qid)
                    pp = persona_map.get(qid)
                    if pb is None or pp is None:
                        continue
                    tok_base.append(prf1_for_pair(pb, gold))
                    tok_pers.append(prf1_for_pair(pp, gold))
                    iou_base.append(iou_for_pair(pb, gold))
                    iou_pers.append(iou_for_pair(pp, gold))

                p_tok, eff_tok, n_tok = paired_significance(tok_base, tok_pers)
                p_iou, eff_iou, n_iou = paired_significance(iou_base, iou_pers)

                out_rows.append({
                    "model": model,
                    "persona": persona_code,
                    "group": group,
                    "p_token_f1": f"{p_tok:.6g}",
                    "effect_token_f1": f"{eff_tok:.6f}",
                    "n_token": n_tok,
                    "p_iou_f1": f"{p_iou:.6g}",
                    "effect_iou_f1": f"{eff_iou:.6f}",
                    "n_iou": n_iou,
                })

    out_csv = os.path.join(REPO_ROOT, "3_analysis", "rationale_significance.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model", "persona", "group",
                "p_token_f1", "effect_token_f1", "n_token",
                "p_iou_f1", "effect_iou_f1", "n_iou",
            ],
        )
        w.writeheader()
        out_rows.sort(key=lambda r: (r["model"], r["group"], r["persona"]))
        w.writerows(out_rows)

    print(f"Saved significance results to: {out_csv}")


if __name__ == "__main__":
    main()

