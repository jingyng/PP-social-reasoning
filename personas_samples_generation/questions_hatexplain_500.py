import os
import json
import random
import requests
from collections import Counter, OrderedDict

HATEXPLAIN_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
CACHE_PATH = "../datasets/hatexplain/cache_hatexplain_dataset.json"

LABEL_MAP = {
    "hatespeech": "Hate speech",
    "offensive": "Offensive language",
    "normal": "Normal"
}
CLASS_ORDER = ["Hate speech", "Offensive language", "Normal"]

# --------------- IO / caching ---------------
def load_hatexplain_cached(cache_path=CACHE_PATH):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    resp = requests.get(HATEXPLAIN_URL, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data

# --------------- helpers ---------------
def is_three_zero(labels):
    c = Counter(labels)
    return len(c) == 1  # all three the same

def normalize_label(lbl):
    if not lbl:
        return None
    return LABEL_MAP.get(lbl.strip().lower())

def _is_binary_vec(vec, L):
    return isinstance(vec, list) and len(vec) == L and all((x == 0 or x == 1) for x in vec)

def is_two_vs_one(labels):
    c = Counter(labels)
    return len(c) == 2 and sorted(c.values()) == [1, 2]

def majority_label_from(labels):
    c = Counter(labels)
    return max(CLASS_ORDER, key=lambda l: (c[l], -CLASS_ORDER.index(l)))

def distribution_for_normal_any(data):
    """Counts Normal 3-0 / 2-1 (majority) / 1-2 (minority) ignoring rationale availability."""
    counts = {"3-0": 0, "2-1_majority": 0, "1-2_minority": 0}
    for _, item in data.items():
        ann = item.get("annotators", [])
        if len(ann) != 3:
            continue
        labels = [normalize_label(a.get("label")) for a in ann]
        if any(l is None for l in labels):
            continue
        c = Counter(labels)
        n = c.get("Normal", 0)
        if n == 3:
            counts["3-0"] += 1
        elif n == 2:
            counts["2-1_majority"] += 1
        elif n == 1:
            counts["1-2_minority"] += 1
    print("Normal label distribution (ignoring rationale availability):")
    for k, v in counts.items():
        print(f"  {k}: {v}")

# --------------- HYBRID policy core ---------------
def build_pool_hybrid(data, include_unanimous=False):
    """
    Build pool with this policy:
      - Always keep 2-vs-1 items with HYBRID rules (same as before):
          * If count(Normal) == 2  -> RELAX: do not require masks.
          * If count(Normal) in {0,1} -> ENFORCE: every non-Normal annotator must have a valid mask.
      - If include_unanimous:
          * If 3-0 Normal -> RELAX: do not require masks.
          * If 3-0 non-Normal -> ENFORCE: all three annotators must have valid masks.
    Returns:
      pool: list of (post_id, item, labels)
      pool_maj_counts: Counter of majority labels in the pool
    """
    pool = []
    pool_maj_counts = Counter()

    dropped_missing_required = 0
    admitted_relaxed_normal_majority = 0
    admitted_relaxed_normal_unanimous = 0

    for post_id, item in data.items():
        ann = item.get("annotators", [])
        tokens = item.get("post_tokens", [])
        if len(ann) != 3 or not tokens:
            continue

        labels = [normalize_label(a.get("label")) for a in ann]
        if any(l is None for l in labels):
            continue

        L = len(tokens)
        rats = item.get("rationales", [])
        c = Counter(labels)
        n_normal = c.get("Normal", 0)

        # --- 3-0 branch (optional) ---
        if include_unanimous and is_three_zero(labels):
            unanimous_label = labels[0]
            if unanimous_label == "Normal":
                # RELAX
                pool.append((post_id, item, labels))
                pool_maj_counts[unanimous_label] += 1
                admitted_relaxed_normal_unanimous += 1
            else:
                # ENFORCE: all three must have valid masks
                ok = True
                for i in range(3):
                    vec = rats[i] if (isinstance(rats, list) and i < len(rats)) else None
                    if not _is_binary_vec(vec, L):
                        ok = False
                        break
                if not ok:
                    dropped_missing_required += 1
                    continue
                pool.append((post_id, item, labels))
                pool_maj_counts[unanimous_label] += 1
            continue

        # --- 2-vs-1 branch (original HYBRID) ---
        if not is_two_vs_one(labels):
            continue

        if n_normal == 2:
            # RELAX
            pool.append((post_id, item, labels))
            pool_maj_counts[majority_label_from(labels)] += 1
            admitted_relaxed_normal_majority += 1
            continue

        # ENFORCE: every non-Normal must have a valid mask
        ok = True
        for i, lab in enumerate(labels):
            if lab != "Normal":
                vec = rats[i] if (isinstance(rats, list) and i < len(rats)) else None
                if not _is_binary_vec(vec, L):
                    ok = False
                    break
        if not ok:
            dropped_missing_required += 1
            continue

        pool.append((post_id, item, labels))
        pool_maj_counts[majority_label_from(labels)] += 1

    # Optional diagnostics
    if dropped_missing_required:
        print(f"Dropped {dropped_missing_required} items missing a required mask.")
    if admitted_relaxed_normal_majority:
        print(f"Admitted {admitted_relaxed_normal_majority} Normal-majority 2v1 items without requiring masks.")
    if admitted_relaxed_normal_unanimous:
        print(f"Admitted {admitted_relaxed_normal_unanimous} Normal 3-0 items without requiring masks.")

    return pool, pool_maj_counts

def extract_available_masks_hybrid(item, labels):
    """
    For HYBRID:
      - If 'Normal' in labels: include any valid masks that exist; if none exist, return [] (caller will handle zeros).
      - Else: all three masks must be valid (ensured already), include all.
    """
    tokens = item["post_tokens"]
    L = len(tokens)
    rats = item.get("rationales", [])
    out = []
    for i in range(3):
        vec = rats[i] if (isinstance(rats, list) and i < len(rats)) else None
        if _is_binary_vec(vec, L):
            out.append(vec)
    return out  # may be empty if Normal present and no masks exist

def merge_or(masks, L):
    if not masks:
        return [0] * L
    if len(masks) == 1:
        return masks[0][:]
    acc = [0] * L
    for m in masks:
        for i in range(L):
            acc[i] = int(acc[i] or m[i])
    return acc

def merge_and(masks, L):
    if not masks:
        return [0] * L
    acc = masks[0][:]
    for m in masks[1:]:
        for i in range(len(acc)):
            acc[i] = int(acc[i] and m[i])
    return acc

# --------------- sampling & writing ---------------
def sample_hybrid(
    n_samples=500,
    seed=42,
    output_path="../datasets/personas_&_questions/hatexplain_500_hybrid.jsonl",
    balanced=False,
    per_class_quota=None
):
    random.seed(seed)
    data = load_hatexplain_cached()

    # Diagnostic: true Normal distribution by labels only
    distribution_for_normal_any(data)

    # Build pool with HYBRID policy + unanimous
    pool, pool_maj = build_pool_hybrid(data, include_unanimous=True)
    print("Pool majority-label distribution (2-vs-1 + 3-0, HYBRID):")
    for c in CLASS_ORDER:
        print(f"  {c}: {pool_maj.get(c, 0)}")


    # Decide plan
    if per_class_quota is not None:
        if sum(per_class_quota.values()) != n_samples:
            raise ValueError("per_class_quota must sum to n_samples.")
        plan = per_class_quota
    elif balanced:
        base = n_samples // len(CLASS_ORDER)
        rem = n_samples % len(CLASS_ORDER)
        plan = {c: base + (1 if i < rem else 0) for i, c in enumerate(CLASS_ORDER)}
    else:
        plan = None

    # Sample
    if plan is None:
        if len(pool) < n_samples:
            raise ValueError(f"Requested {n_samples} but pool has only {len(pool)} eligible items.")
        sampled = random.sample(pool, n_samples)
    else:
        by_maj = {c: [] for c in CLASS_ORDER}
        for rec in pool:
            by_maj[majority_label_from(rec[2])].append(rec)
        sampled = []
        for c in CLASS_ORDER:
            need = plan[c]
            have = len(by_maj[c])
            if have < need:
                raise ValueError(f"Not enough items for majority='{c}'. Need {need}, have {have}.")
            sampled.extend(random.sample(by_maj[c], need))

    # Write JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (post_id, item, labels) in enumerate(sampled):
            tokens = item["post_tokens"]
            L = len(tokens)
            annotators = item["annotators"]

            masks = extract_available_masks_hybrid(item, labels)
            rec = OrderedDict()
            rec["id"] = f"s{idx:04d}"
            rec["post_id"] = post_id
            rec["input_text"] = " ".join(tokens)
            rec["post_tokens"] = tokens

            sorted_annotators = sorted(annotators, key=lambda a: a.get("annotator_id"))
            rec["annotator_ids_sorted"] = [a.get("annotator_id") for a in sorted_annotators]
            rec["annotator_labels_sorted"] = [normalize_label(a.get("label")) for a in sorted_annotators]
            rec["annotator_labels_raw"] = labels

            rec["rationales_all"] = item.get("rationales", [])  # may be missing/invalid for Normal cases
            rec["merged_rationale_or"] = merge_or(masks, L)     # zeros if no masks exist
            rec["merged_rationale_and"] = merge_and(masks, L)   # zeros if no masks exist
            rec["majority_label"] = majority_label_from(labels)

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    sampled_maj = Counter([majority_label_from(labels) for _, _, labels in sampled])
    print(f"[✓] Saved {len(sampled)} 2-vs-1 samples (HYBRID) to {output_path}")
    print("Sampled majority-label distribution (HYBRID):")
    for c in CLASS_ORDER:
        print(f"  {c}: {sampled_maj.get(c, 0)}")

def diagnose_normal_2v1_mask_presence(data):
    labels_only_total = 0
    survive_required_mask = 0
    for _, item in data.items():
        ann = item.get("annotators", [])
        toks = item.get("post_tokens", [])
        if len(ann) != 3 or not toks:
            continue
        labels = [normalize_label(a.get("label")) for a in ann]
        if any(l is None for l in labels) or not is_two_vs_one(labels):
            continue
        if Counter(labels).get("Normal", 0) == 2:  # Normal-majority 2v1
            labels_only_total += 1
            L = len(toks); rats = item.get("rationales", [])
            # minority is the non-Normal
            for i, lab in enumerate(labels):
                if lab != "Normal":
                    vec = rats[i] if (isinstance(rats, list) and i < len(rats)) else None
                    if _is_binary_vec(vec, L):
                        survive_required_mask += 1
                    break
    print(f"Normal 2v1 by labels: {labels_only_total}")
    print(f"…with non-Normal mask present: {survive_required_mask}")


# --------------- run ---------------
if __name__ == "__main__":
    sample_hybrid(
        n_samples=500,
        seed=42,
        output_path="../datasets/hatexplain/hatexplain_500_hybrid.jsonl",
        balanced=False,           # set True or use per_class_quota if you want to control class mix
        per_class_quota=None
    )
