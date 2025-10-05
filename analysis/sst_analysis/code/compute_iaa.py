import json
import os
from collections import defaultdict, Counter
from itertools import combinations

# Optional third-party packages
try:
    from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss_kappa
except Exception:
    sm_fleiss_kappa = None
try:
    import krippendorff as kd
    import numpy as np
except Exception:
    kd = None
    np = None
try:
    from sklearn.metrics import cohen_kappa_score as sk_cohen_kappa_score
except Exception:
    sk_cohen_kappa_score = None


def parse_persona(persona_id: str):
    try:
        _, coded = persona_id.split("___", 1)
        parts = coded.split("_")
        if len(parts) != 3:
            return None
        age_str, gender, ethnicity = parts
        if not age_str.isdigit():
            return None
        if gender not in {"f", "m"}:
            return None
        if ethnicity not in {"b", "l", "w"}:
            return None
        return {"age": int(age_str), "gender": gender, "ethnicity": ethnicity}
    except Exception:
        return None


def diff_attributes(p1, p2):
    diffs = []
    for k in ("age", "gender", "ethnicity"):
        if p1[k] != p2[k]:
            diffs.append(k)
    return tuple(sorted(diffs))


def load_model_results(root_dir: str):
    by_qid = defaultdict(list)
    for fname in os.listdir(root_dir):
        if not fname.endswith(".jsonl"):
            continue
        if fname.startswith("baseline_"):
            continue
        qid = fname.split(".")[0]
        path = os.path.join(root_dir, fname)
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                persona_id = obj.get("persona_id")
                attrs = parse_persona(persona_id or "")
                if attrs is None:
                    continue
                ans_idx = obj.get("model_answer_index")
                ans_txt = obj.get("model_answer")
                by_qid[qid].append({
                    "persona_id": persona_id,
                    "attrs": attrs,
                    "answer_index": ans_idx,
                    "answer": ans_txt,
                })
    return by_qid


def get_label(item):
    # Prefer numeric index; fallback to text
    if item.get("answer_index") is not None:
        return ("idx", item["answer_index"])  # namespace to avoid clashing with text
    return ("txt", item.get("answer"))


def prepare_counts(by_qid):
    # Build per-question category counts
    # Map labels to contiguous integers per model
    label_to_id = {}
    next_id = 0
    per_q_counts = []  # list of Counter per question
    per_q_personas = []  # parallel list of list of (persona_id, attrs, label_id)

    for qid, rows in by_qid.items():
        cnt = Counter()
        labeled = []
        for r in rows:
            lbl = get_label(r)
            if lbl not in label_to_id:
                label_to_id[lbl] = next_id
                next_id += 1
            lid = label_to_id[lbl]
            cnt[lid] += 1
            labeled.append((r["persona_id"], r["attrs"], lid))
        if cnt:
            per_q_counts.append(cnt)
            per_q_personas.append(labeled)
    k = next_id  # number of categories observed
    return per_q_counts, per_q_personas, k


def load_baseline(root_dir: str):
    # Find a baseline_*.jsonl file and load QID -> label
    bl_files = [f for f in os.listdir(root_dir) if f.startswith('baseline_') and f.endswith('.jsonl')]
    if not bl_files:
        return {}
    path = os.path.join(root_dir, bl_files[0])
    q2lbl = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = obj.get('question_id')
            if qid is None and 'persona_id' in obj:
                pid = obj['persona_id']
                qid = pid.split('___',1)[0]
            if qid is None:
                continue
            # Prefer index
            if obj.get('model_answer_index') is not None:
                lbl = ('idx', obj['model_answer_index'])
            else:
                lbl = ('txt', obj.get('model_answer'))
            q2lbl[qid] = lbl
    return q2lbl


def prepare_counts_with_baseline(by_qid, baseline_map):
    # Build per-question category counts including baseline as an additional rater
    label_to_id = {}
    next_id = 0
    per_q_counts = []
    # Also return counts without baseline to compute persona-baseline agreement
    per_q_counts_persona_only = []

    for qid, rows in by_qid.items():
        cnt = Counter()
        for r in rows:
            lbl = get_label(r)
            if lbl not in label_to_id:
                label_to_id[lbl] = next_id
                next_id += 1
            cnt[label_to_id[lbl]] += 1
        per_q_counts_persona_only.append(cnt.copy())
        bl = baseline_map.get(qid)
        if bl is not None:
            if bl not in label_to_id:
                label_to_id[bl] = next_id
                next_id += 1
            cnt[label_to_id[bl]] += 1
        per_q_counts.append(cnt)
    k = next_id
    return per_q_counts, per_q_counts_persona_only, k


def _mode_total(per_q_counts):
    totals = Counter(sum(c.values()) for c in per_q_counts if c)
    if not totals:
        return None
    return max(totals.items(), key=lambda kv: kv[1])[0]


def fleiss_kappa_statsmodels(per_q_counts, k):
    if sm_fleiss_kappa is None:
        return None
    mode_n = _mode_total(per_q_counts)
    if mode_n is None:
        return None
    table = []
    for cnt in per_q_counts:
        if sum(cnt.values()) != mode_n:
            continue
        row = [cnt.get(c, 0) for c in range(k)]
        table.append(row)
    if not table:
        return None
    try:
        return float(sm_fleiss_kappa(table))
    except Exception:
        return None


def persona_keys():
    return [(age, gender, eth) for age in (25, 45) for gender in ("f", "m") for eth in ("b", "l", "w")]


def krippendorff_alpha_pkg(by_qid, label_to_id=None, include_baseline=False, baseline_map=None):
    if kd is None or np is None:
        return None
    # Build coder-by-item matrix with fixed coder identities
    coders = persona_keys()
    if include_baseline:
        coders = coders + [("baseline",)]

    # Map labels to ids if not provided
    if label_to_id is None:
        label_to_id = {}
        next_id = 0
        for rows in by_qid.values():
            for r in rows:
                lbl = get_label(r)
                if lbl not in label_to_id:
                    label_to_id[lbl] = next_id
                    next_id += 1
        if include_baseline and baseline_map:
            for lbl in baseline_map.values():
                if lbl not in label_to_id:
                    label_to_id[lbl] = next_id
                    next_id += 1

    items = list(by_qid.keys())
    data = []  # coder x item
    for coder in coders:
        row = []
        for qid in items:
            if coder == ("baseline",):
                if baseline_map and qid in baseline_map:
                    row.append(label_to_id.get(baseline_map[qid], np.nan))
                else:
                    row.append(np.nan)
                continue
            # persona coder
            target = None
            for r in by_qid[qid]:
                a = r["attrs"]
                if (a["age"], a["gender"], a["ethnicity"]) == coder:
                    target = r
                    break
            if target is None:
                row.append(np.nan)
            else:
                row.append(label_to_id.get(get_label(target)))
        data.append(row)

    try:
        alpha = float(kd.alpha(reliability_data=np.array(data, dtype=float), level_of_measurement='nominal'))
        return alpha
    except Exception:
        return None


def fleiss_kappa(per_q_counts, k):
    # Allows varying number of ratings per item
    if not per_q_counts:
        return float("nan")
    # category proportions across all ratings
    total_ratings = 0
    cat_totals = [0] * k
    for cnt in per_q_counts:
        n_i = sum(cnt.values())
        total_ratings += n_i
        for c in range(k):
            cat_totals[c] += cnt.get(c, 0)
    p = [ct / total_ratings for ct in cat_totals]
    P_bar = 0.0
    m = len(per_q_counts)
    for cnt in per_q_counts:
        n_i = sum(cnt.values())
        if n_i <= 1:
            continue
        Pi = sum(v * (v - 1) for v in cnt.values()) / (n_i * (n_i - 1))
        P_bar += Pi
    P_bar /= m
    P_e = sum(pi * pi for pi in p)
    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def krippendorff_alpha_nominal(per_q_counts, k):
    # Build coincidence matrix O
    if not per_q_counts:
        return float("nan")
    O = [[0.0 for _ in range(k)] for _ in range(k)]
    n_pairs_total = 0
    for cnt in per_q_counts:
        n_i = sum(cnt.values())
        if n_i <= 1:
            continue
        n_pairs_total += n_i * (n_i - 1)
        for c in range(k):
            n_c = cnt.get(c, 0)
            # diagonal
            O[c][c] += n_c * (n_c - 1)
            # off-diagonal
            for c2 in range(c + 1, k):
                n_c2 = cnt.get(c2, 0)
                v = n_c * n_c2
                O[c][c2] += v
                O[c2][c] += v
    if n_pairs_total == 0:
        return float("nan")
    # Observed disagreement Do
    O_total = sum(sum(row) for row in O)
    if O_total == 0:
        return float("nan")
    Do = 1.0 - sum(O[i][i] for i in range(k)) / O_total
    # Expected disagreement De
    m = [sum(O[i][j] for j in range(k)) for i in range(k)]
    M = sum(m)
    if M == 0:
        return float("nan")
    De = 1.0 - sum((mi / M) ** 2 for mi in m)
    if De == 0:
        return 1.0
    return 1.0 - Do / De


def cohen_kappa_from_conf(conf):
    # conf is k x k confusion matrix counts
    k = len(conf)
    N = sum(sum(row) for row in conf)
    if N == 0:
        return float("nan")
    po = sum(conf[i][i] for i in range(k)) / N
    row_marg = [sum(conf[i][j] for j in range(k)) for i in range(k)]
    col_marg = [sum(conf[i][j] for i in range(k)) for j in range(k)]
    pe = sum((row_marg[i] * col_marg[i]) for i in range(k)) / (N * N)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def pairwise_kappas_grouped(per_q_personas, k):
    # Build aggregated confusion matrices per attribute-diff group across all questions
    groups = defaultdict(lambda: [[0 for _ in range(k)] for _ in range(k)])
    for labeled in per_q_personas:
        # labeled: list of (persona_id, attrs, lid)
        for a, b in combinations(labeled, 2):
            attrs_a, lid_a = a[1], a[2]
            attrs_b, lid_b = b[1], b[2]
            diffs = diff_attributes(attrs_a, attrs_b)
            key = diffs or ("same",)
            groups[key][lid_a][lid_b] += 1
            groups[key][lid_b][lid_a] += 1  # symmetric pairs across raters
    # Compute kappa per group
    kappas = {}
    counts = {}
    for key, conf in groups.items():
        N = sum(sum(row) for row in conf)
        kappas[key] = cohen_kappa_from_conf(conf)
        counts[key] = N
    return kappas, counts


def main():
    import argparse
    parser = argparse.ArgumentParser(description="IAA metrics for persona results")
    parser.add_argument("folders", nargs="+", help="Paths to merged result folders")
    args = parser.parse_args()

    for folder in args.folders:
        print(f"\nModel folder: {folder}")
        by_qid = load_model_results(folder)
        per_q_counts, per_q_personas, k = prepare_counts(by_qid)
        print(f"Questions with data: {len(per_q_counts)}  Categories observed: {k}")

        # Prefer packages when available; fallback to internal implementations
        fk_pkg = fleiss_kappa_statsmodels(per_q_counts, k)
        fk = fk_pkg if fk_pkg is not None else fleiss_kappa(per_q_counts, k)
        ka_pkg = krippendorff_alpha_pkg(by_qid)
        ka = ka_pkg if ka_pkg is not None else krippendorff_alpha_nominal(per_q_counts, k)
        print(f"Fleiss' kappa (personas only): {fk:.3f}{' [statsmodels]' if fk_pkg is not None else ''}")
        print(f"Krippendorff's alpha (nominal, personas only): {ka:.3f}{' [krippendorff]' if ka_pkg is not None else ''}")

        # With baseline as an additional rater
        baseline_map = load_baseline(folder)
        if baseline_map:
            per_q_counts_bl, _, k_bl = prepare_counts_with_baseline(by_qid, baseline_map)
            fk_bl_pkg = fleiss_kappa_statsmodels(per_q_counts_bl, k_bl)
            fk_bl = fk_bl_pkg if fk_bl_pkg is not None else fleiss_kappa(per_q_counts_bl, k_bl)
            ka_bl_pkg = krippendorff_alpha_pkg(by_qid, include_baseline=True, baseline_map=baseline_map)
            ka_bl = ka_bl_pkg if ka_bl_pkg is not None else krippendorff_alpha_nominal(per_q_counts_bl, k_bl)
            print(f"Fleiss' kappa (with baseline): {fk_bl:.3f}{' [statsmodels]' if fk_bl_pkg is not None else ''}")
            print(f"Krippendorff's alpha (with baseline): {ka_bl:.3f}{' [krippendorff]' if ka_bl_pkg is not None else ''}")

        kappas, counts = pairwise_kappas_grouped(per_q_personas, k)
        print("Pairwise Cohen's kappa (aggregated) by attribute difference:")
        for key in sorted(kappas.keys(), key=lambda x: (len(x), x)):
            val = kappas[key]
            n = counts[key]
            label = "same" if key == ("same",) else ",".join(key)
            print(f"  {label:>12}: {val:.3f} (pairs={n})")


if __name__ == "__main__":
    main()
