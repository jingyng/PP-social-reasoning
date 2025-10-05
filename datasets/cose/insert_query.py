
import json
import sys
import argparse
import pathlib

def load_test_jsonl(path):
    id_to_query = {}
    with open(path, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Error parsing {path} line {ln}: {e}")
            # Prefer 'annotation_id'; fall back to first evidence docid if needed
            ann_id = obj.get('annotation_id')
            if not ann_id:
                evidences = obj.get('evidences') or []
                if evidences and evidences[0] and isinstance(evidences[0], list) and evidences[0]:
                    ann_id = evidences[0][0].get('docid')
            query = obj.get('query', '')
            if not ann_id:
                continue
            # Split on [sep] and strip whitespace
            parts = [p.strip() for p in query.split('[sep]')]
            # Drop empty strings just in case
            parts = [p for p in parts if p]
            id_to_query[ann_id] = {
                "query": query,
                "query_list": parts
            }
    return id_to_query

def main():
    ap = argparse.ArgumentParser(description="Augment BO_processed.json with query lists from test.jsonl")
    ap.add_argument("--test_jsonl", required=True, help="Path to test.jsonl")
    ap.add_argument("--bo_json", required=True, help="Path to BO_processed.json (array of objects)")
    ap.add_argument("--out", required=False, help="Output JSON file (default: <bo_json stem>_with_queries.json)")
    args = ap.parse_args()
    
    test_map = load_test_jsonl(args.test_jsonl)
    
    with open(args.bo_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit("Expected BO_processed.json to be a JSON array")
    
    missing, updated = 0, 0
    for row in data:
        oid = row.get('originaldata_id')
        if oid in test_map:
            row['query'] = test_map[oid]['query']  # keep the original string too
            row['query_list'] = test_map[oid]['query_list']
            updated += 1
        else:
            # still add empty keys to keep schema consistent
            row.setdefault('query', None)
            row.setdefault('query_list', [])
            missing += 1
    
    out_path = args.out or (str(pathlib.Path(args.bo_json).with_name(pathlib.Path(args.bo_json).stem + "_with_queries.json")))
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Done. Updated {updated} records; {missing} had no matching annotation_id. Wrote: {out_path}")

if __name__ == "__main__":
    main()
