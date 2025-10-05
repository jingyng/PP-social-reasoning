import os
import json

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    args = ap.parse_args()

    total = 0
    fail = 0
    bad = []
    for fn in os.listdir(args.folder):
        if not fn.endswith('.jsonl'):
            continue
        with open(os.path.join(args.folder, fn), 'r') as f:
            for line in f:
                total += 1
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                pid = o.get('persona_id', '')
                if '___' not in pid:
                    fail += 1
                    bad.append(pid)
    print('total', total, 'fail', fail)
    if bad:
        print('examples:', bad[:20])

if __name__ == '__main__':
    main()

