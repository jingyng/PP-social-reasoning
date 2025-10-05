import os
from collections import Counter
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('folder')
args = ap.parse_args()

c = Counter()
for f in os.listdir(args.folder):
    if not f.endswith('.jsonl'):
        continue
    with open(os.path.join(args.folder, f), 'r') as fh:
        n = sum(1 for _ in fh)
    c[n] += 1
print(dict(sorted(c.items())))

