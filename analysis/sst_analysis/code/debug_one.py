import json
from collections import Counter
from itertools import combinations

def parse(pid):
    try:
        _, coded = pid.split('___',1)
        a,g,e = coded.split('_')
        return int(a), g, e
    except Exception:
        return None, None, None

path='datasets/results_gpt_oss_120b_cose_merged/QID101.jsonl'
rows=[]
with open(path) as f:
    for line in f:
        o=json.loads(line)
        rows.append(o)

print('rows', len(rows))
print([r['persona_id'] for r in rows])

keys=Counter()
for a,b in combinations(rows,2):
    pa=parse(a['persona_id'])
    pb=parse(b['persona_id'])
    diffs=[k for k,(x,y) in zip(['age','gender','ethnicity'], zip(pa,pb)) if x!=y]
    key=tuple(diffs) or ('same',)
    keys[key]+=1
print('pairwise keys:', keys)
