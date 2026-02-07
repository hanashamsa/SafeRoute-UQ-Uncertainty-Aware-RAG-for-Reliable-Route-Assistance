# notebooks/compute_dataset_stats.py
import json, math
from pathlib import Path
OUT="data/stats.json"
lines = open("data/generations.jsonl","r",encoding="utf-8").read().splitlines()
r_list=[]
entropy_list=[]
for l in lines:
    j=json.loads(l)
    r = max([x.get("score",0.0) for x in j.get("retrieved",[])]) if j.get("retrieved") else 0.0
    me = j.get("generation",{}).get("mean_entropy")
    if me is not None:
        entropy_list.append(float(me))
    r_list.append(float(r))
import json
stats = {
    "r_min": min(r_list) if r_list else 0.0,
    "r_max": max(r_list) if r_list else 1.0,
    "entropy_min": min(entropy_list) if entropy_list else 0.0,
    "entropy_max": max(entropy_list) if entropy_list else 6.0,
    "count": len(r_list)
}
Path("data").mkdir(parents=True, exist_ok=True)
open(OUT,"w",encoding="utf-8").write(json.dumps(stats,indent=2))
print("Wrote",OUT, stats)
