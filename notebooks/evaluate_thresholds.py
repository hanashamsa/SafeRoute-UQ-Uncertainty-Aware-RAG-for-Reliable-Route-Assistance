import json, math, os, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score

FUSED="data/generations_fused.jsonl"
LABELS="data/labels.jsonl"
OUT_DECISIONS="data/decisions.jsonl"

def load_jsonl(path):
    return [json.loads(l) for l in open(path,'r',encoding='utf-8').read().splitlines()]

assert Path(FUSED).exists(), f"{FUSED} missing"
assert Path(LABELS).exists(), f"{LABELS} missing. Create and label first."

fused = load_jsonl(FUSED)
labels = {d["query_id"]: d["label"] for d in load_jsonl(LABELS)}

# build arrays
ids=[]
trusts=[]
y=[]
for obj in fused:
    qid = obj.get("query_id")
    ids.append(qid)
    trusts.append(float(obj.get("trust",0.0)))
    lab = labels.get(qid, None)
    if lab is None:
        y.append(None)
    else:
        y.append(1 if lab=="correct" else 0)

# filter only labeled
labeled_idx = [i for i,v in enumerate(y) if v is not None]
if not labeled_idx:
    print("No labeled items found. Label data/labels.jsonl and re-run."); sys.exit(1)

trust_arr = np.array([trusts[i] for i in labeled_idx])
y_arr = np.array([y[i] for i in labeled_idx])

# AUROC
try:
    auroc = roc_auc_score(y_arr, trust_arr)
except Exception as e:
    auroc = None

# AUPRC
prec, rec, _ = precision_recall_curve(y_arr, trust_arr)
auprc = auc(rec, prec)

#  ECE 
def compute_ece(conf, labels, bins=10):
    bins_edges = np.linspace(0.0,1.0,bins+1)
    ece_sum=0.0
    n=len(conf)
    for i in range(bins):
        lo, hi = bins_edges[i], bins_edges[i+1]
        idx = np.where((conf>=lo)&(conf<hi))[0]
        if len(idx)==0: continue
        avg_conf = conf[idx].mean()
        avg_acc = labels[idx].mean()
        ece_sum += (len(idx)/n) * abs(avg_conf - avg_acc)
    return ece_sum

ece = compute_ece(trust_arr, y_arr, bins=10)

# risk-coverage curve
ths = np.linspace(0.0,1.0,101)
coverage=[]
error_rate=[]
for t in ths:
    keep_idx = np.where(trust_arr>=t)[0]
    if len(keep_idx)==0:
        coverage.append(0.0); error_rate.append(0.0); continue
    kept_labels = y_arr[keep_idx]
    cov = len(keep_idx)/len(y_arr)
    err = 1.0 - (kept_labels.mean())  # error rate among kept
    coverage.append(cov); error_rate.append(err)

# choose default thresholds
th_green = 0.8
th_amber = 0.6

# Decisions and counts
decisions=[]
for obj in fused:
    qid = obj["query_id"]
    t = float(obj.get("trust",0.0))
    if t >= th_green:
        decision="green"
    elif t >= th_amber:
        decision="amber"
    else:
        decision="red"
    decisions.append({"query_id":qid,"trust":t,"decision":decision,"query":obj.get("query")})
# write
with open(OUT_DECISIONS,'w',encoding='utf-8') as f:
    for d in decisions: f.write(json.dumps(d,ensure_ascii=False)+"\n")

# summary
print("EVAL SUMMARY")
print("Labeled items:", len(y_arr))
print("AUROC:", auroc)
print("AUPRC:", auprc)
print("ECE (10 bins):", ece)
# risk-coverage point at green threshold
keep_idx = np.where(trust_arr>=th_green)[0]
cov_green = len(keep_idx)/len(trust_arr) if len(trust_arr)>0 else 0.0
err_green = 1 - y_arr[keep_idx].mean() if len(keep_idx)>0 else 0.0
print(f"Green threshold {th_green}: coverage={cov_green:.3f}, error_rate={err_green:.3f} (kept {len(keep_idx)})")
# amber
keep_idx = np.where(trust_arr>=th_amber)[0]
cov_amber = len(keep_idx)/len(trust_arr) if len(trust_arr)>0 else 0.0
err_amber = 1 - y_arr[keep_idx].mean() if len(keep_idx)>0 else 0.0
print(f"Amber threshold {th_amber}: coverage={cov_amber:.3f}, error_rate={err_amber:.3f} (kept {len(keep_idx)})")
print("Decisions file:", OUT_DECISIONS, "size bytes:", os.path.getsize(OUT_DECISIONS))
print("\nTop 5 Lowest-trust labeled examples (trust,label,query_id):")
order = np.argsort(trust_arr)
for i in order[:5]:
    print(trust_arr[i], y_arr[i], ids[labeled_idx[i]])

# print risk-coverage sample lines
print("\nSample risk-coverage (threshold,coverage,error_rate) for some thresholds:")
for t,c,e in zip(ths[::10], coverage[::10], error_rate[::10]):
    print(f"{t:.2f} {c:.3f} {e:.3f}")

#  CSV summary of trust vs correctness
csv_out="results/trust_vs_label.csv"
os.makedirs("results", exist_ok=True)
with open(csv_out,"w",encoding="utf-8") as f:
    f.write("query_id,trust,label\n")
    for i,idx in enumerate(labeled_idx):
        f.write(f"{ids[idx]},{trusts[idx]},{y[idx]}\n")
print("Wrote csv:",csv_out)
