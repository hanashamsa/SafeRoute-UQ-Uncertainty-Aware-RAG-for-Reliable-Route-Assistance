import json
import math
import numpy as np



STATS = {
    "r_min": 0.33648794889450073,
    "r_max": 0.6508126258850098,
    "entropy_min": 0.4706683007832796,
    "entropy_max": 1.211749160612623,
}

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))



ALPHA = 0.4   # generation
BETA = 0.6    # retrieval
SCALE = 6.0
SHIFT = -0.5



rows = []
with open("data/generations.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

results = []

for obj in rows:
    query = obj["query"]

    # retrieval confidence
    retrieved = obj.get("retrieved", [])
    R_raw = max([r.get("score", 0.0) for r in retrieved]) if retrieved else 0.0

    r_min, r_max = STATS["r_min"], STATS["r_max"]
    R_norm = 0.0 if r_max <= r_min else (R_raw - r_min) / (r_max - r_min)
    R_norm = float(np.clip(R_norm, 0.0, 1.0))

    # generation uncertainty
    mean_entropy = obj.get("generation", {}).get("mean_entropy")

    e_min, e_max = STATS["entropy_min"], STATS["entropy_max"]
    if mean_entropy is None:
        gen_unc_norm = 0.6
    else:
        gen_unc_norm = (mean_entropy - e_min) / (e_max - e_min)
        gen_unc_norm = float(np.clip(gen_unc_norm, 0.0, 1.0))

    gen_conf = 1.0 - gen_unc_norm

    # fusion
    raw = ALPHA * gen_conf + BETA * R_norm
    trust = sigmoid(SCALE * (raw + SHIFT))

    results.append((trust, query))



results.sort(reverse=True, key=lambda x: x[0])



print("\n=== Trust scores for 50 questions (highest first) ===\n")
for i, (trust, query) in enumerate(results, 1):
    label = "GREEN" if trust >= 0.8 else "AMBER" if trust >= 0.6 else "RED"
    print(f"{i:02d}. Trust={trust:.3f} [{label}] | {query}")

