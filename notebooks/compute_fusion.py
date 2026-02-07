import json, math, os
from pathlib import Path

MODE = os.environ.get("UQ_MODE", "fused")


INP = "data/generations.jsonl"
OUT = "data/generations_fused.jsonl"



def safe_load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f.read().splitlines()]

def safe_write_line(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def compute_R_score(retrieved):
    if not retrieved:
        return 0.0
    scores = [float(r.get("score", 0.0) or 0.0) for r in retrieved]
    return max(scores)

def normalize(vals):
    if not vals:
        return [], 0.0, 1.0
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-9:
        return [0.0 for _ in vals], lo, hi
    return [(v - lo) / (hi - lo) for v in vals], lo, hi

def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))



def main():
    assert Path(INP).exists(), f"{INP} not found"
    data = safe_load_lines(INP)

    # --- collect raw signals ---
    R_raw = []
    Gen_entropy_raw = []
    Poly_perp_raw = []

    for obj in data:
        # Retrieval
        R_raw.append(compute_R_score(obj.get("retrieved", [])))

        # Handcrafted generation uncertainty
        gen = obj.get("generation", {})
        Gen_entropy_raw.append(gen.get("mean_entropy", 10.0))

        # LM-Polygraph uncertainty (signal is perplexity, but we convert to confidence later)
        Poly_perp_raw.append(gen.get("poly_perplexity", None))

    # Normalize signals
    R_norm, rmin, rmax = normalize(R_raw)
    G_norm, gmin, gmax = normalize(Gen_entropy_raw)



    # weights
    alpha = 1.0   # generation
    beta = 1.0    # retrieval
    delta = 1.0   # polygraph

    if Path(OUT).exists():
        Path(OUT).unlink()

    # --- fusion ---
    for i, obj in enumerate(data):
        rscore = R_norm[i]
        gen_unc = G_norm[i]
        gen_conf = max(0.0, 1.0 - gen_unc)

        poly_perp = Poly_perp_raw[i]
        if poly_perp is None:
            poly_conf = 0.0
        else:
            poly_conf = 1.0 / (1.0 + poly_perp)


        if MODE == "retrieval_only":
            raw = rscore

        elif MODE == "generation_only":
            raw = gen_conf

        elif MODE == "polygraph_only":
            raw = poly_conf

        elif MODE == "fused":
            raw = alpha * gen_conf + beta * rscore + delta * poly_conf

        else:
            raise ValueError(f"Unknown UQ_MODE: {MODE}")

        trust = logistic(raw)

        fused = dict(obj)
        fused.update({
            "mode": MODE,
            "features": {
                "R_raw": R_raw[i],
                "R_norm": rscore,
                "Gen_raw_entropy": Gen_entropy_raw[i],
                "Gen_conf_norm": gen_conf,
                "Poly_raw_perplexity": poly_perp,
                "Poly_conf": poly_conf,
                "fusion_raw": raw
            },
            "trust": trust
        })

        safe_write_line(OUT, fused)

    trusts = [json.loads(l)["trust"] for l in open(OUT, "r", encoding="utf-8")]
    print("Wrote", len(trusts), "records to", OUT)
    print("Trust stats:",
          "min", min(trusts),
          "median", sorted(trusts)[len(trusts)//2],
          "max", max(trusts))


if __name__ == "__main__":
    main()
