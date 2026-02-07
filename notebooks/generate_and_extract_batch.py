import os, json, math, time
from pathlib import Path
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.retriever import Retriever
from src.models.whitebox_adapter import poly_generate_and_extract

# ================= CONFIG =================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 3
MAX_NEW_TOKENS = 120
OUT_PATH = "data/generations.jsonl"
TEST_QUERIES = "data/test_queries.jsonl"

# Monte Carlo polygraph settings
POLY_SAMPLES = 5
POLY_TEMPERATURE = 0.7

# Semantic variance
EMBED_MODEL = "all-MiniLM-L6-v2"
SEM_SAMPLES = 4
SEM_MAX_NEW = 80

# =========================================

def build_prompt(query, retrieved):
    header = "You are an assistant that explains if a route is impacted by notices.\n"
    context = ""
    for r in retrieved:
        context += f"[DOC:{r['id']} | score={r['score']:.3f}] {r['text']}\n\n"
    return header + "Context:\n" + context + "\nQuestion:\n" + query + "\nAnswer concisely:"

def safe_write_line(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def semantic_variance(samples, embedder):
    embs = embedder.encode(samples, convert_to_numpy=True)
    if embs.shape[0] <= 1:
        return 0.0, 0.0
    cov = np.cov(embs, rowvar=False)
    trace = float(np.trace(cov))
    norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    sims = norm @ norm.T
    idx = np.triu_indices(sims.shape[0], k=1)
    return trace, float(np.std(sims[idx]))

def main():
    os.makedirs("data", exist_ok=True)
    retriever = Retriever()
    embedder = SentenceTransformer(EMBED_MODEL)

    if not Path(TEST_QUERIES).exists():
        raise FileNotFoundError(TEST_QUERIES)

    if Path(OUT_PATH).exists():
        Path(OUT_PATH).unlink()

    queries = [json.loads(l) for l in open(TEST_QUERIES, "r", encoding="utf-8")]

    print("Will process", len(queries), "queries")

    for qi, qobj in enumerate(queries, start=1):
        qtext = qobj.get("query") or qobj.get("text") or qobj.get("question")
        print(f"[{qi}/{len(queries)}] Query:", qtext[:80])

        retrieved = retriever.retrieve(qtext, k=TOP_K)
        prompt = build_prompt(qtext, retrieved)

        # ===== POLYGRAPH GENERATION =====
        gen = poly_generate_and_extract(
            MODEL_NAME,
            prompt,
            device=DEVICE,
            max_new_tokens=MAX_NEW_TOKENS,
            n_samples=POLY_SAMPLES,
            temperature=POLY_TEMPERATURE
        )

        # ===== SEMANTIC SAMPLING =====
        samples = []
        for _ in range(SEM_SAMPLES):
            try:
                samp = poly_generate_and_extract(
                    MODEL_NAME,
                    prompt,
                    device=DEVICE,
                    max_new_tokens=SEM_MAX_NEW,
                    n_samples=1,
                    temperature=POLY_TEMPERATURE
                )
                samples.append(samp["text"])
            except Exception:
                break

        sem_trace, sem_pairstd = semantic_variance(samples, embedder) if samples else (0.0, 0.0)

        out = {
            "query_id": qobj.get("id", f"q{qi}"),
            "query": qtext,
            "prompt": prompt[:4000],
            "retrieved": retrieved,
            "generation": {
                "text": gen["text"],
                "mean_entropy": gen["mean_entropy"],
                "geom_mean_top1": gen["geom_mean_top1"],
                "poly_perplexity": gen["poly_perplexity"],
                "poly_samples": gen["poly_samples"]
            },
            "sampling": {
                "samples_count": len(samples),
                "samples": samples[:4],
                "semantic_trace": sem_trace,
                "semantic_pairwise_std": sem_pairstd
            },
            "timestamp": time.time()
        }

        safe_write_line(OUT_PATH, out)

    print("Done. Wrote outputs to", OUT_PATH)

if __name__ == "__main__":
    main()
