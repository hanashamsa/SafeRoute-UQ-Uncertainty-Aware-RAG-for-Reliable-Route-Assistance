import json, os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
OUT_INDEX = "data/faiss_index.idx"
META = "data/meta.json"

model_name = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)

# load chunks
chunks = []
with open("data/corpus.jsonl","r",encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

texts = [c["text"] for c in chunks]
print("Encoding", len(texts), "texts with", model_name)
embs = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

d = embs.shape[1]
index = faiss.IndexFlatIP(d)  # inner product for cosine after normalization
# normalize embeddings for cosine
faiss.normalize_L2(embs)
index.add(embs)
faiss.write_index(index, OUT_INDEX)
# save metadata
with open(META,"w",encoding="utf-8") as f:
    json.dump(chunks, f)
print("Index saved to", OUT_INDEX)
