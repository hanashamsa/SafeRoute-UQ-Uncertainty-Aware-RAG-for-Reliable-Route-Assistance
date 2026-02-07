import json
from sentence_transformers import SentenceTransformer
import faiss

class Retriever:
    def __init__(self, index_path="data/faiss_index.idx", meta_path="data/meta.json", model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def retrieve(self, query, k=5):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(D[0], I[0]):
            info = self.meta[idx]
            results.append({
                "id": info["id"],
                "text": info["text"],
                "score": float(score),
                "source": info.get("source", None)
            })
        return results
