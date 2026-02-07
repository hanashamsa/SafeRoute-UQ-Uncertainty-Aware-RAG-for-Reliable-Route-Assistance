import json, os
from tqdm import tqdm
OUT_CHUNKS = "data/corpus.jsonl"
WINDOW = 200   # tokens per chunk
STEP = 100

def chunk_text(text, window=WINDOW, step=STEP):
    words = text.split()
    chunks = []
    for i in range(0, max(1, len(words)), step):
        chunk = " ".join(words[i:i+window])
        if chunk.strip():
            chunks.append(chunk)
        if i+window >= len(words): break
    return chunks

items = []
with open("data/raw_corpus.jsonl","r",encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        chunks = chunk_text(d["text"])
        for idx, c in enumerate(chunks):
            items.append({
                "id": f"{d['id']}_c{idx}",
                "parent": d["id"],
                "text": c,
                "source": d["source"],
                "timestamp": d.get("timestamp")
            })

with open(OUT_CHUNKS,"w",encoding="utf-8") as out:
    for it in items:
        out.write(json.dumps(it) + "\n")
print("Wrote", len(items), "chunks to", OUT_CHUNKS)
