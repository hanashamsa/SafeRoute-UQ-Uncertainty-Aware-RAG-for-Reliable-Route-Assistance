import os, json
import pdfplumber

RAW_DIR = "corpus_raw"
OUT_PATH = "data/raw_corpus.jsonl"

def extract_pdf(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    return "\n".join(text)

items = []
i = 0
for fname in os.listdir(RAW_DIR):
    path = os.path.join(RAW_DIR, fname)
    if not os.path.isfile(path): continue
    if fname.lower().endswith(".pdf"):
        text = extract_pdf(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    if not text.strip(): continue
    item = {
        "id": f"doc_{i}",
        "source": fname,
        "timestamp": None,
        "text": text.replace("\n", " ").strip()[:20000]  # cap length
    }
    items.append(item)
    i += 1

with open(OUT_PATH, "w", encoding="utf-8") as out:
    for it in items:
        out.write(json.dumps(it) + "\n")
print("Wrote", len(items), "docs to", OUT_PATH)
