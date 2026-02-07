import pandas as pd
import os

CSV_DIR = "data/csv_sources"
OUT_DIR = "corpus_raw"
MAX_ROWS_PER_FILE = 300   
doc_id = 0

os.makedirs(OUT_DIR, exist_ok=True)

def row_to_text(row):
    parts = []
    for col, val in row.items():
        if pd.notna(val):
            parts.append(f"{col}: {val}")
    return " | ".join(parts)

for fname in os.listdir(CSV_DIR):
    if not fname.endswith(".csv"):
        continue

    csv_path = os.path.join(CSV_DIR, fname)
    print(f"Processing {fname}")

    row_count = 0

    # READ IN CHUNKS
    for chunk in pd.read_csv(csv_path, chunksize=100):
        for _, row in chunk.iterrows():
            text = row_to_text(row)
            with open(f"{OUT_DIR}/csv_doc_{doc_id}.txt", "w") as f:
                f.write(text)
            doc_id += 1
            row_count += 1

            if row_count >= MAX_ROWS_PER_FILE:
                break

        if row_count >= MAX_ROWS_PER_FILE:
            break

print(f"Created {doc_id} text documents")
