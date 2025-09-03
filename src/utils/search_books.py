# search_books.py
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Paths from embed_books.py
EMBED_PATH = "data/interim_large/books_embeddings.npy"
MAP_PATH = "data/interim_large/books_embeddings_map.parquet"
META_PATH = "data/interim_large/books_with_text.parquet"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Load data ---
book_vecs = np.load(EMBED_PATH)            # (N, D), already L2-normalized
id_map = pd.read_parquet(MAP_PATH)         # volume_id → row_idx
meta = pd.read_parquet(META_PATH)          # book metadata
model = SentenceTransformer(MODEL_NAME)

def encode_query(query: str) -> np.ndarray:
    v = model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v.astype(np.float32)

def search(query: str, k: int = 5):
    qv = encode_query(query)  # (1, D)
    sims = (book_vecs @ qv.T).ravel()   # cosine since normalized
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]

    results = []
    for i in idx:
        vol_id = id_map.loc[id_map["row_idx"] == i, "volume_id"].iloc[0]
        row = meta.loc[meta["volume_id"] == vol_id].iloc[0]
        results.append({
            "volume_id": vol_id,
            "title": row.get("title", ""),
            "authors": row.get("authors", ""),
            "categories": row.get("categories", ""),
            "similarity": float(sims[i])
        })
    return results

if __name__ == "__main__":
    query = "introduction to machine learning"
    for r in search(query, k=5):
        print(f"[{r['similarity']:.3f}] {r['title']} — {r['authors']} ({r['categories']})")
