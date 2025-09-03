# books_embedder.py
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load .env
load_dotenv()

# --- Paths from env ---
BASE_DIR = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
INPUT_PATH = Path(os.getenv("BOOKS_META_CSV", BASE_DIR / "backend/data/interim_large/books_metadata.csv"))
OUT_DIR = Path(os.getenv("BOOKS_EMBED_OUT", BASE_DIR / "backend/data/interim_large"))
MODEL_NAME = os.getenv("BOOKS_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def build_text(row: pd.Series) -> str:
    """Compose text for embedding from book metadata"""
    title = (row.get("title") or "").strip()
    authors = (row.get("authors") or "").strip()
    cats = (row.get("categories") or "").strip()
    desc = (row.get("description") or "").strip()
    if len(desc) > 1000:
        desc = desc[:1000]
    parts = []
    if title: parts.append(f"Title: {title}")
    if authors: parts.append(f"Authors: {authors}")
    if cats: parts.append(f"Categories: {cats}")
    if desc: parts.append(f"Description: {desc}")
    return " | ".join(parts) if parts else title

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms

def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    # 0) Turn empty/whitespace strings anywhere into NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # 1) Explicitly coerce numeric-ish columns to numbers
    NUM_COLS = ["pageCount", "averageRating", "ratingsCount"]
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace({"": np.nan, "nan": np.nan, "None": np.nan})
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Float64")  # nullable floats

    # 2) Fill ONLY text columns with "", never numeric ones
    TEXT_COLS = [
        "title", "authors", "categories", "description", "language",
        "publishedDate", "previewLink", "infoLink", "isbn13", "isbn10"
    ]
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("")

    # 3) Safety: if any numeric col still looks 'object', force it off object
    for c in NUM_COLS:
        if c in df.columns and df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Float64")

    # 4) Hard assert to catch offenders early
    if "pageCount" in df.columns:
        bad_mask = df["pageCount"].apply(lambda v: isinstance(v, str))
        if bad_mask.any():
            df.loc[bad_mask, "pageCount"] = np.nan
            df["pageCount"] = df["pageCount"].astype("Float64")

    # (Optional but recommended) ensure volume_id exists & is not null
    if "volume_id" not in df.columns:
        raise ValueError("Input CSV must contain 'volume_id' column.")
    df = df.dropna(subset=["volume_id"])

    df["volume_id"] = df["volume_id"].astype(str)

    df["text_for_embedding"] = df.apply(build_text, axis=1)

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    texts = df["text_for_embedding"].tolist()
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    print(f"Preparing to embed {len(df)} books")
    print(f"Model: {args.model}, Batch size: {args.batch_size}")

    emb = l2_normalize(emb).astype(np.float32)

    # Save vectors + mapping + metadata
    vec_path = OUT_DIR / "books_embeddings.npy"
    map_path = OUT_DIR / "books_embeddings_map.parquet"
    meta_path = OUT_DIR / "books_with_text.parquet"

    np.save(vec_path, emb)
    pd.DataFrame({"volume_id": df["volume_id"], "row_idx": np.arange(len(df))}).to_parquet(map_path, index=False)

    # Robust save: Parquet with CSV fallback (while keeping numeric dtypes)
    try:
        df.drop(columns=["text_for_embedding"]).to_parquet(meta_path, index=False)
        print(f"Saved metadata → {meta_path}")
    except Exception as e:
        csv_fallback = meta_path.with_suffix(".csv")
        df.drop(columns=["text_for_embedding"]).to_csv(csv_fallback, index=False)
        print(f"Parquet failed ({e}). Saved CSV fallback → {csv_fallback}")

    print(f"Saved vectors: {emb.shape} → {vec_path}")
    print(f"Saved id→row map → {map_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed book metadata into vectors.")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    main(args)
