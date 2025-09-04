# build_book_topics.py — books-only topic vectors (minimal)
import os, re
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def clean(s: str) -> str:
    s = (str(s) if s is not None else "").lower().strip()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s)

def main():
    # Load the .env nearest to the repo tree and override any stale OS env
    env_path = Path(find_dotenv(usecwd=True))
    if not env_path:
        raise RuntimeError("No .env found. Place it next to manage.py.")
    load_dotenv(env_path, override=True)
    base = env_path.parent

    def need_path(key: str) -> Path:
        v = os.getenv(key)
        if not v:
            raise RuntimeError(f"Missing env var: {key}")
        p = Path(v)
        return p if p.is_absolute() else (base / p)

    books_csv = need_path("BOOKS_CSV")
    outdir    = need_path("TOPICS_OUTDIR")
    outdir.mkdir(parents=True, exist_ok=True)

    # Hyperparams (defaults if not in .env)
    K        = int(os.getenv("TOPICS_K", 30))
    MIN_DF   = int(os.getenv("TOPICS_MIN_DF", 5))
    MAX_DF   = float(os.getenv("TOPICS_MAX_DF", 0.5))
    MAX_FEAT = int(os.getenv("TOPICS_MAX_FEATURES", 50000))
    MAX_ITER = int(os.getenv("TOPICS_MAX_ITER", 20))
    SEED     = int(os.getenv("TOPICS_RANDOM_STATE", 42))
    SAMPLE   = int(os.getenv("TOPICS_SAMPLE", 0))  # 0 = all

    if not books_csv.exists():
        raise FileNotFoundError(f"BOOKS_CSV not found: {books_csv}")
    df = pd.read_csv(books_csv)

    # Pick an ID column (fallback to row index)
    for c in ("book_id", "id", "isbn13", "isbn"):
        if c in df.columns:
            df = df.rename(columns={c: "id"})
            break
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))

    # Likely text columns in book metadata
    text_cols_pref = [
        "title", "subtitle", "authors", "author", "description",
        "categories", "category", "tags", "publisher"
    ]
    text_cols = [c for c in text_cols_pref if c in df.columns]
    if not text_cols:
        raise RuntimeError("No text columns found in books CSV")

    df = df[["id"] + text_cols].copy()
    df["text"] = df[text_cols].astype(str).agg(" ".join, axis=1).map(clean)
    df = df[["id", "text"]].dropna().drop_duplicates("id")
    if SAMPLE and len(df) > SAMPLE:
        df = df.sample(SAMPLE, random_state=SEED).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No book rows after cleaning")

    # Vectorize + LDA
    vect = CountVectorizer(stop_words="english", min_df=MIN_DF, max_df=MAX_DF, max_features=MAX_FEAT)
    X = vect.fit_transform(df["text"])
    lda = LatentDirichletAllocation(
        n_components=K, max_iter=MAX_ITER, learning_method="batch",
        random_state=SEED, evaluate_every=0
    )
    theta = lda.fit_transform(X)  # (n_items, K)

    # Output: one file with per-book topic distribution
    topic_cols = [f"topic_{i:02d}" for i in range(K)]
    out = pd.DataFrame(theta, columns=topic_cols)
    out.insert(0, "id", df["id"].to_numpy())

    try:
        out_path = outdir / "book_topic_vectors.parquet"
        out.to_parquet(out_path, index=False)
    except Exception:
        out_path = outdir / "book_topic_vectors.csv"
        out.to_csv(out_path, index=False)

    print(f"[OK] {len(out)} rows → {out_path}")

if __name__ == "__main__":
    main()
