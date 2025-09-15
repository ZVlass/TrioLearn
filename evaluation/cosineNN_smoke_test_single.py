
"""
cosineNN_smoke_test.py
A tiny, grader-friendly cosine NN smoke test over topic vectors.
- No Django. No hard-coded Windows paths.
- Expects: data/topics/all_topic_vectors.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

# -------- config (repo-relative) --------
ROOT = Path(__file__).resolve().parents[1]  # TrioLearn/
PARQUET = ROOT / "backend" / "data" / "topics" / "all_topic_vectors.parquet"
TITLE_COLS = ["title", "name"]  # best-effort for display
K = 5
N_SAMPLES = 20  # for self-retrieval test

def load_vectors(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    # prefer a single list/array column if present
    vec_col = next((c for c in df.columns
                    if df[c].dropna().head(10).apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).all()), None)
    if vec_col is not None:
        X = np.vstack(df[vec_col].apply(np.asarray).to_list()).astype(np.float32)
        feat_cols = [vec_col]
    else:
        # fallback: all numeric columns as topic features
        feat_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        # drop obvious numeric IDs if present
        drop_exact = {"id", "global_id", "external_id", "year"}
        feat_cols = [c for c in feat_cols if c not in drop_exact]
        X = df[feat_cols].astype(np.float32).to_numpy()
    # L2 norm
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, 1e-8)
    return df, Xn, feat_cols

def title_of(row):
    for c in TITLE_COLS:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    return "(untitled)"

def ild_at_k(Xn, idxs):
    if len(idxs) < 2: return 0.0
    sub = Xn[idxs]
    sims = sub @ sub.T
    dists = 1.0 - sims
    triu = np.triu_indices(len(idxs), k=1)
    return float(dists[triu].mean())

def main():
    print(f"[info] Loading topic vectors: {PARQUET}")
    df, Xn, feat_cols = load_vectors(PARQUET)
    n = len(df)
    print(f"[info] Rows={n}, feature_source={'vector_col' if len(feat_cols)==1 else f'{len(feat_cols)} numeric cols'}")

    # Self-retrieval Recall@1 (sanity)
    rng = np.random.default_rng(42)
    sample = rng.choice(n, size=min(N_SAMPLES, n), replace=False)
    hits = 0
    for i in sample:
        scores = Xn[i] @ Xn.T
        top = int(np.argmax(scores))
        hits += int(top == i)
    recall1 = hits / len(sample)
    print(f"[metric] Self-retrieval Recall@1 on n={len(sample)}: {recall1:.3f}")

    # Show a tiny qualitative example
    i = int(sample[0])
    scores = Xn[i] @ Xn.T
    top_idx = np.argpartition(-scores, K)[:K]
    # sort within top-K
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    ild = ild_at_k(Xn, top_idx[:K])

    base_title = title_of(df.iloc[i])
    print("\n[example] Query item:", base_title)
    print("[example] Top-5 neighbours:")
    for rank, j in enumerate(top_idx[:K], 1):
        t = title_of(df.iloc[j])
        print(f"  {rank:>2}. {t}   (cos={scores[j]:.3f})")
    print(f"[metric] ILD@{K} for this list: {ild:.3f}")

if __name__ == "__main__":
    main()
