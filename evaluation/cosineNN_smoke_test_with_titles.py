#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cosineNN_smoke_test_with_titles.py
- Loads topic vectors from parquet (repo-relative).
- Attaches titles via (modality, external_id) from metadata CSVs, or falls back to parquet.
- Computes Recall@1 and Mean ILD@K.
- Prints a Top-5 neighbour example with titles, excluding the self-item.
- Saves the example list to nn_demo_top5.csv for report use.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------- Config ----------------
ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "backend" / "data" / "topics" / "all_topic_vectors.parquet"
K = 5
N_SAMPLES_RECALL = 20
N_SAMPLES_MEAN_ILD = 50

# Candidate metadata locations
META_CANDIDATES = [
    ROOT / "data" / "interim" / "courses_metadata.csv",
    ROOT / "data" / "interim" / "books_metadata.csv",
    ROOT / "data" / "interim" / "ml_videos_metadata.csv",
    ROOT / "backend" / "data" / "interim" / "courses_metadata.csv",
    ROOT / "backend" / "data" / "interim" / "books_metadata.csv",
    ROOT / "backend" / "data" / "interim" / "ml_videos_metadata.csv",
]
TITLE_COLS = ["title", "Title", "name"]

# ---------------- Helpers ----------------
def _norm_str(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        try:
            f = float(s)
            if f.is_integer():
                s = str(int(f))
        except Exception:
            pass
    return s or None

def build_title_map() -> dict:
    title_map = {}
    for p in META_CANDIDATES:
        if not p.exists():
            continue
        try:
            m = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        fname = p.name.lower()
        if "book" in fname:
            modality, id_cols = "book", ["external_id","isbn13","isbn10","volume_id"]
        elif "video" in fname:
            modality, id_cols = "video", ["external_id","youtube_id","video_id","id"]
        elif "course" in fname:
            modality, id_cols = "course", ["external_id","course_id","id"]
        else:
            continue
        title_col = next((c for c in TITLE_COLS if c in m.columns), None)
        if not title_col: 
            continue
        for _, row in m.iterrows():
            title = _norm_str(row.get(title_col))
            if not title: 
                continue
            ext_id = None
            for c in id_cols:
                ext_id = _norm_str(row.get(c))
                if ext_id: break
            if not ext_id: 
                continue
            key = (modality, ext_id)
            if key not in title_map:
                title_map[key] = title
    return title_map

def attach_titles(df: pd.DataFrame, title_map: dict) -> pd.DataFrame:
    base = df.copy()
    base["modality_norm"] = base["modality"].map(_norm_str)
    base["external_id_norm"] = base["external_id"].map(_norm_str)
    titles = []
    for mod, ext in zip(base["modality_norm"], base["external_id_norm"]):
        t = title_map.get((mod, ext)) if mod and ext else None
        titles.append(t)
    base["title_display"] = titles
    if "title" in base.columns:
        base["title_display"] = base["title_display"].where(base["title_display"].notna(), base["title"])
        base["title_display"] = base["title_display"].apply(lambda x: _norm_str(x))
    base["title_display"] = base["title_display"].fillna("(untitled)")
    return base

def load_vectors(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    vec_col = "topic_vector" if "topic_vector" in df.columns else None
    if vec_col:
        X = np.vstack(df[vec_col].apply(np.asarray).to_list()).astype(np.float32)
        feat_src = f"vector_col:{vec_col}"
    else:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        X = df[numeric_cols].astype(np.float32).to_numpy()
        feat_src = f"{len(numeric_cols)}_numeric_cols"
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, 1e-8)
    return df, Xn, feat_src

def ild_at_k(Xn, idxs):
    if len(idxs) < 2: return 0.0
    sub = Xn[idxs]
    sims = sub @ sub.T
    dists = 1.0 - sims
    triu = np.triu_indices(len(idxs), k=1)
    return float(dists[triu].mean())

# ---------------- Main ----------------
def main():
    print(f"[info] Loading topic vectors: {PARQUET}")
    df, Xn, feat_src = load_vectors(PARQUET)
    n = len(df)
    print(f"[info] Rows={n}, feature_source={feat_src}")

    title_map = build_title_map()
    df = attach_titles(df, title_map)
    non_empty = int((df['title_display'].astype(str).str.strip() != "(untitled)").sum())
    print(f"[info] Titles attached: {non_empty}/{n}")

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n, size=min(N_SAMPLES_RECALL,n), replace=False)
    hits = 0
    for i in sample_idx:
        scores = Xn[i] @ Xn.T
        if int(np.argmax(scores)) == i: hits += 1
    print(f"[metric] Self-retrieval Recall@1 on n={len(sample_idx)}: {hits/len(sample_idx):.3f}")

    sample_for_ild = rng.choice(n, size=min(N_SAMPLES_MEAN_ILD,n), replace=False)
    ilds = []
    for qi in sample_for_ild:
        scores = Xn[qi] @ Xn.T
        top_idx = np.argpartition(-scores, K)[:K]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        ilds.append(ild_at_k(Xn, top_idx[:K]))
    print(f"[metric] Mean ILD@{K} over {len(ilds)} queries: {np.mean(ilds):.3f} ± {np.std(ilds):.3f}")

    # Example with non-untitled if possible
    mask = df['title_display'].astype(str).str.strip().ne("(untitled)")
    q_idx = int(mask.idxmax()) if mask.any() else 0

    scores = Xn[q_idx] @ Xn.T
    scores[q_idx] = -1.0  # exclude self
    top_idx = np.argpartition(-scores, K)[:K]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    ild = ild_at_k(Xn, top_idx[:K])

    print("\n[example] Query item:", df.iloc[q_idx]['title_display'])
    print("[example] Top-5 neighbours:")
    demo_rows = []
    for rank, j in enumerate(top_idx[:K], 1):
        title = df.iloc[j]['title_display']
        cos = float(scores[j])
        demo_rows.append({"rank":rank,"title":title,"cosine":cos})
        print(f"  {rank:>2}. {title}   (cos={cos:.3f})")
    print(f"[metric] ILD@{K} for this list: {ild:.3f}")

    pd.DataFrame(demo_rows).to_csv("nn_demo_top5.csv",index=False)
    print("[saved] nn_demo_top5.csv")

if __name__ == "__main__":
    main()
