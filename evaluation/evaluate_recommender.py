
"""
evaluate_recommender.py
A SIMPLE, self-contained evaluator for the TrioLearn recommender.
- Computes Precision@K (with manual labels placeholders)
- Computes Intra-List Diversity (ILD) using topic vectors if available, else TF-IDF fallback
- Computes Title-Echo@K (how often titles simply mirror the query)
Outputs:
- Console logs
- CSV: evaluation_results.csv (per-query metrics)
- CSV: recommendations_<timestamp>.csv (all recommended items for screenshots)
Usage:
    python evaluate_recommender.py
Optional ENV:
    TOPIC_VECS: path to all_topic_vectors.parquet (default: data/topics/all_topic_vectors.parquet)
    TOP_K: integer (default: 5)
    LEARNER_LEVEL: beginner|intermediate|advanced (default: intermediate)
"""
import os
import sys
import json
import time
import math
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# sklearn is used only for simple text features for Title-Echo & ILD fallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# Setup Django
from django.conf import settings
if not settings.configured:
    settings.configure(BASE_DIR=str(ROOT))


# Import the project recommender entrypoint
try:
    from backend.apps.recommender.engine import get_recommendations

except Exception as e:
    print("[error] Failed to import get_recommendations from apps.recommender.engine")
    raise

# ----------------------------
# Config
# ----------------------------
TOP_K = 5
LEARNER_LEVEL = "intermediate"
TOPIC_VECS_PATH = Path(r"C:\Users\jvlas\source\repos\TrioLearn\backend\data\topics\all_topic_vectors.parquet")
TEST_QUERIES = [
    "neural networks basics",
    "data visualization in Python",
    "transformers and attention for NLP",
]
# ----------------------------
# Manual relevance labels (PLACEHOLDER)
# Fill this dict with booleans for each query corresponding to the *best* modality list.
# Example: if TOP_K=5, provide 5 booleans in the list. Leave as-is to compute Precision@K later.
# ----------------------------
MANUAL_LABELS: Dict[str, List[bool]] = {
    # "neural networks basics": [True, True, False, True, False],
    # "data visualization in Python": [True, False, True, False, False],
    # "transformers and attention for NLP": [True, True, True, False, True],
}

# ----------------------------
# Utilities
# ----------------------------
def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def _vectorize_titles(texts: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    vec = TfidfVectorizer(lowercase=True, stop_words="english", max_features=5000)
    X = vec.fit_transform(texts)
    return vec, X

def title_echo_rate(query: str, titles: List[str], threshold: float = 0.6) -> float:
    """Title-Echo@K: fraction of titles with high cosine to the query (TF-IDF)."""
    if not titles:
        return 0.0
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2))
    X = vec.fit_transform([query] + titles)
    sims = cosine_similarity(X[0], X[1:]).ravel()
    echo = (sims >= threshold).sum()
    return echo / len(titles)

def ild_from_topic_vectors(recs_df: pd.DataFrame, topics_df: pd.DataFrame) -> Tuple[float, int]:
    """
    Compute ILD using topic vectors if we can match by 'global_id'.
    Returns (ILD, matched_count). Falls back to text if not enough matches.
    """
    if "global_id" not in recs_df.columns:
        return float("nan"), 0
    if "global_id" not in topics_df.columns:
        return float("nan"), 0

    # pick numeric topic columns
    topic_cols = _numeric_cols(topics_df)
    if not topic_cols:
        return float("nan"), 0

    sub = topics_df.set_index("global_id").reindex(recs_df["global_id"]).dropna()
    if len(sub) < 2:
        return float("nan"), len(sub)

    V = sub[topic_cols].to_numpy()
    # Cosine distance = 1 - cosine_similarity
    S = cosine_similarity(V)
    # take upper triangle mean distance
    triu = np.triu_indices(len(V), k=1)
    dists = 1.0 - S[triu]
    return float(np.mean(dists)), len(sub)

def ild_from_text(recs_df: pd.DataFrame) -> float:
    """TF-IDF title+description fallback ILD."""
    texts = (recs_df["title"].fillna("") + " " + recs_df.get("description", pd.Series([""]*len(recs_df))).fillna("")).tolist()
    if len(texts) < 2:
        return 0.0
    _, X = _vectorize_titles(texts)
    S = cosine_similarity(X)
    triu = np.triu_indices(X.shape[0], k=1)
    dists = 1.0 - S[triu]
    return float(np.mean(dists))

def precision_at_k(labels: List[bool], k: int) -> float:
    if not labels:
        return float("nan")
    k = min(k, len(labels))
    return sum(labels[:k]) / float(k)

def flatten_results(query: str, results: dict, top_k: int) -> pd.DataFrame:
    """
    Produce a single DataFrame of up to 3*top_k rows across modalities to aid screenshots.
    Keeps columns: query, modality, rank, title, similarity_score, global_id (if present), url (if present)
    """
    frames = []
    order = [(results["best_type"], results["top"])]
    # add supporting
    for mod, df in results["supporting"].items():
        order.append((mod, df))

    for modality, df in order:
        df = df.copy().head(top_k).reset_index(drop=True)
        df["query"] = query
        df["modality"] = modality
        df["rank"] = np.arange(1, len(df)+1)
        keep = ["query", "modality", "rank", "title"]
        for col in ["similarity_score", "global_id", "url", "external_id", "platform", "provider", "level"]:
            if col in df.columns:
                keep.append(col)
        frames.append(df[keep])
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["query","modality","rank","title","similarity_score","global_id","url"])

# ----------------------------
# Main
# ----------------------------
def main():
    # Try loading topic vectors
    topics_df = None
    if TOPIC_VECS_PATH.exists():
        try:
            topics_df = pd.read_parquet(TOPIC_VECS_PATH)
            # Ensure a global_id column is present; best-effort rename if needed
            if "global_id" not in topics_df.columns:
                # try common fallbacks
                for alt in ["id", "item_id", "course_id"]:
                    if alt in topics_df.columns:
                        topics_df = topics_df.rename(columns={alt: "global_id"})
                        break
            print(f"[info] Loaded topic vectors from: {TOPIC_VECS_PATH} (rows={len(topics_df)})")
        except Exception as e:
            print(f"[warn] Could not read topic vectors at {TOPIC_VECS_PATH}: {e}")
            topics_df = None
    else:
        print(f"[warn] Topic vectors parquet not found at {TOPIC_VECS_PATH}. Will use TF-IDF fallback for ILD.")

    # Collect all recommendations and metrics
    all_recs = []
    metrics_rows = []

    for q in TEST_QUERIES:
        print("\n" + "="*72)
        print(f"Query: {q}")
        results = get_recommendations(q, learner_level=LEARNER_LEVEL, top_k=TOP_K)

        # Best modality list for headline metrics
        best_mod = results["best_type"]
        top_df = results["top"].copy().reset_index(drop=True)

        # Title-Echo@K on top list
        titles = top_df["title"].astype(str).tolist()
        echo_rate = title_echo_rate(q, titles, threshold=0.6)

        # ILD on top list (topic vectors first, else text)
        if topics_df is not None:
            ild, matched = ild_from_topic_vectors(top_df, topics_df)
            if not math.isfinite(ild):
                ild = ild_from_text(top_df)
                ild_source = f"TF-IDF fallback (matched={matched})"
            else:
                ild_source = f"Topic vectors (matched={matched})"
        else:
            ild = ild_from_text(top_df)
            ild_source = "TF-IDF fallback"

        # Precision@K (if labels provided)
        labels = MANUAL_LABELS.get(q, [])
        p_at_k = precision_at_k(labels, TOP_K) if labels else float("nan")

        # Log to console
        print(f"[best modality] {best_mod}")
        print("[top results]")
        cols_to_show = ["title"]
        if "similarity_score" in top_df.columns:
            cols_to_show.append("similarity_score")
        print(top_df[cols_to_show].head(TOP_K).to_string(index=False))

        if labels:
            print(f"Precision@{TOP_K}: {p_at_k:.3f} (from manual labels)")
        else:
            print(f"Precision@{TOP_K}: N/A (add labels in MANUAL_LABELS to compute)")
        print(f"Intra-List Diversity (ILD@{TOP_K}): {ild:.3f} [{ild_source}]")
        print(f"Title-Echo@{TOP_K}: {echo_rate:.3f} (lower is better)")

        # Save recommendations (for screenshots)
        flat = flatten_results(q, results, TOP_K)
        all_recs.append(flat)

        # Metrics row
        metrics_rows.append({
            "query": q,
            "best_modality": best_mod,
            f"precision@{TOP_K}": p_at_k,
            f"ild@{TOP_K}": ild,
            f"title_echo@{TOP_K}": echo_rate,
            "ild_source": ild_source
        })

    # Write outputs
    out_metrics = pd.DataFrame(metrics_rows)
    out_recs = pd.concat(all_recs, ignore_index=True) if all_recs else pd.DataFrame()

    metrics_path = Path(f"csv/evaluation_results.csv")
    recs_path = Path(f"csv/recommendations_{now_ts()}.csv")
    out_metrics.to_csv(metrics_path, index=False)
    out_recs.to_csv(recs_path, index=False)

    print("\n" + "-"*72)
    print(f"[saved] Metrics -> {metrics_path.resolve()}")
    print(f"[saved] Recommendations -> {recs_path.resolve()}")
    print("Tip: Add manual labels in MANUAL_LABELS to compute Precision@K.\n"
          "     You can also change TEST_QUERIES, TOP_K, and LEARNER_LEVEL at the top of this script.\n"
          "     Set TOPIC_VECS env var if your parquet lives elsewhere.")

if __name__ == "__main__":
    main()
