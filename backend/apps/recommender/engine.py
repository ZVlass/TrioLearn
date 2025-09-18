# apps/recommender/engine.py
from __future__ import annotations

import os, math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------- Basics ----------------
EPS = 1e-12
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def _p(env_key: str, default_rel: str) -> Path:
    from django.conf import settings
    return Path(os.getenv(env_key, Path(settings.BASE_DIR) / default_rel))

# ---------------- JSON safety ----------------
def _json_safe_scalar(x):
    if isinstance(x, (np.floating,)): x = float(x)
    if isinstance(x, (np.integer,)):  x = int(x)
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x): return None
    return x

def json_safe(obj):
    if isinstance(obj, dict):  return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [json_safe(v) for v in obj]
    if isinstance(obj, tuple): return [json_safe(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return json_safe(obj.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return {k: json_safe(v) for k, v in obj.replace([np.nan, np.inf, -np.inf], None).to_dict().items()}
    return _json_safe_scalar(obj)

# ---------------- Math helpers ----------------
def _safe_normalize(X: np.ndarray, axis=1) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=axis, keepdims=True) + EPS)

def _cosine_sim(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    if q.ndim == 1: q = q[None, :]
    sims = (q @ M.T).ravel()
    return np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)

# ---------------- Model & data ----------------
@lru_cache(maxsize=1)
def _load_model():
    name = os.getenv("EMBED_MODEL", DEFAULT_MODEL)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)

@lru_cache(maxsize=1)
def _load_data() -> Dict[str, dict]:
    books_csv   = _p("BOOKS_CSV",   "data/interim/books_metadata.csv")
    courses_csv = _p("COURSES_CSV", "data/interim/courses_metadata_clean.csv")
    videos_csv  = _p("VIDEOS_CSV",  "data/interim/videos_metadata_clean.csv")

    book_embs   = _p("BOOK_EMBED_NPY",   "data/embeddings/books_embeddings.npy")
    course_embs = _p("COURSE_EMBED_NPY", "data/embeddings/courses_embeddings.npy")
    video_embs  = _p("VIDEO_EMBED_NPY",  "data/embeddings/videos_embeddings.npy")

    books_df, courses_df, videos_df = pd.read_csv(books_csv), pd.read_csv(courses_csv), pd.read_csv(videos_csv)
    book_E   = _safe_normalize(np.load(book_embs),   axis=1)
    course_E = _safe_normalize(np.load(course_embs), axis=1)
    video_E  = _safe_normalize(np.load(video_embs),  axis=1)

    print(f"books:   {books_csv} {books_df.shape} embs: {book_E.shape}")
    print(f"courses: {courses_csv} {courses_df.shape} embs: {course_E.shape}")
    print(f"videos:  {videos_csv} {videos_df.shape} embs: {video_E.shape}")

    return {
        "books":   {"df": books_df,   "E": book_E},
        "courses": {"df": courses_df, "E": course_E},
        "videos":  {"df": videos_df,  "E": video_E},
    }

def _embed_query(text: str) -> np.ndarray:
    vec = _load_model().encode([text], normalize_embeddings=False)
    return _safe_normalize(np.asarray(vec), axis=1)[0]

# ---------------- Record formatting ----------------
def _canon(row: dict) -> dict:
    href = row.get("url") or row.get("link") or row.get("webpage") or row.get("preview_url") or "#"
    title = row.get("title") or row.get("name") or row.get("course_title") or "Untitled"
    display_id = row.get("external_id") or row.get("id") or row.get("isbn") or ""
    row["href"], row["display_title"], row["display_id"] = href, title, display_id
    return row

def _format_records(df: pd.DataFrame, idxs: np.ndarray, modality: str, sims: np.ndarray) -> List[dict]:
    subset = df.iloc[idxs].copy().replace([np.nan, np.inf, -np.inf], None)
    out: List[dict] = []
    for rank, (i, s) in enumerate(zip(idxs, sims[idxs]), start=1):
        base = subset.loc[i].to_dict()
        base.update({"modality": modality, "rank": int(rank), "score": float(s) if np.isfinite(s) else 0.0})
        out.append(json_safe(_canon(base)))
    return out

# ---------------- Recommender core ----------------
def _recommend_for(modality: str, qv: np.ndarray, k: int) -> List[dict]:
    store = _load_data()[modality]
    sims = _cosine_sim(qv, store["E"])
    k = max(1, min(int(k or 10), len(sims)))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return _format_records(store["df"], top, modality, sims)

def _predict_best_type(query: str, level: Optional[str]) -> Optional[str]:
    try:
        from apps.recommender.ml_selector import predict_best_modality
        g = (predict_best_modality(query, level or "") or "").lower()
        if g.startswith("book"): return "books"
        if g.startswith("course"): return "courses"
        if g.startswith("video"): return "videos"
    except Exception:
        pass
    return None

def _pick_surprise(payload: dict, best: Optional[str]):
    lists = {k: payload.get(k, []) for k in ("books", "courses", "videos")}
    if best in lists:
        others = [k for k in lists if k != best and lists[k]]
        if not others: return None
        # pick the other modality with highest first-item score (simple & effective)
        key = max(others, key=lambda k: float(lists[k][0].get("score", 0.0)))
        it = dict(lists[key][0]); it["surprise_reason"] = f"Top {key} to diversify against {best}."
        return json_safe(it)
    # no best -> pick highest first-item score overall
    nonempty = [k for k in lists if lists[k]]
    if not nonempty: return None
    key = max(nonempty, key=lambda k: float(lists[k][0].get("score", 0.0)))
    it = dict(lists[key][0]); it["surprise_reason"] = f"Top {key} overall."
    return json_safe(it)

# ---------------- Public API ----------------
def get_recommendations(query: str, level: Optional[str] = None, top_k: int = 10) -> Dict[str, List[dict]]:
    if not query or not query.strip():
        return {"best_type": None, "books": [], "courses": [], "videos": [], "surprise": None}

    qv = _embed_query(query.strip())
    payload = {
        "books":   _recommend_for("books", qv, top_k),
        "courses": _recommend_for("courses", qv, top_k),
        "videos":  _recommend_for("videos", qv, top_k),
    }
    payload["best_type"] = _predict_best_type(query, level)
    payload["surprise"] = _pick_surprise(payload, payload["best_type"])
    return json_safe(payload)

# Backwards-compat alias
def recommend_query(query: str, level: Optional[str] = None, top_k: int = 10):
    return get_recommendations(query, level, top_k)
