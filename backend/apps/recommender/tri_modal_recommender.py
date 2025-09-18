import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    from apps.recommender.similarity_search import get_top_k
    from apps.recommender.ml_selector import predict_best_modality
except ModuleNotFoundError:
    # if we run evaluation
    from backend.apps.recommender.similarity_search import get_top_k
    from backend.apps.recommender.ml_selector import predict_best_modality


# Load your embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def recommend_tri_modal_ml(
    query,
    books_df, book_embs,
    courses_df, course_embs,
    videos_df, video_embs,
    learner_level="intermediate",
    top_k=3
):
    """
    Recommend top resources across all three modalities, prioritizing the best one using ML.

    Args:
        query (str): User input query
        *_df (pd.DataFrame): Metadata tables
        *_embs (np.ndarray): Embedding matrices
        learner_level (str): 'beginner' | 'intermediate' | 'advanced'
        top_k (int): Number of top results per modality

    Returns:
        dict: {
            'best_type': str,
            'top': pd.DataFrame,
            'supporting': dict[str, pd.DataFrame]
        }
    """
    # Embed the query
    query_vec = model.encode(query, normalize_embeddings=True)

    # Predict best modality
    
    best_type = predict_best_modality(query, learner_level)  # e.g., 'video', 'book', 'course'

    # Map model output to the actual keys in results
    key_map = {"video": "videos", "book": "books", "course": "courses"}
    best_key = key_map[best_type]


    # Compute top-k for all modalities
    top_courses = get_top_k(courses_df, course_embs, query_vec, k=top_k)
    top_books = get_top_k(books_df, book_embs, query_vec, k=top_k)
    top_videos = get_top_k(videos_df, video_embs, query_vec, k=top_k)

    # Step 4: Package results
    results = {
        "courses": top_courses,
        "books": top_books,
        "videos": top_videos
    }

    support_frames = [df for k, df in results.items() if k != best_key and df is not None and not df.empty]
    surprise_item = None
    if support_frames:
        pool = pd.concat([df.head(min(25, len(df))) for df in support_frames], ignore_index=True)
        surprise_item = pool.sample(1, random_state=42).to_dict(orient="records")[0]


    print("best_type predicted by model:", best_type)
    print("Available keys in results:", list(results.keys()))

    return {
    "best_type": best_type,
    "top": results[best_key],
    "supporting": {k: v for k, v in results.items() if k != best_key},
    "surprise": surprise_item,  
}

# ==== TRIOMODAL DEBUG UTILITIES (safe to keep; no runtime side-effects) ====
import os, json, traceback
from pathlib import Path

def _exists(p):
    try:
        return Path(p).exists()
    except Exception:
        return False

def _readable_path(p):
    try:
        p = str(p)
        return p if len(p) < 200 else p[:200] + "…"
    except Exception:
        return str(p)

def _model_name_from_obj(m):
    # Best-effort: parse model name from repr, e.g. SentenceTransformer("all-MiniLM-L6-v2")
    try:
        s = str(m)
        import re
        mt = re.search(r'SentenceTransformer\("([^"]+)"\)', s)
        return mt.group(1) if mt else s
    except Exception:
        return str(type(m))

def _collect_env_status():
    env = {
        "QUERY_EMBED_MODEL": os.getenv("QUERY_EMBED_MODEL"),
        "COURSE_EMBED_NPY": os.getenv("COURSE_EMBED_NPY"),
        "VIDEO_EMBED_NPY": os.getenv("VIDEO_EMBED_NPY"),
        "BOOKS_EMBED_OUT": os.getenv("BOOKS_EMBED_OUT"),
        "COURSE_EMBED_META_CSV": os.getenv("COURSE_EMBED_META_CSV"),
        "VIDEO_EMBED_META_CSV": os.getenv("VIDEO_EMBED_META_CSV"),
        "BOOKS_META_CSV": os.getenv("BOOKS_META_CSV"),
    }
    # derive books_embeddings.npy from folder
    if env["BOOKS_EMBED_OUT"]:
        env["BOOKS_EMBED_FILE"] = str(Path(env["BOOKS_EMBED_OUT"]) / "books_embeddings.npy")
    else:
        env["BOOKS_EMBED_FILE"] = None

    checks = []
    for k, v in env.items():
        if k.endswith("_NPY") or k.endswith("_CSV") or k.endswith("_FILE") or k.endswith("_OUT"):
            checks.append((k, bool(v) and _exists(v), _readable_path(v)))
        else:
            # model name presence only
            checks.append((k, v is not None and len(v) > 0, v))
    return env, checks

def _safe_call_recommender(query="transformers for nlp", level="intermediate", k=5):
    # Try to call whatever public function this module exposes
    # Common names: recommend / recommend_query / get_recommendations
    fn = None
    for name in ("recommend", "recommend_query", "get_recommendations"):
        if name in globals() and callable(globals()[name]):
            fn = globals()[name]
            break
    if fn is None:
        return {"ok": False, "error": "No recommender entrypoint found (expected one of: recommend/recommend_query/get_recommendations)."}
    try:
        out = fn(query, level=level, top_k=k) if "level" in fn.__code__.co_varnames else fn(query, top_k=k)
        return {"ok": True, "result": out}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}

def debug_tri_modal_health(query="transformers for nlp", level="intermediate", k=5, verbose=True):
    """
    Prints a compact health report to help identify why no recommendations are showing.
    Usage (from project root):
      python manage.py shell -c "from apps.recommender.tri_modal_recommender import debug_tri_modal_health; debug_tri_modal_health()"
    """
    report = {}

    # 1) Model alignment
    env_model = os.getenv("QUERY_EMBED_MODEL", "(not set)")
    used_model = _model_name_from_obj(globals().get("model", "(model not initialised)"))
    report["model_env"] = env_model
    report["model_used_by_module"] = used_model
    report["model_mismatch_warning"] = (env_model not in (None, "", "(not set)")) and (env_model != used_model)

    # 2) Paths and existence
    env_values, checks = _collect_env_status()
    report["env"] = env_values
    report["path_checks"] = [{"key": k, "exists_ok": ok, "value": v} for (k, ok, v) in checks]

    # 3) Sanity call to the recommender
    sanity = _safe_call_recommender(query=query, level=level, k=k)
    report["recommender_call"] = {"ok": sanity.get("ok", False)}
    if not sanity["ok"]:
        report["recommender_call"]["error"] = sanity.get("error")
        report["recommender_call"]["trace"] = sanity.get("trace")
    else:
        res = sanity["result"] or {}
        # Normalize keys we care about
        keys = ["best_type", "courses", "books", "videos", "top", "supporting", "surprise"]
        summary = {}
        for key in keys:
            val = res.get(key, None)
            if isinstance(val, list):
                summary[key] = f"list[{len(val)}]"
            elif hasattr(val, "shape"):
                try:
                    summary[key] = f"array shape={val.shape}"
                except Exception:
                    summary[key] = "array"
            elif val is None:
                summary[key] = None
            else:
                # brief scalar/dict indication
                summary[key] = type(val).__name__
        report["recommender_summary"] = summary

        # Specific red flags
        report["flags"] = {
            "no_items_any_modality": all((not res.get("courses"), not res.get("books"), not res.get("videos"))),
            "missing_best_type": "best_type" not in res or not res.get("best_type"),
            "surprise_missing_modality": bool(res.get("surprise")) and ("modality" not in res["surprise"]),
        }

    if verbose:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return report

# Optional: quick CLI run when executing the module directly
if __name__ == "__main__":
    debug_tri_modal_health()
# ==== END DEBUG UTILITIES ====
