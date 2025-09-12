import os
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from django.conf import settings
from sentence_transformers import SentenceTransformer

# FIX: use backend.apps instead of apps
from backend.apps.recommender.tri_modal_recommender import recommend_tri_modal_ml


def _p(env_key: str, default_rel: str) -> Path:
    """Env override, else BASE_DIR / default_rel"""
    return Path(os.getenv(env_key, Path(settings.BASE_DIR) / default_rel))


@lru_cache(maxsize=1)
def _load_data():
    data_dir = _p("DATA_DIR", "data")

    books_csv   = _p("BOOKS_CSV",   "data/interim/books_metadata.csv")
    courses_csv = _p("COURSES_CSV", "data/interim/courses_metadata.csv")
    videos_csv  = _p("VIDEOS_CSV",  "data/interim/videos_metadata.csv")

    book_embs_npy   = _p("BOOK_EMBS_NPY",   "data/embeddings/book_embeddings.npy")
    course_embs_npy = _p("COURSE_EMBS_NPY", "data/embeddings/course_embeddings.npy")
    video_embs_npy  = _p("VIDEO_EMBS_NPY",  "data/embeddings/ml_videos_embeddings.npy")

    missing = [p for p in [books_csv, courses_csv, videos_csv,
                           book_embs_npy, course_embs_npy, video_embs_npy]
               if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing data files:\n  " + "\n  ".join(str(m) for m in missing)
        )

    books_df   = pd.read_csv(books_csv)
    courses_df = pd.read_csv(courses_csv)
    videos_df  = pd.read_csv(videos_csv)

    book_embs   = np.load(book_embs_npy)
    course_embs = np.load(course_embs_npy)
    video_embs  = np.load(video_embs_npy)

    return {
        "books_df": books_df,
        "courses_df": courses_df,
        "videos_df": videos_df,
        "book_embs": book_embs,
        "course_embs": course_embs,
        "video_embs": video_embs,
    }


@lru_cache(maxsize=1)
def _get_model():
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


def get_recommendations(query: str, learner_level: str = "intermediate", top_k: int = 3):
    data = _load_data()
    _ = _get_model()
    return recommend_tri_modal_ml(
        query=query,
        books_df=data["books_df"],   book_embs=data["book_embs"],
        courses_df=data["courses_df"], course_embs=data["course_embs"],
        videos_df=data["videos_df"],   video_embs=data["video_embs"],
        learner_level=learner_level,
        top_k=top_k,
    )
