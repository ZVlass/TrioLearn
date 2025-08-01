import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from apps.recommender.tri_modal_recommender import recommend_tri_modal_ml

# --- Load model and embeddings once ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../data"))

books_df = pd.read_csv(os.path.join(DATA_DIR, "interim/books_metadata.csv"))
courses_df = pd.read_csv(os.path.join(DATA_DIR, "interim/courses_metadata.csv"))
videos_df = pd.read_csv(os.path.join(DATA_DIR, "interim/ml_videos_metadata.csv"))

book_embs = np.load(os.path.join(DATA_DIR, "embeddings/book_embeddings.npy"))
course_embs = np.load(os.path.join(DATA_DIR, "embeddings/course_embeddings.npy"))
video_embs = np.load(os.path.join(DATA_DIR, "embeddings/ml_videos_embeddings.npy"))

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_recommendations(query, learner_level="intermediate", top_k=3):
    return recommend_tri_modal_ml(
        query=query,
        books_df=books_df, book_embs=book_embs,
        courses_df=courses_df, course_embs=course_embs,
        videos_df=videos_df, video_embs=video_embs,
        learner_level=learner_level,
        top_k=top_k
    )


