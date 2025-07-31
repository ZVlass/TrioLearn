import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

from recommenders.tri_modal_recommender import recommend_tri_modal_ml


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
LEARNER_LEVEL = "intermediate"  # beginner | intermediate | advanced

# Paths to data
books_path = "./data/interim/books_metadata.csv"
courses_path = "./data/interim/courses_metadata.csv"
videos_path = "./data/interim/ml_videos_metadata.csv"

book_embs_path = "./data/embeddings/book_embeddings.npy"
course_embs_path = "./data/embeddings/course_embeddings.npy"
video_embs_path = "./data/embeddings/ml_videos_embeddings.npy"


print(" Loading data and embeddings...")

books_df = pd.read_csv(books_path)
courses_df = pd.read_csv(courses_path)
videos_df = pd.read_csv(videos_path)

book_embs = np.load(book_embs_path)
course_embs = np.load(course_embs_path)
video_embs = np.load(video_embs_path)


print("\n Enter your learning goal or topic:")
query = input(">> ").strip()


print("\n Running tri-modal recommender...\n")
results = recommend_tri_modal_ml(
    query=query,
    books_df=books_df, book_embs=book_embs,
    courses_df=courses_df, course_embs=course_embs,
    videos_df=videos_df, video_embs=video_embs,
    learner_level=LEARNER_LEVEL,
    top_k=TOP_K
)


print(f" Best Modality: {results['best_type'].upper()}\n")
print("Top Recommendations:\n")
print(results["top"][["title", "similarity"]])

print("\n Supporting Recommendations:")
for k, df in results["supporting"].items():
    print(f"\n-- {k.upper()} --")
    print(df[["title", "similarity"]])
