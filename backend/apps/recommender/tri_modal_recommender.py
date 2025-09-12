import numpy as np
from sentence_transformers import SentenceTransformer
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

    print("best_type predicted by model:", best_type)
    print("Available keys in results:", list(results.keys()))

    return {
        "best_type": best_type,
        "top": results[best_key],
        "supporting": {
            k: v for k, v in results.items() if k != best_key
        }
    }
