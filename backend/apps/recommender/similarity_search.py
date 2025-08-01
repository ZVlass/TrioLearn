import numpy as np
import pandas as pd

def cosine_similarity_matrix(vectors, query_vector):
    """
    Compute cosine similarity between a matrix of vectors and a single query vector.
    Assumes all vectors are L2-normalized.
    """
    return np.dot(vectors, query_vector)

def get_top_k(df, embeddings, query_vector, k=3):
    """
    Return the top-k items in `df` most similar to `query_vector` based on cosine similarity.
    
    Args:
        df (pd.DataFrame): Metadata table (same length as `embeddings`)
        embeddings (np.ndarray): Matrix of shape (N, D)
        query_vector (np.ndarray): Shape (D,)
        k (int): Number of top results to return

    Returns:
        pd.DataFrame: Top-k rows from `df`, with `similarity` column added
    """
    similarities = cosine_similarity_matrix(embeddings, query_vector)
    top_idx = np.argsort(similarities)[::-1][:k]
    top_scores = similarities[top_idx]

    result_df = df.iloc[top_idx].copy()
    result_df["similarity"] = top_scores
    return result_df
