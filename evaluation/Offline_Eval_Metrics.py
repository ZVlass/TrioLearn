from backend.apps.recommender.engine import get_recommendations
import pandas as pd
import numpy as np

# Load topic vectors for diversity calculation
topic_vectors = pd.read_parquet("data/topics/all_topic_vectors.parquet")

# Example queries
queries = [
    "neural networks basics",
    "data visualization in Python",
    "transformers and attention for NLP"
]

def evaluate_query(q):
    recs = get_recommendations(q, top_k=5)

    # Precision@K (manual relevance marking required)
    print(f"\nQuery: {q}")
    print("Top recommendations:")
    print(recs["top"][["title", "similarity_score"]])

    # Diversity
    idxs = recs["top"].index
    vecs = topic_vectors.loc[idxs].values
    dists = 1 - np.inner(vecs, vecs)  # cosine distances
    diversity = dists[np.triu_indices(len(vecs), k=1)].mean()
    print(f"Intra-list diversity: {diversity:.3f}")

for q in queries:
    evaluate_query(q)
