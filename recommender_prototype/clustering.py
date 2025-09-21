

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_users(user_df: pd.DataFrame, n_clusters: int = 5) -> np.ndarray:

    """Cluster users by activity features and return labels."""

    features = user_df.drop(columns=['id_student'])
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(features)
    sil_score = silhouette_score(features, labels)
    print(f"Silhouette score: {sil_score:.3f}")
    return labels, model


def build_cluster_profiles(course_embeddings: np.ndarray, popularity: np.ndarray, labels: np.ndarray, top_n: int = 10) -> dict:

    """Compute per-cluster profile embeddings by averaging top-N popular courses."""

    cluster_profiles = {}
    n_clusters = labels.max() + 1
    for c in range(n_clusters):
        # Select top-N globally popular course indices
        top_idxs = np.argsort(popularity)[-top_n:]
        cluster_profiles[c] = course_embeddings[top_idxs].mean(axis=0)
    return cluster_profiles