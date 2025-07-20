

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from recommender.preprocess import preprocess_user_features
from recommender.clustering import cluster_users, build_cluster_profiles

# Load & preprocess
users = preprocess_user_features(DATA_DIR / "oulad_user_features.csv")

# Cluster
labels, kmeans = cluster_users(users, n_clusters=5)
users['cluster'] = labels

# Build profiles
embs = np.load(DATA_DIR / "course_embeddings.npy")
pop = load_real_popularity()  # you’d implement this
profiles = build_cluster_profiles(embs, pop, labels, top_n=10)

# Save outputs
users.to_csv(OUTPUT_DIR / "user_clusters.csv", index=False)
np.save(OUTPUT_DIR / "cluster_profiles.npy", np.vstack(list(profiles.values())))
