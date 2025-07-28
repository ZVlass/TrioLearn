
import pandas as pd
import numpy as np
import random
from scipy.sparse import lil_matrix
import os

# Load inputs
oulad_df = pd.read_csv("./data/interim/oulad_clustered.csv")
coursera_df = pd.read_csv("./data/interim/coursera_normalized.csv")

# Parameters
num_users = len(oulad_df)
num_courses = len(coursera_df)
user_clusters = oulad_df["user_cluster"].values
user_ids = oulad_df["id_student"].values

# Course features (precomputed)
rating_weight = {'low': 0.2, 'medium': 0.6, 'high': 1.0}
rating_scores = coursera_df["rating_level"].map(rating_weight).fillna(0.5).values
popularities = coursera_df["popularity"].fillna(0.5).values
difficulties = coursera_df["difficulty_num"].fillna(1).values

# Initialize ratings matrix (using float now)
ratings_matrix = lil_matrix((num_users, num_courses), dtype=np.float32)

# Generate synthetic ratings
def simulate_rating(prob: float) -> float:
    """
    Turn a probability (0–1) into a synthetic rating (1–5),
    favoring high ratings when probability is high.
    """
    if prob < 0.2:
        return 1.0
    elif prob < 0.4:
        return 2.0
    elif prob < 0.6:
        return 3.0
    elif prob < 0.8:
        return 4.0
    else:
        return 5.0

# Simulate user-course ratings
for user_idx, cluster in enumerate(user_clusters):
    for course_idx in range(num_courses):
        rating_score = rating_scores[course_idx]
        popularity = popularities[course_idx]
        difficulty = difficulties[course_idx]

        # Heuristic: difficulty preference by cluster
        if cluster == 0:
            difficulty_pref = 0.9 if difficulty == 0 else 0.4
        elif cluster == 1:
            difficulty_pref = 0.6 if difficulty == 1 else 0.3
        elif cluster == 2:
            difficulty_pref = 0.1
        elif cluster == 3:
            difficulty_pref = 0.7 if difficulty in [0, 1] else 0.3
        else:
            difficulty_pref = 0.4

        # Final interaction probability
        prob = 0.4 * rating_score + 0.4 * popularity + 0.2 * difficulty_pref

        if random.random() < prob * 0.5:
            synthetic_rating = simulate_rating(prob)
            ratings_matrix[user_idx, course_idx] = synthetic_rating

# Convert to long triplet format
ratings_df = pd.DataFrame.sparse.from_spmatrix(
    ratings_matrix,
    index=user_ids,
    columns=[f"course_{i}" for i in range(num_courses)]
).stack().reset_index()
ratings_df.columns = ["user_id", "course_id", "rating"]

# Save triplets
os.makedirs("data/processed", exist_ok=True)
ratings_df.to_csv("data/processed/user_course_ratings_triplet_2.csv", index=False)
print("Saved: data/processed/user_course_ratings_triplet_2.csv")
