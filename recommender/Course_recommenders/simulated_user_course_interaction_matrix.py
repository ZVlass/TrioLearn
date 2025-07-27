import pandas as pd
import numpy as np
import random
from scipy.sparse import lil_matrix

# Load preprocessed input files
oulad_df = pd.read_csv("./data/interim/oulad_clustered.csv")
coursera_df = pd.read_csv("./data/interim/coursera_normalized.csv")

# Parameters
num_users = len(oulad_df)
num_courses = len(coursera_df)
user_clusters = oulad_df["user_cluster"].values
user_ids = oulad_df["id_student"].values

# Prepare course feature arrays
rating_weight = {'low': 0.2, 'medium': 0.6, 'high': 1.0}

# Map and clean
rating_scores = coursera_df["rating_level"].map(rating_weight).fillna(0.5).values
popularities = coursera_df["popularity"].fillna(0.5).values
difficulties = coursera_df["difficulty_num"].fillna(1).values

# Initialize interaction matrix
interaction_matrix = lil_matrix((num_users, num_courses), dtype=np.int8)

# Simulate interactions
for user_idx, cluster in enumerate(user_clusters):
    for course_idx in range(num_courses):
        rating_score = rating_scores[course_idx]
        popularity = popularities[course_idx]
        difficulty = difficulties[course_idx]

        # Heuristic for difficulty preference by cluster
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

        # Simulate interaction probability
        interaction_prob = 0.4 * rating_score + 0.4 * popularity + 0.2 * difficulty_pref

        if random.random() < interaction_prob * 0.5:
            interaction_matrix[user_idx, course_idx] = 1

# Convert to DataFrame
interaction_df = pd.DataFrame.sparse.from_spmatrix(
    interaction_matrix,
    index=user_ids,
    columns=[f"course_{i}" for i in range(num_courses)]
)

# Save result
interaction_df.to_csv("./data/interim/simulated_user_course_interactions.csv", index_label="id_student")
print("Saved: data/interim/simulated_user_course_interactions.csv")
