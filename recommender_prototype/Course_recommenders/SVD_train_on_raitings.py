

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import os
import pickle

#  Load and clean the ratings

# Load triplet file (user_id, course_id, rating)
ratings_df = pd.read_csv("./data/processed/user_course_ratings_triplet_2.csv")

# Drop missing or non-numeric ratings
ratings_df = ratings_df.dropna(subset=["rating"])
ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce")

# Remove invalid ratings
ratings_df = ratings_df[ratings_df["rating"] > 0]
ratings_df["rating"] = ratings_df["rating"].clip(upper=5)

print(f" Loaded {len(ratings_df)} valid user-course-rating rows")

# Prepare Surprise dataset 

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[["user_id", "course_id", "rating"]], reader)

# Split using Surprise's safe splitter (ensures all IDs appear in training)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train SVD 

model = SVD()
model.fit(trainset)

#  Predict and Evaluate 

predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

#  Precision@K, Recall@K 

def precision_recall_at_k(predictions, k=20):
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = [], []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]
        n_rel = sum((true_r >= 4) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= 4) for (est, _) in top_k)
        n_rel_and_rec_k = sum(((true_r >= 4) and (est >= 4)) for (est, true_r) in top_k)

        precision = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recall = n_rel_and_rec_k / n_rel if n_rel else 0

        precisions.append(precision)
        recalls.append(recall)

    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)

precision_k, recall_k = precision_recall_at_k(predictions, k=10)

#  Print Result

print("\n Evaluation Results:")
print(f"RMSE         : {rmse:.4f}")
print(f"Precision@10 : {precision_k:.4f}")
print(f"Recall@10    : {recall_k:.4f}")

# Save model to disk

with open("./outputs/models/svd_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("SVD model saved to models/svd_model.pkl")



