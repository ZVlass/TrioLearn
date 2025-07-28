

import pickle
import pandas as pd
from surprise import Dataset, Reader
import json
import os

#  Load SVD model 
with open("./outputs/models/svd_model.pkl", "rb") as f:
    model = pickle.load(f)

print(" SVD model loaded.")

#  Load ratings and rebuild trainset
ratings_df = pd.read_csv("data/processed/user_course_ratings_triplet_2.csv")
ratings_df = ratings_df.dropna(subset=["rating"])
ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce")
ratings_df = ratings_df[ratings_df["rating"] > 0]
ratings_df["rating"] = ratings_df["rating"].clip(upper=5)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[["user_id", "course_id", "rating"]], reader)
trainset = data.build_full_trainset()

print("Trainset rebuilt.")

# Load course metadata for enrichment 
metadata_df = pd.read_csv("data/interim/coursera_normalized.csv")
metadata_df["course_id"] = ["course_" + str(i) for i in metadata_df.index]
metadata_df = metadata_df[["course_id", "Title"]]

#  Select 10 user IDs 
user_ids = ratings_df["user_id"].unique()[:10]  # Select first 10 unique users
print(f"Generating top-N for users: {user_ids.tolist()}")

# Generate top-N recommendations per selected user 
def get_top_n_for_users(model, trainset, users_raw, n=10):
    all_items = set(trainset.all_items())
    top_n = {}

    for uid_raw in users_raw:
        try:
            uid_inner = trainset.to_inner_uid(uid_raw)
        except ValueError:
            continue  # skip unknown users

        rated_items = set(j for (j, _) in trainset.ur[uid_inner])
        unseen_items = all_items - rated_items

        predictions = [
            (trainset.to_raw_iid(iid), model.predict(uid_raw, trainset.to_raw_iid(iid)).est)
            for iid in unseen_items
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n[uid_raw] = predictions[:n]

    return top_n

top_n = get_top_n_for_users(model, trainset, users_raw=user_ids, n=10)

# Format and enrich with course titles 
user_json_outputs = {}

for user_id, recs in top_n.items():
    enriched = []
    for course_id, score in recs:
        title = metadata_df.loc[metadata_df["course_id"] == course_id, "Title"].values
        title = title[0] if len(title) > 0 else "Unknown"
        enriched.append({
            "course_id": course_id,
            "title": title,
            "predicted_rating": round(score, 3)
        })
    user_json_outputs[user_id] = enriched

# Save as individual JSON files 
os.makedirs("data/api/user_recommendations", exist_ok=True)

for user_id, recs in user_json_outputs.items():
    out_path = f"data/api/user_recommendations/{user_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2, ensure_ascii=False)

print("JSON recommendations saved to data/api/user_recommendations/")
