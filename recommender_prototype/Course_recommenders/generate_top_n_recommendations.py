
import pickle
import pandas as pd
from surprise import Dataset, Reader

#  Load trained SVD model 
with open("./outputs/models/svd_model.pkl", "rb") as f:
    model = pickle.load(f)

print(" Loaded SVD model from disk.")

#  Rebuild trainset from ratings 
ratings_df = pd.read_csv("data/processed/user_course_ratings_triplet_2.csv")
ratings_df = ratings_df.dropna(subset=["rating"])
ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce")
ratings_df = ratings_df[ratings_df["rating"] > 0]
ratings_df["rating"] = ratings_df["rating"].clip(upper=5)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[["user_id", "course_id", "rating"]], reader)
trainset = data.build_full_trainset()

print(" Reconstructed trainset from rating data.")

# Generate top-N (e.g., N=10) per user 
def get_top_n(model, trainset, n=10):
    all_users = trainset.all_users()
    all_items = set(trainset.all_items())
    top_n = {}

    for uid_inner in all_users:
        uid_raw = trainset.to_raw_uid(uid_inner)

        rated_items = set(j for (j, _) in trainset.ur[uid_inner])
        unseen_items = all_items - rated_items

        predictions = [
            (trainset.to_raw_iid(iid), model.predict(uid_raw, trainset.to_raw_iid(iid)).est)
            for iid in unseen_items
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n[uid_raw] = predictions[:n]

    return top_n

top_n = get_top_n(model, trainset, n=10)

#  Save results 
recommendations = []
for user_id, recs in top_n.items():
    for course_id, score in recs:
        recommendations.append({
            "user_id": user_id,
            "course_id": course_id,
            "predicted_rating": score
        })

top_n_df = pd.DataFrame(recommendations)
print(top_n_df)

top_n_df.to_csv("data/processed/top_10_recommendations.csv", index=False)
print(" Top-10 recommendations saved to data/processed/top_10_recommendations.csv")
