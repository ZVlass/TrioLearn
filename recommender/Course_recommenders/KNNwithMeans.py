import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans
from surprise.model_selection import cross_validate

# Load simulated dataset
df = pd.read_csv("./data/interim/simulated_user_ratings.csv")  

# Prepare the data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Create the KNNWithMeans model
algo = KNNWithMeans(sim_options={'user_based': True})

# Evaluate the model
results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Print summary
print("KNNWithMeans (User-Based) Results:")
for k, v in results.items():
    print(f"{k}: {v}")
