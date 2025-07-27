import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate

# Simulated interactions (replace this with your actual DataFrame)
df_interactions = pd.read_csv("./data/interim/simulated_user_ratings.csv")  

# Load into Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_interactions[['user_id', 'item_id', 'rating']], reader)

# Define user-based CF model
algo = KNNBasic(sim_options={'user_based': True})

# Evaluate with cross-validation
results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(results)
