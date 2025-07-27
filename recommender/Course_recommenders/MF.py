import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# Load your simulated data
df = pd.read_csv("./data/interim/simulated_user_ratings.csv")  # Adjust path as needed

# Prepare data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Initialize the SVD model
model = SVD()

# Evaluate using 5-fold cross-validation
results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Print results
print("SVD MF Results:")
for k, v in results.items():
    print(f"{k}: {v}")

