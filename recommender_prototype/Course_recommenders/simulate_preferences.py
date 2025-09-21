import pandas as pd
import numpy as np
import os

from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# Load course data
file_path = "./data/raw/coursera_course_dataset_v3.csv"
df_courses = pd.read_csv(file_path)

# Drop rows without valid ratings
df_courses = df_courses.dropna(subset=["Ratings"])

# Create synthetic users
n_users = 500
user_ids = [f"user_{i}" for i in range(n_users)]

# Simulate Preferences

# Assume each user interacts with 10 random courses
interactions = []

interactions = []
for user in user_ids:
    sampled_courses = df_courses.sample(n=10, random_state=np.random.randint(1000))
    for _, row in sampled_courses.iterrows():
        try:
            avg_rating = float(row['Ratings'])
        except:
            continue
        # Add Gaussian noise around average rating
        rating = np.random.normal(loc=avg_rating, scale=0.5)
        rating = max(1.0, min(5.0, rating))  # Clamp between 1 and 5
        interactions.append((user, row['Title'], round(rating, 2)))


df_interactions = pd.DataFrame(interactions, columns=["user_id", "item_id", "rating"])

# Preview first few rows
print(df_interactions.head())

# Define the path to save the simulated interaction data
interim_folder = "./data/interim"
os.makedirs(interim_folder, exist_ok=True)

# Save the DataFrame
file_path = os.path.join(interim_folder, "simulated_user_ratings.csv")
df_interactions.to_csv(file_path, index=False)



