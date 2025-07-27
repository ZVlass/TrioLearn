# Re-import necessary libraries due to environment reset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# load files
coursera_path = "./data/raw/coursera_course_dataset_v3.csv"
coursera_df = pd.read_csv(coursera_path)

#  Normalize Coursera metadata
coursera_clean = coursera_df.copy()
coursera_clean['Ratings'] = pd.to_numeric(coursera_clean['Ratings'], errors='coerce')
coursera_clean['rating_level'] = pd.cut(
    coursera_clean['Ratings'],
    bins=[0, 3.5, 4.2, 5],
    labels=['low', 'medium', 'high']
)

difficulty_map = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
coursera_clean['difficulty_num'] = coursera_clean['Difficulty'].map(difficulty_map)

coursera_clean['students_enrolled'] = (
    coursera_clean['course_students_enrolled']
    .str.replace(',', '', regex=False)
    .astype(float)
)

min_enroll = coursera_clean['students_enrolled'].min()
max_enroll = coursera_clean['students_enrolled'].max()
coursera_clean['popularity'] = (coursera_clean['students_enrolled'] - min_enroll) / (max_enroll - min_enroll)

# Save normalized Coursera data to interim folder
coursera_clean[['Title', 'Ratings', 'rating_level', 'difficulty_num', 'students_enrolled', 'popularity']].to_csv(
    "./data/interim/coursera_normalized.csv", index=False
)

# Show preview
#print(coursera_clean[['Title', 'Ratings', 'rating_level', 'difficulty_num', 'students_enrolled', 'popularity']].head())
