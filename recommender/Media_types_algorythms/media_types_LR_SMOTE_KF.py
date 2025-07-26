import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score, classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# 1. Load & prepare data
df = pd.read_csv('./data/processed/oulad_media_profiles_refined_balanced.csv')
count_cols = ['course_count', 'reading_count', 'video_count']
df['total_count'] = df[count_cols].sum(axis=1)
for c in count_cols:
    df[c.replace('_count', '_prop')] = df[c] / df['total_count']
df['top_bucket'] = (
    df[['course_prop','reading_prop','video_prop']]
      .idxmax(axis=1)
      .str.replace('_prop','')
)

X = df[count_cols]
y = df['top_bucket']

# 2. Build pipeline: SMOTE + balanced LogisticRegression
pipeline = Pipeline([
    ('smote', SMOTE(
        random_state=42,
        # oversample 'reading' to twice its original count
        sampling_strategy={'reading': y.value_counts()['reading'] * 2}
    )),
    ('lr', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ))
])

# 3. Stratified 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Binary‐F1 scorer for the 'reading' class
reading_f1 = make_scorer(
    f1_score,
    average='binary',
    pos_label='reading',
    zero_division=0
)

# 5. Run cross-val and report
scores = cross_val_score(
    pipeline,
    X, y,
    cv=cv,
    scoring=reading_f1,
    n_jobs=-1
)

print("5-fold Stratified CV Reading F1 scores:", np.round(scores, 3))
print("Mean Reading F1:", np.round(scores.mean(), 3), "±", np.round(scores.std(), 3))
