import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, classification_report

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

#  Load & prepare data
df = pd.read_csv('./data/processed/oulad_media_profiles_refined_balanced.csv')
count_cols = ['course_count', 'reading_count', 'video_count']
df['total_count'] = df[count_cols].sum(axis=1)
for c in count_cols:
    df[c.replace('_count','_prop')] = df[c] / df['total_count']
df['top_bucket'] = (
    df[['course_prop','reading_prop','video_prop']]
      .idxmax(axis=1)
      .str.replace('_prop','')
)

X = df[count_cols]
y = df['top_bucket']

# Split off a hold-out test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#  Build the SMOTE→Logistic pipeline
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('lr',   LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

#  Compute original reading count on the *training* set
orig_reading = y_train.value_counts().get('reading', 0)

# Set up the grid: oversample reading to 2×, 3×, or 5× its original
param_grid = {
    'smote__sampling_strategy': [
        {'reading': orig_reading * 2},
        {'reading': orig_reading * 3},
        {'reading': orig_reading * 5},
    ],
    'lr__C': [0.01, 0.1, 1, 10]
}

#  Define stratified CV and a binary-F1 scorer for the reading class
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
reading_f1 = make_scorer(
    f1_score,
    average='binary',
    pos_label='reading',
    zero_division=0
)

#  Run GridSearchCV
grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring=reading_f1,
    cv=cv,
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

#  Report best parameters and CV score
print("Best parameters:", grid.best_params_)
print("Best CV Reading F1: {:.3f}".format(grid.best_score_))

# Evaluate on hold-out test set
y_pred = grid.predict(X_test)
print("\nHold-out test set classification report:")
print(classification_report(y_test, y_pred, digits=3))
print("Reading F1 on test:", 
      f1_score(y_test, y_pred, pos_label='reading', zero_division=0))


# inspect feature importance

coefs = grid.best_estimator_['lr'].coef_[0]
for feat, coef in zip(count_cols, coefs):
    print(f"{feat}: {coef:.3f}")

# evalyate macro-F1 across all buckets

from sklearn.metrics import f1_score
y_pred = grid.predict(X_test)
print("Macro F1:", f1_score(y_test, y_pred, average='macro'))
