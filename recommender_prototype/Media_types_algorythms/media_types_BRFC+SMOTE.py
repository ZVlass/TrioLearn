import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, classification_report

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

# Load & prepare data
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

# Compute original reading count on the *training* set
orig_reading = y_train.value_counts().get('reading', 0)

# Define stratified CV and binary-F1 scorer for 'reading' class
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
reading_f1 = make_scorer(
    f1_score,
    average='binary',
    pos_label='reading',
    zero_division=0
)

# Build the SMOTE → BalancedRandomForest pipeline
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', BalancedRandomForestClassifier(
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    ))
])

# Set up the grid: oversample reading ×2, ×3, ×5 and tune RF hyperparameters
param_grid = {
    'smote__sampling_strategy': [
        {'reading': orig_reading * 2},
        {'reading': orig_reading * 3},
        {'reading': orig_reading * 5},
    ],
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 5, 10]
}

# Run GridSearchCV
grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring=reading_f1,
    cv=cv,
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

# Report best parameters and CV score
print("Best parameters:", grid.best_params_)
print("Best CV Reading F1: {:.3f}".format(grid.best_score_))

# Evaluate on hold-out test set
y_pred = grid.predict(X_test)
print("\nHold-out test set classification report:")
print(classification_report(y_test, y_pred, digits=3))
print("Reading F1 on test:", 
      f1_score(y_test, y_pred, pos_label='reading', zero_division=0))

# Evaluate macro-F1
print("Macro F1:", f1_score(y_test, y_pred, average='macro'))
