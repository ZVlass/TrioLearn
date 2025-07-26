import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier


# Load data
df = pd.read_csv('./data/processed/oulad_media_profiles_refined_balanced.csv')

# Recreate features
count_cols = ['course_count', 'reading_count', 'video_count']
df['total_count'] = df[count_cols].sum(axis=1)
for c in count_cols:
    df[c.replace('_count', '_prop')] = df[c] / df['total_count']
df['top_bucket'] = (
    df[['course_prop', 'reading_prop', 'video_prop']]
    .idxmax(axis=1)
    .str.replace('_prop', '')
)

X = df[count_cols]
y = df['top_bucket']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Logistic Regression Pipeline
orig_reading = y_train.value_counts().get('reading', 0)
lr_pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy={'reading': orig_reading * 5}, random_state=42)),
    ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)

# BalancedRandomForest Pipeline (already used)
rf_pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy={'reading': orig_reading * 5}, random_state=42)),
    ('rf', BalancedRandomForestClassifier(n_estimators=200, max_depth=None, class_weight='balanced_subsample', random_state=42, n_jobs=-1))
])
rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)

# Compare classification reports and F1 scores
lr_report = classification_report(y_test, lr_pred, output_dict=True, zero_division=0)
rf_report = classification_report(y_test, rf_pred, output_dict=True, zero_division=0)

# Collect relevant metrics
comparison = pd.DataFrame({
    'Metric': ['Reading Precision', 'Reading Recall', 'Reading F1', 'Macro F1', 'Accuracy'],
    'Logistic Regression': [
        lr_report['reading']['precision'],
        lr_report['reading']['recall'],
        lr_report['reading']['f1-score'],
        f1_score(y_test, lr_pred, average='macro', zero_division=0),
        (lr_pred == y_test).mean()
    ],
    'Balanced Random Forest': [
        rf_report['reading']['precision'],
        rf_report['reading']['recall'],
        rf_report['reading']['f1-score'],
        f1_score(y_test, rf_pred, average='macro', zero_division=0),
        (rf_pred == y_test).mean()
    ]
})

print(comparison.round(3))
