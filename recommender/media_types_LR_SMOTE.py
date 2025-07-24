import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
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

# 2. Train/test split (stratify to preserve imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build pipeline: SMOTE oversamples 'reading' to 2× its original count
#    You can adjust sampling_strategy={'reading': desired_count}
smote = SMOTE(
    random_state=42,
    sampling_strategy={'reading': y_train.value_counts()['reading'] * 2}
)
clf = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

pipeline = Pipeline([
    ('smote', smote),
    ('lr', clf)
])

# 4. Train & evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("=== SMOTE + LogisticRegression ===")
print(classification_report(y_test, y_pred, digits=3))
print("Reading F1:", f1_score(y_test, y_pred, labels=['reading'], average='macro'))
