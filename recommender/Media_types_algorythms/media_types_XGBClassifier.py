from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd



df = pd.read_csv('./data/processed/oulad_media_profiles_refined_balanced.csv')

# Recreate 'top_bucket' as per original logic
count_cols = ['course_count', 'reading_count', 'video_count']
df['total_count'] = df[count_cols].sum(axis=1)
for c in count_cols:
    df[c.replace('_count', '_prop')] = df[c] / df['total_count']
df['top_bucket'] = df[['course_prop', 'reading_prop', 'video_prop']].idxmax(axis=1).str.replace('_prop', '')

# Encode target
label_encoder = LabelEncoder()
df['top_bucket_encoded'] = label_encoder.fit_transform(df['top_bucket'])

# Prepare data for XGBoost
X = df[count_cols]
y = df['top_bucket_encoded']

# Confirm target is 1D and integer
print("y shape:", y.shape)
print("y dtype:", y.dtype)
print("Unique target classes:", pd.Series(y).value_counts())

# Check for NaNs in features or target
print("NaNs in X:", X.isna().sum().sum())
print("NaNs in y:", pd.Series(y).isna().sum())
print("Any infinite values in X?", np.isinf(X).values.sum() > 0)

num_classes = len(np.unique(y))  # should be 2


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Simple split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Fit a basic model with fixed parameters
model = XGBClassifier(
    objective='multi:softprob' if num_classes > 2 else 'binary:logistic',
    num_class=num_classes if num_classes > 2 else None,
    use_label_encoder=False,
    eval_metric='mlogloss',
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100
)


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
print("Test accuracy:", accuracy_score(y_test, y_pred))
