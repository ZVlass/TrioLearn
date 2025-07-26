import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
import joblib

#  Load and Prepare Data 
df = pd.read_csv('./data/processed/oulad_media_profiles_refined_balanced.csv')

# Compute proportions
count_cols = ['course_count', 'reading_count', 'video_count']
df['total_count'] = df[count_cols].sum(axis=1)
for c in count_cols:
    df[c.replace('_count', '_prop')] = df[c] / df['total_count']

# Define top_bucket
df['top_bucket'] = df[['course_prop', 'reading_prop', 'video_prop']].idxmax(axis=1).str.replace('_prop', '')

# Encode target
label_encoder = LabelEncoder()
df['top_bucket_encoded'] = label_encoder.fit_transform(df['top_bucket'])
X = df[count_cols].copy()
y = df['top_bucket_encoded'].astype(int).values.ravel()

# Class info
class_counts = pd.Series(y).value_counts()
num_classes = len(np.unique(y))
print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
print("Class counts:", class_counts.to_dict())

# Compute scale_pos_weight if binary classification and class imbalance exists
if num_classes == 2:
    majority = class_counts.max()
    minority = class_counts.min()
    scale_pos_weight = majority / minority
else:
    scale_pos_weight = 1  # not used for multiclass

#  Model and Parameter Grid 
xgb_clf = XGBClassifier(
    objective='multi:softprob' if num_classes > 2 else 'binary:logistic',
    num_class=num_classes if num_classes > 2 else None,
    use_label_encoder=False,
    eval_metric='mlogloss',
    verbosity=0,
    scale_pos_weight=scale_pos_weight
)

param_dist = {
    'max_depth': randint(3, 8),
    'learning_rate': uniform(0.01, 0.15),
    'n_estimators': randint(50, 150),
    'subsample': uniform(0.8, 0.2),
    'colsample_bytree': uniform(0.8, 0.2)
}

random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Run Randomized Search 
random_search.fit(X, y)

# Best model and parameters
print("\n Best Parameters Found:")
print(random_search.best_params_)
print("\n Best Cross-Validated Accuracy:")
print(f"{random_search.best_score_:.4f}")

#   Evaluate on Holdout 
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\n Classification Report on Holdout Test Set:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

#  Save model and encoder
import os

# Get current file's directory (e.g., the script running this code)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct output directory path relative to project root
output_dir = os.path.join(BASE_DIR, "..", "outputs", "model")
os.makedirs(output_dir, exist_ok=True)

# Save paths
model_path = os.path.join(output_dir, "xgb_best_media_type_model.pkl")
encoder_path = os.path.join(output_dir, "label_encoder.pkl")

# Save artifacts
joblib.dump(best_model, model_path)
joblib.dump(label_encoder, encoder_path)

print(f"\n Model saved to: {os.path.abspath(model_path)}")
print(f"\n LabelEncoder saved to: {os.path.abspath(encoder_path)}")