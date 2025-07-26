import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import uniform, randint
import joblib
import os

#  Load and Prepare Data 
df = pd.read_csv('./data/processed/oulad_media_profiles_refined_balanced.csv')

count_cols = ['course_count', 'reading_count', 'video_count']
df['total_count'] = df[count_cols].sum(axis=1)
for c in count_cols:
    df[c.replace('_count', '_prop')] = df[c] / df['total_count']

df['top_bucket'] = df[['course_prop', 'reading_prop', 'video_prop']].idxmax(axis=1).str.replace('_prop', '')
label_encoder = LabelEncoder()
df['top_bucket_encoded'] = label_encoder.fit_transform(df['top_bucket'])

X = df[count_cols].copy()
y = df['top_bucket_encoded'].astype(int).values.ravel()
num_classes = len(np.unique(y))

# Define SMOTE + XGBoost Pipeline
xgb = XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    verbosity=0
)

pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy='auto', random_state=42)),
    ('xgb', xgb)
])

#  Hyperparameter Space 
param_dist = {
    'xgb__max_depth': randint(3, 8),
    'xgb__learning_rate': uniform(0.01, 0.15),
    'xgb__n_estimators': randint(50, 150),
    'xgb__subsample': uniform(0.8, 0.2),
    'xgb__colsample_bytree': uniform(0.8, 0.2)
}

# --------------------- Randomized Search --------------------- #
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,
    scoring='f1_macro',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

search.fit(X, y)

#  Report Best Results 
print("\n Best Parameters (SMOTE + XGBoost):")
print(search.best_params_)
print("\n Best Macro F1 (CV):", search.best_score_)

#  Evaluate on Holdout 
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
best_pipeline = search.best_estimator_
best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

print("\n Classification Report (Holdout):")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save Model and Encoder  #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(BASE_DIR, "..", "outputs", "model")
os.makedirs(output_dir, exist_ok=True)

joblib.dump(best_pipeline, os.path.join(output_dir, "xgb_smote_model.pkl"))
joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))
print("\n Saved pipeline and encoder to:", output_dir)
