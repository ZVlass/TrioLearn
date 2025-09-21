from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

#   Define XGBoost parameters
params = {
    'objective': 'multi:softprob',
    'num_class': len(label_encoder.classes_),
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'verbosity': 0
}

# Train with early stopping
evallist = [(dtrain, 'train'), (dvalid, 'eval')]
evals_result = {}
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=evallist,
    early_stopping_rounds=10,
    evals_result=evals_result,
)

# Predict and evaluate
y_pred_probs = bst.predict(dvalid)
y_pred = np.argmax(y_pred_probs, axis=1)

#print(report = classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Plot training vs validation log loss
plt.figure(figsize=(10, 6))
plt.plot(evals_result['train']['mlogloss'], label='Train Log Loss')
plt.plot(evals_result['eval']['mlogloss'], label='Validation Log Loss')
plt.xlabel("Boosting Round")
plt.ylabel("Multiclass Log Loss")
plt.title("XGBoost Training vs Validation Log Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


