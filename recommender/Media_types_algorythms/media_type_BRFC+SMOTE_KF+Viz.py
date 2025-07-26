import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

# Load the data
df = pd.read_csv('./data/processed/oulad_media_profiles_refined_balanced.csv')

# Define feature and target engineering
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Calculate original count
orig_reading = y_train.value_counts().get('reading', 0)

# Define pipeline and param grid
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', BalancedRandomForestClassifier(
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    ))
])

param_grid = {
    'smote__sampling_strategy': [
        {'reading': orig_reading * 2},
        {'reading': orig_reading * 3},
        {'reading': orig_reading * 5},
    ],
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 5, 10]
}

reading_f1 = make_scorer(f1_score, average='binary', pos_label='reading', zero_division=0)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring=reading_f1,
    cv=cv,
    n_jobs=-1,
    verbose=0,
    return_train_score=True
)

grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=grid.best_estimator_.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid.best_estimator_.classes_)

# Feature Importance
importances = grid.best_estimator_['rf'].feature_importances_
features = X.columns
feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Confusion Matrix
disp.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title("Confusion Matrix")

# Feature Importances
sns.barplot(data=feat_imp_df, x='Importance', y='Feature', ax=axes[1])
axes[1].set_title("Feature Importances")

# Learning Curve (train vs test F1)
mean_train_scores = [np.mean(scores) for scores in grid.cv_results_['mean_train_score']]
mean_test_scores = [np.mean(scores) for scores in grid.cv_results_['mean_test_score']]
param_labels = [str(p) for p in grid.cv_results_['params']]

axes[2].plot(param_labels, mean_train_scores, label='Train F1', marker='o')
axes[2].plot(param_labels, mean_test_scores, label='CV F1', marker='o')
axes[2].set_xticks(range(len(param_labels)))
axes[2].set_xticklabels(param_labels, rotation=90)
axes[2].legend()
axes[2].set_title("Learning Curve (F1 Reading Class)")
axes[2].set_ylabel("F1 Score")
axes[2].set_xlabel("Parameter Combination")

plt.tight_layout()
plt.show()

